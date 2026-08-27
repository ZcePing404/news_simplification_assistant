import logging

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from services.remote_model_client import (
    RemoteInferenceError,
    RemoteModelClient
)


load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# The laptop application is now a thin client. Model loading, prompting,
# generation, validation, and retry run on the central Modal GPU endpoint.
model_client = RemoteModelClient()


@app.route("/")
def home():
    return render_template(
        "index.html",
        models=model_client.get_available_models()
    )


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    raw_article = data.get("article", "")
    model_key = str(data.get("model", "")).strip()

    try:
        article = str(raw_article).strip()

        if not article:
            return jsonify({
                "success": False,
                "error": "Please enter a news article."
            }), 400

        if not model_client.is_valid_model(model_key):
            return jsonify({
                "success": False,
                "error": "Invalid model selected."
            }), 400

        result = model_client.generate(
            model_key=model_key,
            article=article
        )

        app.logger.info(
            "Remote generation completed: model=%s words=%s retry=%s "
            "diagnostics=%s",
            model_key,
            result.get("word_count"),
            result.get("retry_used"),
            result.get("generation_diagnostics")
        )

        return jsonify(result)

    except FileNotFoundError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except RemoteInferenceError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 502

    except Exception as error:
        app.logger.exception("Generation failed")

        return jsonify({
            "success": False,
            "error": f"Generation failed: {error}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
