from flask import Flask, jsonify, render_template, request

from services.model_manager import ModelManager
from services.output_parser import OutputParser
from services.preprocessor import NewsPreprocessor
from services.prompt_service import PromptService


app = Flask(__name__)

# Services are created once and reused by Flask requests.
preprocessor = NewsPreprocessor()
prompt_service = PromptService()
model_manager = ModelManager(prompt_service=prompt_service)
output_parser = OutputParser()


@app.route("/")
def home():
    return render_template(
        "index.html",
        models=model_manager.get_available_models()
    )


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    raw_article = data.get("article", "")
    model_key = str(data.get("model", "")).strip()

    try:
        # Apply the same article cleaning used during dataset preprocessing.
        article = preprocessor.clean(raw_article)

        if not article:
            return jsonify({
                "success": False,
                "error": "Please enter a news article."
            }), 400

        word_count = preprocessor.count_words(article)

        # The source training dataset only retained articles below 500 words.
        if not preprocessor.is_valid_length(article):
            return jsonify({
                "success": False,
                "error": (
                    f"The article contains {word_count} words. "
                    "This system was trained using articles below 500 words. "
                    "Please provide a shorter article."
                )
            }), 400

        if not model_manager.is_valid_model(model_key):
            return jsonify({
                "success": False,
                "error": "Invalid model selected."
            }), 400

        raw_result = model_manager.generate(
            model_key=model_key,
            article=article
        )

        parsed_result = output_parser.parse(raw_result)

        return jsonify({
            "success": True,
            "model": model_manager.get_display_name(model_key),

            # Kept for compatibility with the earlier HTML/JavaScript.
            "result": raw_result,

            # Recommended structured response for the new grouped UI.
            "summary": parsed_result["summary"],
            "aspects": parsed_result["aspects"],

            "word_count": word_count
        })

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
