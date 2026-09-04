import logging

from flask import Flask, jsonify, render_template, request

from services.model_manager import ModelManager
from services.output_parser import OutputParser
from services.preprocessor import NewsPreprocessor
from services.prompt_service import PromptService

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Local inference services. ModelManager loads only the selected model and
# unloads the previous one so the application stays within laptop VRAM.
preprocessor = NewsPreprocessor()
prompt_service = PromptService()
output_parser = OutputParser()
model_manager = ModelManager(prompt_service)


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
        article = preprocessor.clean(raw_article)

        if not article:
            return jsonify({
                "success": False,
                "error": "Please enter a news article."
            }), 400

        if not model_manager.is_valid_model(model_key):
            return jsonify({
                "success": False,
                "error": "Invalid model selected."
            }), 400

        word_count = preprocessor.count_words(article)
        warnings = []

        if not preprocessor.is_valid_length(article):
            warnings.append(
                f"This article contains {word_count} words and is longer "
                "than the articles used for training. Generation will "
                "continue, but quality may be lower and content near the "
                "end may be truncated to fit the model context."
            )

        generation = model_manager.generate_attempt(
            model_key=model_key,
            article=article,
            corrective=False
        )
        parsed = output_parser.parse(generation["raw_output"])

        result = {
            "success": True,
            "model": model_manager.get_display_name(model_key),
            "result": generation["raw_output"],
            "generated_text": generation["raw_output"],
            "summary": parsed["summary"],
            "aspects": parsed["aspects"],
            "word_count": word_count,
            "warning": " ".join(warnings) if warnings else None,
            "retry_used": False,
            "generation_diagnostics": {
                "selected_attempt": 1,
                "attempts": [{
                    "corrective": False,
                    "input_token_length": generation[
                        "input_token_length"
                    ],
                    "generated_token_count": generation[
                        "generated_token_count"
                    ],
                    "article_was_truncated": generation[
                        "article_was_truncated"
                    ],
                    "source_keywords": generation["source_keywords"]
                }]
            }
        }

        app.logger.info(
            "Local generation completed: model=%s words=%s retry=%s "
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
