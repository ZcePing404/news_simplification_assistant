import logging

from flask import Flask, jsonify, render_template, request

from services.model_manager import ModelManager
from services.output_parser import OutputParser
from services.output_validator import OutputValidator
from services.preprocessor import NewsPreprocessor
from services.prompt_service import PromptService


app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Services are created once and reused by Flask requests.
preprocessor = NewsPreprocessor()
prompt_service = PromptService()
model_manager = ModelManager(prompt_service=prompt_service)
output_parser = OutputParser()
output_validator = OutputValidator()


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

        warnings = []

        # Longer articles are accepted, but they are outside the training
        # length distribution and may be shortened to fit OPT's context.
        if not preprocessor.is_valid_length(article):
            warnings.append(
                f"This article contains {word_count} words and is longer "
                "than the articles used for training. Generation will "
                "continue, but quality may be lower and content near the "
                "end may be truncated to fit the model context."
            )

        if not model_manager.is_valid_model(model_key):
            return jsonify({
                "success": False,
                "error": "Invalid model selected."
            }), 400

        first_attempt = model_manager.generate_attempt(
            model_key=model_key,
            article=article
        )

        first_parsed = output_parser.parse(
            first_attempt["raw_output"]
        )
        first_quality = output_validator.evaluate(
            raw_output=first_attempt["raw_output"],
            article=article,
            summary=first_parsed["summary"]
        )

        attempts = [
            {
                "generation": first_attempt,
                "parsed": first_parsed,
                "quality": first_quality
            }
        ]

        if not first_quality["is_valid"]:
            corrective_attempt = model_manager.generate_attempt(
                model_key=model_key,
                article=article,
                corrective=True
            )
            corrective_parsed = output_parser.parse(
                corrective_attempt["raw_output"]
            )
            corrective_quality = output_validator.evaluate(
                raw_output=corrective_attempt["raw_output"],
                article=article,
                summary=corrective_parsed["summary"]
            )
            attempts.append({
                "generation": corrective_attempt,
                "parsed": corrective_parsed,
                "quality": corrective_quality
            })

        selected = max(
            attempts,
            key=lambda attempt: (
                attempt["quality"]["is_valid"],
                attempt["quality"]["quality_score"]
            )
        )

        retry_used = len(attempts) > 1
        selected_quality = selected["quality"]

        if retry_used and selected_quality["is_valid"]:
            warnings.append(
                "The first response failed the quality checks, so a "
                "corrective retry was used."
            )

        if not selected_quality["is_valid"]:
            warnings.append(
                "The model could not produce a fully reliable simplified "
                "response after the quality checks. Review the output "
                "carefully because "
                + "; ".join(selected_quality["issues"])
                + "."
            )

        selected_generation = selected["generation"]
        parsed_result = selected["parsed"]
        raw_result = selected_generation["raw_output"]

        diagnostic_attempts = [
            {
                **attempt["quality"],
                "corrective": attempt["generation"]["corrective"],
                "input_token_length": attempt["generation"][
                    "input_token_length"
                ],
                "generated_token_count": attempt["generation"][
                    "generated_token_count"
                ],
                "article_was_truncated": attempt["generation"][
                    "article_was_truncated"
                ],
                "source_keywords": attempt["generation"][
                    "source_keywords"
                ]
            }
            for attempt in attempts
        ]

        for attempt_number, diagnostic in enumerate(
            diagnostic_attempts,
            start=1
        ):
            app.logger.info(
                "Output validation: attempt=%s corrective=%s "
                "issue_detected=%s passed=%s "
                "copied_source_sentence_detected=%s "
                "copied_sentence_count=%s issues=%s",
                attempt_number,
                diagnostic["corrective"],
                diagnostic["validation_issue_detected"],
                diagnostic["is_valid"],
                diagnostic["copied_source_sentence_detected"],
                diagnostic["copied_sentence_count"],
                diagnostic["issues"] or ["none"]
            )

        app.logger.info(
            "Generation diagnostics: model=%s words=%s retry=%s "
            "selected_attempt=%s attempts=%s",
            model_key,
            word_count,
            retry_used,
            attempts.index(selected) + 1,
            diagnostic_attempts
        )

        return jsonify({
            "success": True,
            "model": model_manager.get_display_name(model_key),

            # Kept for compatibility with the earlier HTML/JavaScript.
            "result": raw_result,

            # Recommended structured response for the new grouped UI.
            "summary": parsed_result["summary"],
            "aspects": parsed_result["aspects"],

            "word_count": word_count,
            "warning": " ".join(warnings) if warnings else None,
            "retry_used": retry_used,
            "generation_diagnostics": {
                "selected_attempt": attempts.index(selected) + 1,
                "attempts": diagnostic_attempts
            }
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
