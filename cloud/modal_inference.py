import hmac
import os
from pathlib import Path

import modal
from fastapi import HTTPException, Request, status


APP_NAME = "news-simplification-inference"
ADAPTER_REPO = "hehehehe6625/news-simplication"
MODEL_FAMILIES = {
    "opt125m": {
        "base_model": "facebook/opt-125m",
        "adapter_subfolder": "opt-125m"
    },
    "opt13b": {
        "base_model": "facebook/opt-1.3b",
        "adapter_subfolder": "opt-1.3b"
    },
    "opt27b": {
        "base_model": "facebook/opt-2.7b",
        "adapter_subfolder": "opt-2.7b"
    }
}
MODEL_ROUTES = {
    "opt125m_base": {
        "family": "opt125m",
        "use_adapter": False,
        "display_name": "Original OPT-125M"
    },
    "opt125m_news": {
        "family": "opt125m",
        "use_adapter": True,
        "display_name": "Fine-tuned OPT-125M"
    },
    "opt13b_base": {
        "family": "opt13b",
        "use_adapter": False,
        "display_name": "Original OPT-1.3B"
    },
    "opt13b_news": {
        "family": "opt13b",
        "use_adapter": True,
        "display_name": "Fine-tuned OPT-1.3B"
    },
    "opt27b_base": {
        "family": "opt27b",
        "use_adapter": False,
        "display_name": "Original OPT-2.7B"
    },
    "opt27b_news": {
        "family": "opt27b",
        "use_adapter": True,
        "display_name": "Fine-tuned OPT-2.7B"
    }
}
SUPPORTED_MODELS = {
    key: route["display_name"]
    for key, route in MODEL_ROUTES.items()
}
NEWS_MODEL_KEY = "opt27b_news"

MAX_CONTEXT = 2048
MAX_NEW_TOKENS = 384

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_PATH = Path("/cache/huggingface")

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "transformers==4.57.6",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "bitsandbytes==0.50.0",
        "huggingface_hub",
        "numpy",
        "scikit-learn",
        "fastapi[standard]"
    )
    .add_local_dir(
        LOCAL_PROJECT_ROOT / "services",
        "/root/services",
        copy=True
    )
    .add_local_file(
        LOCAL_PROJECT_ROOT
        / "data"
        / "processed"
        / "prompt_templates.json",
        "/root/data/processed/prompt_templates.json",
        copy=True
    )
    .add_local_file(
        LOCAL_PROJECT_ROOT
        / "data"
        / "processed"
        / "stage1_train.jsonl",
        "/root/data/processed/stage1_train.jsonl",
        copy=True
    )
    .add_local_file(
        LOCAL_PROJECT_ROOT
        / "data"
        / "processed"
        / "train.jsonl",
        "/root/data/processed/train.jsonl",
        copy=True
    )
)

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(
    "news-simplification-model-cache",
    create_if_missing=True
)


def _download_snapshots() -> dict:
    """Download every base model and adapter into the persistent cache."""
    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]

    adapter_snapshot = snapshot_download(
        repo_id=ADAPTER_REPO,
        token=token,
        cache_dir=MODEL_CACHE_PATH
    )

    snapshots = {}
    missing = []

    for family, config in MODEL_FAMILIES.items():
        base_snapshot = snapshot_download(
            repo_id=config["base_model"],
            token=token,
            cache_dir=MODEL_CACHE_PATH
        )
        adapter_path = (
            Path(adapter_snapshot) / config["adapter_subfolder"]
        )

        required_files = (
            adapter_path / "adapter_config.json",
            adapter_path / "adapter_model.safetensors"
        )
        missing.extend(
            str(path) for path in required_files if not path.is_file()
        )
        snapshots[family] = {
            "base_snapshot": base_snapshot,
            "adapter_path": str(adapter_path)
        }

    if missing:
        raise FileNotFoundError(
            "A Hugging Face adapter subfolder is incomplete. Missing: "
            + ", ".join(missing)
        )

    return {
        "adapter_snapshot": adapter_snapshot,
        "models": snapshots
    }


@app.function(
    image=runtime_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_cache},
    timeout=3600
)
def download_models() -> dict:
    """Warm the persistent cache without reserving a GPU."""
    result = _download_snapshots()
    model_cache.commit()
    return result


@app.cls(
    image=runtime_image,
    gpu="T4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("news-api-secret")
    ],
    volumes={"/cache": model_cache},
    timeout=1800,
    scaledown_window=600,
    max_containers=1
)
class NewsInference:
    @modal.enter()
    def load_model(self):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )

        from services.output_parser import OutputParser
        from services.output_validator import OutputValidator
        from services.preprocessor import NewsPreprocessor
        from services.prompt_service import PromptService

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        snapshots = _download_snapshots()["models"]
        self.model_bundles = {}

        for family, config in MODEL_FAMILIES.items():
            snapshot = snapshots[family]
            adapter_path = Path(snapshot["adapter_path"])
            adapter_config = PeftConfig.from_pretrained(str(adapter_path))

            if (
                adapter_config.base_model_name_or_path
                != config["base_model"]
            ):
                raise ValueError(
                    f"Adapter base mismatch for {family}: expected "
                    f"{config['base_model']}, found "
                    f"{adapter_config.base_model_name_or_path}."
                )

            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                snapshot["base_snapshot"],
                quantization_config=quantization_config,
                device_map={"": 0}
            )
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_path),
                adapter_name="news"
            )
            model.eval()
            model.config.use_cache = True

            self.model_bundles[family] = {
                "model": model,
                "tokenizer": tokenizer
            }

        self.torch = torch
        self.preprocessor = NewsPreprocessor()
        self.prompt_service = PromptService()
        self.output_parser = OutputParser()
        self.output_validator = OutputValidator()

        model_cache.commit()

    def _generate_attempt(
        self,
        article: str,
        model_key: str,
        corrective: bool
    ) -> dict:
        route = MODEL_ROUTES[model_key]
        bundle = self.model_bundles[route["family"]]
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]

        prompt_details = self.prompt_service.fit_prompt_to_context(
            tokenizer=tokenizer,
            article=article,
            reserved_tokens=MAX_NEW_TOKENS,
            corrective=corrective,
            return_details=True
        )
        inputs = tokenizer(
            prompt_details["prompt"],
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt"
        )
        input_device = next(
            parameter.device
            for parameter in model.parameters()
            if parameter.device.type != "meta"
        )
        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[1]

        def run_generation():
            with self.torch.inference_mode():
                return model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        if not route["use_adapter"]:
            with model.disable_adapter():
                output_ids = run_generation()
        else:
            model.set_adapter("news")
            output_ids = run_generation()

        generated_ids = output_ids[0][input_length:]
        raw_output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        return {
            "raw_output": raw_output,
            "input_token_length": prompt_details["input_token_length"],
            "generated_token_count": int(generated_ids.shape[0]),
            "article_was_truncated": prompt_details[
                "article_was_truncated"
            ],
            "source_keywords": (
                self.prompt_service.extract_source_keywords(article)
            ),
            "corrective": corrective
        }

    def _run_workflow(self, payload: dict) -> dict:
        raw_article = payload.get("article", "")
        model_key = str(payload.get("model", "")).strip()

        if model_key not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model selected."
            )

        article = self.preprocessor.clean(raw_article)

        if not article:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please enter a news article."
            )

        word_count = self.preprocessor.count_words(article)
        warnings = []

        if not self.preprocessor.is_valid_length(article):
            warnings.append(
                f"This article contains {word_count} words and is longer "
                "than the articles used for training. Generation will "
                "continue, but quality may be lower and content near the "
                "end may be truncated to fit the model context."
            )

        attempts = []

        for corrective in (False, True):
            generation = self._generate_attempt(
                article=article,
                model_key=model_key,
                corrective=corrective
            )
            parsed = self.output_parser.parse(generation["raw_output"])
            quality = self.output_validator.evaluate(
                raw_output=generation["raw_output"],
                article=article,
                summary=parsed["summary"]
            )
            attempts.append({
                "generation": generation,
                "parsed": parsed,
                "quality": quality
            })

            print(
                "Output validation:",
                f"model={model_key}",
                f"attempt={len(attempts)}",
                f"corrective={corrective}",
                f"issue_detected={quality['validation_issue_detected']}",
                f"passed={quality['is_valid']}",
                "copied_source_sentence_detected="
                f"{quality['copied_source_sentence_detected']}",
                f"copied_sentence_count={quality['copied_sentence_count']}",
                f"issues={quality['issues'] or ['none']}"
            )

            if quality["is_valid"]:
                break

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

        return {
            "success": True,
            "model": SUPPORTED_MODELS[model_key],
            "result": selected["generation"]["raw_output"],
            "summary": selected["parsed"]["summary"],
            "aspects": selected["parsed"]["aspects"],
            "word_count": word_count,
            "warning": " ".join(warnings) if warnings else None,
            "retry_used": retry_used,
            "generation_diagnostics": {
                "selected_attempt": attempts.index(selected) + 1,
                "attempts": diagnostic_attempts
            }
        }

    @modal.method()
    def smoke_test(self, model_key: str = NEWS_MODEL_KEY) -> dict:
        """Run a short authenticated test through the Modal CLI."""
        result = self._run_workflow({
            "model": model_key,
            "article": (
                "Heavy rain damaged a bridge in Kuala Lumpur on Monday. "
                "Officials closed the bridge while engineers inspected "
                "the damage. No injuries were reported, and drivers were "
                "asked to use another route."
            )
        })

        return {
            "model": result["model"],
            "summary": result["summary"],
            "aspects": result["aspects"],
            "warning": result["warning"],
            "retry_used": result["retry_used"],
            "generation_diagnostics": result[
                "generation_diagnostics"
            ]
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def generate(self, request: Request):
        expected_token = os.environ["API_TOKEN"]
        authorization = request.headers.get("Authorization", "")
        scheme, _, supplied_token = authorization.partition(" ")

        if (
            scheme.lower() != "bearer"
            or not supplied_token
            or not hmac.compare_digest(supplied_token, expected_token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API token."
            )

        try:
            payload = await request.json()
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be valid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be a JSON object."
            )

        return self._run_workflow(payload)


@app.local_entrypoint()
def test_model(model: str = NEWS_MODEL_KEY):
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            "model must be one of: "
            + ", ".join(SUPPORTED_MODELS)
        )

    print(NewsInference().smoke_test.remote(model))
