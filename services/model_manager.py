import gc
import threading
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


class ModelManager:
    """
    Owns OPT model configuration, model switching, LoRA adapter loading,
    and deterministic text generation.

    Only one model is kept loaded at a time to reduce RAM/VRAM usage.
    """

    MAX_NEW_TOKENS = 384

    def __init__(self, prompt_service):
        self.project_root = Path(__file__).resolve().parents[1]
        self.prompt_service = prompt_service

        model_families = {
            "opt125m": {
                "base_model": "facebook/opt-125m",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-125m-news-simplifier"
                )
            },

            "opt13b": {
                "base_model": "facebook/opt-1.3b",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-1.3b-news-model"
                )
            },

            "opt27b": {
                "base_model": "facebook/opt-2.7b",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-2.7b-news-model"
                )
            }
        }

        self.model_configs = {}

        for family, config in model_families.items():
            size = {
                "opt125m": "125M",
                "opt13b": "1.3B",
                "opt27b": "2.7B"
            }[family]

            self.model_configs[f"{family}_base"] = {
                **config,
                "family": family,
                "display_name": f"Original OPT-{size}",
                "use_adapter": False
            }
            self.model_configs[f"{family}_news"] = {
                **config,
                "family": family,
                "display_name": f"Fine-tuned OPT-{size}",
                "use_adapter": True
            }

        self.current_model_key = None
        self.current_model_family = None
        self.current_model = None
        self.current_tokenizer = None

        # Prevent two local Flask requests from trying to switch models
        # at the same time.
        self._generation_lock = threading.Lock()

    def get_available_models(self) -> dict:
        """
        Returns template-friendly model information without exposing
        Path objects.
        """
        return {
            key: {
                "display_name": config["display_name"],
                "installed": True
            }
            for key, config in self.model_configs.items()
            if self._model_is_available(config)
        }

    def is_valid_model(self, model_key: str) -> bool:
        return (
            model_key in self.model_configs
            and self._model_is_available(self.model_configs[model_key])
        )

    def get_display_name(self, model_key: str) -> str:
        self._require_valid_model_key(model_key)

        return self.model_configs[
            model_key
        ]["display_name"]

    def generate(
        self,
        model_key: str,
        article: str
    ) -> str:
        return self.generate_attempt(
            model_key=model_key,
            article=article,
            corrective=False
        )["raw_output"]

    def generate_attempt(
        self,
        model_key: str,
        article: str,
        corrective: bool = False
    ) -> dict:
        """
        Generate with the selected fine-tuned OPT model.

        do_sample=False intentionally matches the deterministic generation
        setup used by the evaluation notebook.
        """

        with self._generation_lock:
            model, tokenizer = self.load_model(
                model_key
            )

            prompt_details = (
                self.prompt_service
                .fit_prompt_to_context(
                    tokenizer=tokenizer,
                    article=article,
                    reserved_tokens=self.MAX_NEW_TOKENS,
                    corrective=corrective,
                    return_details=True
                )
            )

            prompt = prompt_details["prompt"]

            inputs = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt"
            )

            input_device = self._get_input_device(
                model
            )

            inputs = {
                key: value.to(input_device)
                for key, value in inputs.items()
            }

            input_length = (
                inputs["input_ids"].shape[1]
            )

            def run_generation():
                with torch.inference_mode():
                    return model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

            config = self.model_configs[model_key]

            if config["use_adapter"]:
                output_ids = run_generation()
            elif isinstance(model, PeftModel):
                with model.disable_adapter():
                    output_ids = run_generation()
            else:
                output_ids = run_generation()

            generated_ids = output_ids[0][
                input_length:
            ]

            result = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            return {
                "raw_output": result.strip(),
                "input_token_length": prompt_details[
                    "input_token_length"
                ],
                "generated_token_count": int(
                    generated_ids.shape[0]
                ),
                "article_was_truncated": prompt_details[
                    "article_was_truncated"
                ],
                "source_keywords": (
                    self.prompt_service
                    .extract_source_keywords(article)
                ),
                "corrective": corrective
            }

    def load_model(self, model_key: str):
        self._require_valid_model_key(
            model_key
        )

        if (
            self.current_model_family
            == self.model_configs[model_key]["family"]
            and self.current_model is not None
            and self.current_tokenizer is not None
        ):
            self.current_model_key = model_key
            return (
                self.current_model,
                self.current_tokenizer
            )

        self.unload_model()

        config = self.model_configs[
            model_key
        ]

        adapter_path = config[
            "adapter_path"
        ]

        if not self._model_is_available(config):
            raise FileNotFoundError(
                f"{config['display_name']} adapter files were not found at: "
                f"{adapter_path}. Expected adapter_config.json and "
                "adapter_model.safetensors."
            )

        adapter_installed = self._adapter_is_installed(adapter_path)
        base_model_name = config["base_model"]

        if adapter_installed:
            adapter_config = PeftConfig.from_pretrained(
                str(adapter_path)
            )
            adapter_base_model = adapter_config.base_model_name_or_path

            if not adapter_base_model:
                raise ValueError(
                    f"{config['display_name']} adapter does not identify "
                    "its base model."
                )

            if adapter_base_model != base_model_name:
                raise ValueError(
                    f"{config['display_name']} adapter expects base model "
                    f"'{adapter_base_model}', but the application is "
                    f"configured for '{base_model_name}'."
                )

        print(
            f"Loading local {base_model_name} on the GPU"
        )

        tokenizer = self._load_tokenizer(
            base_model_name=base_model_name,
            adapter_path=adapter_path
        )

        base_model = self._load_base_model(
            base_model_name
        )

        if adapter_installed:
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_path)
            )
        else:
            model = base_model

        model.eval()

        self.current_model_key = model_key
        self.current_model_family = config["family"]
        self.current_model = model
        self.current_tokenizer = tokenizer

        print(
            f"Loaded {config['display_name']} successfully."
        )

        return model, tokenizer

    def unload_model(self):
        if self.current_model is not None:
            del self.current_model

        if self.current_tokenizer is not None:
            del self.current_tokenizer

        self.current_model = None
        self.current_tokenizer = None
        self.current_model_key = None
        self.current_model_family = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_tokenizer(
        self,
        base_model_name: str,
        adapter_path: Path
    ):
        """
        Prefer tokenizer files saved alongside the adapter. If they are
        unavailable, fall back to the original Hugging Face base model.
        """

        tokenizer_source = (
            str(adapter_path)
            if (adapter_path / "tokenizer_config.json").exists()
            else base_model_name
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        return tokenizer

    @staticmethod
    def _load_base_model(base_model_name: str):
        if torch.cuda.is_available():
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True
                )
            )

            return (
                AutoModelForCausalLM
                .from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map={"": 0}
                )
            )

        # CPU fallback. This avoids requiring bitsandbytes when CUDA
        # is unavailable, though large models will be much slower.
        return (
            AutoModelForCausalLM
            .from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        )

    @staticmethod
    def _get_input_device(model):
        """
        Find the device on which input tensors should be placed.
        """
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device

        return torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    def _require_valid_model_key(
        self,
        model_key: str
    ):
        if model_key not in self.model_configs:
            raise ValueError(
                f"Unknown model key: {model_key}"
            )

    @staticmethod
    def _adapter_is_installed(adapter_path: Path) -> bool:
        return (
            adapter_path.is_dir()
            and (adapter_path / "adapter_config.json").is_file()
            and (adapter_path / "adapter_model.safetensors").is_file()
        )

    def _model_is_available(self, config: dict) -> bool:
        return (
            not config["use_adapter"]
            or self._adapter_is_installed(config["adapter_path"])
        )
