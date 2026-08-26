import gc
import threading
from pathlib import Path

import torch
from peft import PeftModel
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

        self.model_configs = {
            "opt125m": {
                "display_name": "OPT-125M",
                "base_model": "facebook/opt-125m",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-125m-news-simplifier"
                )
            },

            "opt13b": {
                "display_name": "OPT-1.3B",
                "base_model": "facebook/opt-1.3b",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-1.3b-news-simplifier"
                )
            },

            "opt27b": {
                "display_name": "OPT-2.7B",
                "base_model": "facebook/opt-2.7b",
                "adapter_path": (
                    self.project_root
                    / "models"
                    / "opt-2.7b-news-simplifier"
                )
            }
        }

        self.current_model_key = None
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
                "installed": config["adapter_path"].exists()
            }
            for key, config in self.model_configs.items()
        }

    def is_valid_model(self, model_key: str) -> bool:
        return model_key in self.model_configs

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
        """
        Generate with the selected fine-tuned OPT model.

        do_sample=False intentionally matches the deterministic generation
        setup used by the evaluation notebook.
        """

        with self._generation_lock:
            model, tokenizer = self.load_model(
                model_key
            )

            prompt = (
                self.prompt_service
                .fit_prompt_to_context(
                    tokenizer=tokenizer,
                    article=article,
                    reserved_tokens=self.MAX_NEW_TOKENS
                )
            )

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

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_ids = output_ids[0][
                input_length:
            ]

            result = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            return result.strip()

    def load_model(self, model_key: str):
        self._require_valid_model_key(
            model_key
        )

        if (
            self.current_model_key == model_key
            and self.current_model is not None
            and self.current_tokenizer is not None
        ):
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

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"{config['display_name']} adapter was not found at: "
                f"{adapter_path}"
            )

        base_model_name = config[
            "base_model"
        ]

        print(
            f"Loading {config['display_name']} "
            f"from {adapter_path}"
        )

        tokenizer = self._load_tokenizer(
            base_model_name=base_model_name,
            adapter_path=adapter_path
        )

        base_model = self._load_base_model(
            base_model_name
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path)
        )

        model.eval()

        self.current_model_key = model_key
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
