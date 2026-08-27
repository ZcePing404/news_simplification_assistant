import os

import requests


class RemoteInferenceError(RuntimeError):
    """Raised when the central inference service cannot serve a request."""


class RemoteModelClient:
    MODEL_NAMES = {
        "opt125m_base": "Original OPT-125M",
        "opt125m_news": "Fine-tuned OPT-125M",
        "opt13b_base": "Original OPT-1.3B",
        "opt13b_news": "Fine-tuned OPT-1.3B",
        "opt27b_base": "Original OPT-2.7B",
        "opt27b_news": "Fine-tuned OPT-2.7B"
    }

    def __init__(self, endpoint_url=None, api_token=None):
        self.endpoint_url = str(
            endpoint_url
            or os.environ.get("NEWS_INFERENCE_API_URL", "")
        ).strip()
        self.api_token = str(
            api_token
            or os.environ.get("NEWS_INFERENCE_API_TOKEN", "")
        ).strip()
        self.session = requests.Session()

    def get_available_models(self) -> dict:
        return {
            key: {
                "display_name": display_name,
                "installed": True
            }
            for key, display_name in self.MODEL_NAMES.items()
        }

    def is_valid_model(self, model_key: str) -> bool:
        return model_key in self.MODEL_NAMES

    def generate(self, model_key: str, article: str) -> dict:
        if not self.endpoint_url:
            raise RemoteInferenceError(
                "NEWS_INFERENCE_API_URL is not configured."
            )

        if not self.api_token:
            raise RemoteInferenceError(
                "NEWS_INFERENCE_API_TOKEN is not configured."
            )

        if not self.is_valid_model(model_key):
            raise ValueError(f"Unknown model key: {model_key}")

        try:
            response = self.session.post(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.api_token}"
                },
                json={
                    "model": model_key,
                    "article": article
                },
                timeout=(15, 600)
            )
        except requests.RequestException as error:
            raise RemoteInferenceError(
                "Could not reach the central inference service: "
                f"{error}"
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise RemoteInferenceError(
                "The central inference service returned a non-JSON "
                f"response with HTTP {response.status_code}."
            ) from error

        if not response.ok or not payload.get("success"):
            message = payload.get("detail") or payload.get("error")

            raise RemoteInferenceError(
                str(message)
                if message
                else (
                    "The central inference service failed with HTTP "
                    f"{response.status_code}."
                )
            )

        return payload
