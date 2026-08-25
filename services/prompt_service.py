import json
from pathlib import Path


class PromptService:
    """
    Loads the exact prompt template used by the OPT notebooks and
    prepares prompts while respecting the 2048-token context window.
    """

    MAX_CONTEXT = 2048
    DEFAULT_RESERVED_TOKENS = 384
    SAFETY_MARGIN = 8

    def __init__(self, template_path=None):
        project_root = Path(__file__).resolve().parents[1]

        self.template_path = Path(
            template_path
            or project_root
            / "data"
            / "processed"
            / "prompt_templates.json"
        )

        self.prompt_template = self._load_template()

    def _load_template(self) -> str:
        if not self.template_path.exists():
            raise FileNotFoundError(
                "Prompt template file was not found at: "
                f"{self.template_path}"
            )

        with self.template_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        template = data.get("prompt_template")

        if not isinstance(template, str) or not template.strip():
            raise ValueError(
                "'prompt_template' is missing or empty in "
                f"{self.template_path}"
            )

        if "{article}" not in template:
            raise ValueError(
                "The prompt template must contain the "
                "'{article}' placeholder."
            )

        return template

    def build_prompt(self, article: str) -> str:
        return self.prompt_template.format(
            article=str(article).strip()
        )

    @staticmethod
    def token_length(
        tokenizer,
        text: str,
        add_special_tokens: bool = True
    ) -> int:
        return len(
            tokenizer(
                str(text),
                add_special_tokens=add_special_tokens,
                truncation=False
            )["input_ids"]
        )

    def fit_prompt_to_context(
        self,
        tokenizer,
        article: str,
        reserved_tokens: int = None
    ) -> str:
        """
        Mirrors the context-fitting approach used in the OPT notebook:
        reserve output tokens, determine the article budget, and trim
        only as much article text as required to keep the prompt within
        the model context.
        """

        if reserved_tokens is None:
            reserved_tokens = self.DEFAULT_RESERVED_TOKENS

        if reserved_tokens <= 0:
            raise ValueError(
                "reserved_tokens must be greater than zero."
            )

        input_limit = self.MAX_CONTEXT - reserved_tokens

        if input_limit <= 0:
            raise ValueError(
                "Reserved output tokens leave no room for input."
            )

        article = str(article).strip()

        article_ids = tokenizer(
            article,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        empty_prompt_tokens = self.token_length(
            tokenizer,
            self.build_prompt(""),
            add_special_tokens=True
        )

        article_budget = max(
            0,
            input_limit
            - empty_prompt_tokens
            - self.SAFETY_MARGIN
        )

        used_ids = article_ids[:article_budget]

        while True:
            used_article = tokenizer.decode(
                used_ids,
                skip_special_tokens=True
            ).strip()

            prompt = self.build_prompt(used_article)

            prompt_tokens = self.token_length(
                tokenizer,
                prompt,
                add_special_tokens=True
            )

            if prompt_tokens <= input_limit:
                return prompt

            if not used_ids:
                raise ValueError(
                    "The prompt template itself is too large "
                    "for the configured context window."
                )

            excess = prompt_tokens - input_limit

            new_length = max(
                0,
                len(used_ids) - excess - 4
            )

            used_ids = used_ids[:new_length]
