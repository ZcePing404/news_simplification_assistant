import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class PromptService:
    MAX_CONTEXT = 2048
    DEFAULT_RESERVED_TOKENS = 384
    SAFETY_MARGIN = 8
    KEYWORD_TOP_K = 8
    KEYWORD_MAX_FEATURES = 30000

    SUMMARY_GUIDANCE = (
        "For the Summary, state the central news event first. "
        "Include the most important outcome and essential facts before supporting or background details. "
        "Do not omit facts needed to understand what happened."
    )

    CORRECTIVE_GUIDANCE = (
        "Critical correction: Rewrite the important information in simpler language. Do not copy complete sentences from the article. "
        "Produce 2 to 4 concise summary sentences, then include Aspects and every required aspect label in the requested order."
    )

    def __init__(self, template_path=None):
        project_root = Path(__file__).resolve().parents[1]
        default_template_path = project_root / "data" / "processed" / "prompt_templates.json"
        self.template_path = Path(template_path or default_template_path)

        processed_dir = project_root / "data" / "processed"
        self.keyword_training_paths = (processed_dir / "stage1_train.jsonl", processed_dir / "train.jsonl")

        template = self._load_template()
        self.prompt_template = self._build_full_prompt_template(template)
        self._keyword_vectorizer = None
        self._keyword_feature_names = None

    def _load_template(self) -> str:
        if not self.template_path.exists():
            raise FileNotFoundError(f"Missing prompt template: {self.template_path}")

        with self.template_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        template = data.get("prompt_template")

        if not isinstance(template, str) or not template.strip():
            raise ValueError("Missing prompt_template.")

        if template.count("{article}") != 1:
            raise ValueError("Prompt must contain {article} once.")

        return template

    def build_prompt(self, article: str, corrective: bool = False) -> str:
        article = str(article).strip()
        corrective_block = self.CORRECTIVE_GUIDANCE + "\n\n" if corrective else ""
        return self.prompt_template.format(keyword_block=self._build_keyword_block(article), corrective_block=corrective_block, article=article)

    def _build_full_prompt_template(self, template: str) -> str:
        marker = "Article:"

        if marker not in template:
            raise ValueError("Missing Article section.")

        inserted = self.SUMMARY_GUIDANCE + "\n\n{keyword_block}{corrective_block}" + marker
        return template.replace(marker, inserted, 1)

    def _build_keyword_block(self, article: str) -> str:
        keywords = self.extract_source_keywords(article)

        if not keywords:
            return ""

        return "Important source terms: " + "; ".join(keywords) + "\n\n"

    def extract_source_keywords(self, article: str) -> list:
        article = str(article).strip()

        if not article:
            return []

        self._ensure_keyword_vectorizer()
        vector = self._keyword_vectorizer.transform([article])

        if vector.nnz == 0:
            return []

        row = vector.tocoo()
        pairs = sorted(zip(row.col, row.data), key=lambda item: item[1], reverse=True)
        selected = []
        seen = set()

        for feature_index, _ in pairs:
            term = str(self._keyword_feature_names[feature_index]).strip()

            if term and term not in seen:
                selected.append(term)
                seen.add(term)

            if len(selected) >= self.KEYWORD_TOP_K:
                break

        return selected

    def _ensure_keyword_vectorizer(self) -> None:
        if self._keyword_vectorizer is not None:
            return

        articles = []

        for path in self.keyword_training_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing keyword data: {path}")

            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON: {path}:{line_number}") from error

                    article = str(record.get("article", "")).strip()

                    if not article:
                        raise ValueError(f"Missing article: {path}:{line_number}")

                    articles.append(article)

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=self.KEYWORD_MAX_FEATURES, min_df=2, lowercase=True)
        vectorizer.fit(articles)
        self._keyword_vectorizer = vectorizer
        self._keyword_feature_names = np.asarray(vectorizer.get_feature_names_out())

    @staticmethod
    def token_length(tokenizer, text: str, add_special_tokens: bool = True) -> int:
        return len(tokenizer(str(text), add_special_tokens=add_special_tokens, truncation=False)["input_ids"])

    def fit_prompt_to_context(self, tokenizer, article: str, reserved_tokens: int = None, corrective: bool = False, return_details: bool = False):
        if reserved_tokens is None:
            reserved_tokens = self.DEFAULT_RESERVED_TOKENS

        if reserved_tokens <= 0:
            raise ValueError("reserved_tokens must be positive.")

        input_limit = self.MAX_CONTEXT - reserved_tokens

        if input_limit <= 0:
            raise ValueError("No input context available.")

        article = str(article).strip()
        article_ids = tokenizer(article, add_special_tokens=False, truncation=False)["input_ids"]
        empty_prompt_tokens = self.token_length(tokenizer, self.build_prompt("", corrective=corrective), add_special_tokens=True)
        article_budget = max(0, input_limit - empty_prompt_tokens - self.SAFETY_MARGIN)
        used_ids = article_ids[:article_budget]

        while True:
            used_article = tokenizer.decode(used_ids, skip_special_tokens=True).strip()
            prompt = self.build_prompt(used_article, corrective=corrective)
            prompt_tokens = self.token_length(tokenizer, prompt, add_special_tokens=True)

            if prompt_tokens <= input_limit:
                if return_details:
                    return {
                        "prompt": prompt,
                        "input_token_length": prompt_tokens,
                        "article_was_truncated": len(used_ids) < len(article_ids)
                    }

                return prompt

            if not used_ids:
                raise ValueError("Prompt exceeds the context window.")

            excess = prompt_tokens - input_limit
            new_length = max(0, len(used_ids) - excess - 4)
            used_ids = used_ids[:new_length]
