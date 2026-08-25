import re
import unicodedata


class NewsPreprocessor:
    """
    Cleans user-entered news articles using the same article-side
    preprocessing rules used by preprocessing.ipynb.
    """

    MAX_ARTICLE_WORDS_EXCLUSIVE = 500

    WORD_RE = re.compile(
        r"\b\w+(?:[’'-]\w+)*\b",
        flags=re.UNICODE
    )

    CONTROL_CHARACTER_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    WHITESPACE_RE = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Same Unicode normalization used during preprocessing.
        text = unicodedata.normalize("NFKC", text)

        # Normalize Windows / old Mac line endings.
        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # Replace control characters with spaces.
        text = self.CONTROL_CHARACTER_RE.sub(" ", text)

        # Article preprocessing does not preserve newlines.
        # Collapse all whitespace to a single space.
        text = self.WHITESPACE_RE.sub(" ", text).strip()

        return text

    def count_words(self, text: str) -> int:
        if not isinstance(text, str):
            return 0

        return len(self.WORD_RE.findall(text))

    def is_valid_length(self, text: str) -> bool:
        return (
            self.count_words(text)
            < self.MAX_ARTICLE_WORDS_EXCLUSIVE
        )
