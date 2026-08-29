import re


class OutputParser:
    """
    Converts the model's fixed-format text response into a structured
    summary + aspects dictionary for the Flask UI.
    """

    DEFAULT_VALUE = "Not stated"

    ASPECT_KEYS = (
        "event",
        "people_organizations",
        "date_time",
        "location",
        "causes",
        "effects",
        "countermeasures"
    )

    PATTERNS = {
        "summary": re.compile(
            r"Summary:\s*(.*?)"
            r"(?=\s*Aspects:|\s*Event:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "event": re.compile(
            r"Event:\s*(.*?)"
            r"(?=\s*People\s*/\s*Organizations:|\s*People:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "people_organizations": re.compile(
            r"People\s*/\s*Organizations:\s*(.*?)"
            r"(?=\s*Date\s*&\s*Time:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "people": re.compile(
            r"People:\s*(.*?)"
            r"(?=\s*Organization:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "organization": re.compile(
            r"Organization:\s*(.*?)"
            r"(?=\s*Date\s*&\s*Time:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "date_time": re.compile(
            r"Date\s*&\s*Time:\s*(.*?)"
            r"(?=\s*Location:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "location": re.compile(
            r"Location:\s*(.*?)"
            r"(?=\s*Causes:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "causes": re.compile(
            r"Causes:\s*(.*?)"
            r"(?=\s*Effects:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        "effects": re.compile(
            r"Effects:\s*(.*?)"
            r"(?=\s*Countermeasures:|$)",
            flags=re.IGNORECASE | re.DOTALL
        ),

        # Stop at another Summary/Aspects block if a small model
        # starts repeating the output template.
        "countermeasures": re.compile(
            r"Countermeasures:\s*(.*?)"
            r"(?=\s*Summary:|\s*Aspects:|$)",
            flags=re.IGNORECASE | re.DOTALL
        )
    }

    def parse(self, text: str) -> dict:
        text = self._normalize(text)

        summary = self._extract(
            text,
            "summary"
        )

        aspects = {
            key: self._extract(text, key)
            for key in self.ASPECT_KEYS
        }

        if aspects["people_organizations"] == self.DEFAULT_VALUE:
            aspects["people_organizations"] = self._extract_legacy_entities(text)

        return {
            "summary": summary,
            "aspects": aspects
        }

    def _extract_legacy_entities(self, text: str) -> str:
        """Combine output from checkpoints trained with the old two-label schema."""
        values = []

        for key in ("people", "organization"):
            value = self._extract(text, key)

            if value.lower() != self.DEFAULT_VALUE.lower() and value not in values:
                values.append(value)

        return "; ".join(values) if values else self.DEFAULT_VALUE

    def _extract(self, text: str, key: str) -> str:
        pattern = self.PATTERNS[key]
        match = pattern.search(text)

        if not match:
            return self.DEFAULT_VALUE

        value = self._clean_value(
            match.group(1)
        )

        return value or self.DEFAULT_VALUE

    @staticmethod
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""

        return (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    @staticmethod
    def _clean_value(value: str) -> str:
        # Preserve meaningful sentence content while making accidental
        # line wrapping from generation cleaner for the web UI.
        return re.sub(
            r"\s+",
            " ",
            str(value)
        ).strip()
