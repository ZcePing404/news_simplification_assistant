import re


class OutputValidator:
    """Checks structure and obvious source-copying before display."""

    ORDERED_LABELS = (
        "Summary",
        "Aspects",
        "Event",
        "People",
        "Organization",
        "Date & Time",
        "Location",
        "Causes",
        "Effects",
        "Countermeasures"
    )

    MIN_SUMMARY_SENTENCES = 2
    MAX_SUMMARY_SENTENCES = 4
    MIN_COPIED_SENTENCE_WORDS = 5
    LONG_COPIED_SENTENCE_WORDS = 20
    MAX_COPIED_SENTENCE_RATIO = 0.5

    WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
    SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

    def evaluate(
        self,
        raw_output: str,
        article: str,
        summary: str
    ) -> dict:
        raw_output = str(raw_output or "").strip()
        article = str(article or "").strip()
        summary = str(summary or "").strip()

        structure_complete = self._has_complete_structure(raw_output)
        sentences = self._sentences(summary)
        sentence_count = len(sentences)
        summary_present = bool(summary and summary.lower() != "not stated")

        normalized_article = self._normalize_words(article)
        copied_sentence_word_counts = []

        for sentence in sentences:
            normalized_sentence = self._normalize_words(sentence)
            word_count = len(self.WORD_RE.findall(sentence))

            if (
                word_count >= self.MIN_COPIED_SENTENCE_WORDS
                and normalized_sentence
                and normalized_sentence in normalized_article
            ):
                copied_sentence_word_counts.append(word_count)

        copied_sentence_count = len(copied_sentence_word_counts)
        copied_sentence_ratio = (
            copied_sentence_count / sentence_count
            if sentence_count
            else 0.0
        )
        longest_copied_sentence_words = max(
            copied_sentence_word_counts,
            default=0
        )

        excessive_copying = (
            copied_sentence_ratio >= self.MAX_COPIED_SENTENCE_RATIO
            or longest_copied_sentence_words
            >= self.LONG_COPIED_SENTENCE_WORDS
        )

        sentence_count_valid = (
            self.MIN_SUMMARY_SENTENCES
            <= sentence_count
            <= self.MAX_SUMMARY_SENTENCES
        )

        issues = []

        if not structure_complete:
            issues.append("the required output labels are incomplete or out of order")

        if not summary_present:
            issues.append("the summary is missing")

        if summary_present and not sentence_count_valid:
            issues.append("the summary does not contain 2 to 4 sentences")

        if excessive_copying:
            issues.append("the summary copies too much source text verbatim")

        source_token_coverage = self._source_token_coverage(
            article,
            summary
        )

        is_valid = not issues

        quality_score = (
            (5.0 if structure_complete else 0.0)
            + (2.0 if summary_present else 0.0)
            + (1.0 if sentence_count_valid else 0.0)
            + (3.0 if not excessive_copying else 0.0)
            - copied_sentence_ratio
        )

        return {
            "is_valid": is_valid,
            "validation_issue_detected": not is_valid,
            "issues": issues,
            "quality_score": quality_score,
            "structure_complete": structure_complete,
            "summary_sentence_count": sentence_count,
            "copied_source_sentence_detected": (
                copied_sentence_count > 0
            ),
            "copied_sentence_count": copied_sentence_count,
            "copied_sentence_ratio": round(copied_sentence_ratio, 4),
            "longest_copied_sentence_words": (
                longest_copied_sentence_words
            ),
            "source_token_coverage": round(source_token_coverage, 4)
        }

    def _has_complete_structure(self, raw_output: str) -> bool:
        positions = []

        for label in self.ORDERED_LABELS:
            matches = list(re.finditer(
                rf"(?im)^\s*{re.escape(label)}\s*:",
                raw_output
            ))

            if len(matches) != 1:
                return False

            positions.append(matches[0].start())

        return positions == sorted(positions)

    def _sentences(self, text: str) -> list:
        return [
            sentence.strip()
            for sentence in self.SENTENCE_BOUNDARY_RE.split(text)
            if sentence.strip()
        ]

    def _normalize_words(self, text: str) -> str:
        return " ".join(
            token.lower()
            for token in self.WORD_RE.findall(text)
        )

    def _source_token_coverage(
        self,
        article: str,
        summary: str
    ) -> float:
        article_tokens = set(self.WORD_RE.findall(article.lower()))
        summary_tokens = self.WORD_RE.findall(summary.lower())

        if not summary_tokens:
            return 0.0

        matches = sum(
            token in article_tokens
            for token in summary_tokens
        )

        return matches / len(summary_tokens)
