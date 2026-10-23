import re
from typing import Optional, Tuple


class OutputGuard:

    MAX_OUTPUT_LENGTH = 10000
    MAX_REPEAT = 20

    PROMPT_LEAK_PATTERNS = [
        r"system prompt",
        r"developer prompt",
        r"hidden prompt",
        r"internal instructions?",
        r"initial instructions?",
        r"confidential instructions?",
        r"chain of thought",
        r"reasoning process",
    ]

    CONTEXT_LEAK_PATTERNS = [
        r"metadata",
        r"vector id",
        r"embedding",
        r"chunk id",
        r"document id",
        r"internal document",
        r"file path",
    ]

    API_KEY_PATTERNS = [
        r"sk-[A-Za-z0-9]{20,}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"ghp_[A-Za-z0-9]{36}",
    ]

    EMAIL_PATTERN = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    PHONE_PATTERN = re.compile(
        r"\+?\d[\d\s\-]{8,}\d"
    )

    REPEAT_PATTERN = re.compile(
        rf"(.)\1{{{MAX_REPEAT},}}"
    )

    @classmethod
    def validate(
        cls,
        question:str,
        answer: str,
        context: str,
    ) -> Tuple[bool, Optional[str]]:

        # -------------------------
        # Empty answer
        # -------------------------

        if answer is None:
            return False, "Answer is empty."

        answer = answer.strip()

        if len(answer) == 0:
            return False, "Answer is empty."

        # -------------------------
        # Output length
        # -------------------------

        if len(answer) > cls.MAX_OUTPUT_LENGTH:
            return False, "Answer exceeds maximum length."

        # -------------------------
        # Retrieved context exists
        # -------------------------

        if context is None:
            return False, "No retrieved context."

        context = context.strip()

        if len(context) == 0:
            return False, "Retrieved context is empty."

        lower_answer = answer.lower()

        # -------------------------
        # Prompt leakage
        # -------------------------

        if cls._contains(
            lower_answer,
            cls.PROMPT_LEAK_PATTERNS,
        ):
            return False, "Prompt leakage detected."

        # -------------------------
        # Internal context leakage
        # -------------------------

        if cls._contains(
            lower_answer,
            cls.CONTEXT_LEAK_PATTERNS,
        ):
            return False, "Internal context leakage detected."

        # -------------------------
        # API keys
        # -------------------------

        if cls._contains(
            answer,
            cls.API_KEY_PATTERNS,
        ):
            return False, "API key detected."

        # -------------------------
        # Email
        # -------------------------

        if cls.EMAIL_PATTERN.search(answer):
            return False, "Email detected."

        # -------------------------
        # Phone
        # -------------------------

        if cls.PHONE_PATTERN.search(answer):
            return False, "Phone number detected."

        # -------------------------
        # Repeated tokens
        # -------------------------

        if cls.REPEAT_PATTERN.search(answer):
            return False, "Repeated characters detected."

        # -------------------------
        # Groundedness Check
        # -------------------------

        if not cls._grounded(answer, context):
            return False, "Answer is not grounded in retrieved context."

        return True, None

    @staticmethod
    def _contains(text: str, patterns: list[str]) -> bool:

        for pattern in patterns:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    @staticmethod
    def _grounded(
        answer: str,
        context: str,
    ) -> bool:
        """
        Simple heuristic groundedness check.

        This is intentionally lightweight.
        Replace later with an LLM judge if desired.
        """

        answer_words = {
            w.lower()
            for w in re.findall(r"\w+", answer)
            if len(w) > 4
        }

        context_words = {
            w.lower()
            for w in re.findall(r"\w+", context)
            if len(w) > 4
        }

        if not answer_words:
            return True

        overlap = len(answer_words & context_words)

        score = overlap / len(answer_words)

        return score >= 0.30