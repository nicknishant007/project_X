
import re
from typing import Optional, Tuple


class InputGuard:

    MIN_LENGTH = 3
    MAX_LENGTH = 3000
    MAX_REPEAT = 25

    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+.*instructions?",
        r"override\s+.*instructions?",
        r"you\s+are\s+now",
        r"act\s+as\s+system",
        r"system\s+prompt",
        r"developer\s+prompt",
        r"hidden\s+prompt",
        r"reveal\s+.*prompt",
        r"repeat\s+.*prompt",
        r"print\s+.*prompt",
    ]

    JAILBREAK_PATTERNS = [
        r"\bdan\b",
        r"developer\s+mode",
        r"jailbreak",
        r"evilgpt",
        r"unfiltered",
        r"no\s+restrictions",
        r"without\s+restrictions",
        r"ignore\s+openai",
        r"bypass\s+safety",
        r"pretend\s+to\s+be",
    ]

    SQL_PATTERNS = [
        r"drop\s+table",
        r"delete\s+from",
        r"truncate\s+table",
        r"union\s+select",
        r"insert\s+into",
        r"update\s+\w+",
        r"select\s+.*password",
        r"xp_cmdshell",
    ]

    XSS_PATTERNS = [
        r"<script",
        r"</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"onclick\s*=",
    ]

    CONTROL_CHAR_PATTERN = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
    )

    REPEAT_PATTERN = re.compile(
        rf"(.)\1{{{MAX_REPEAT},}}"
    )

    @classmethod
    def validate(cls, question: str) -> Tuple[bool, Optional[str]]:

        if question is None:
            return False, "Question is missing."

        question = question.strip()

        if len(question) == 0:
            return False, "Question cannot be empty."

        if len(question) < cls.MIN_LENGTH:
            return False, "Question is too short."

        if len(question) > cls.MAX_LENGTH:
            return False, "Question is too long."

        if cls.CONTROL_CHAR_PATTERN.search(question):
            return False, "Control characters detected."

        if cls.REPEAT_PATTERN.search(question):
            return False, "Excessive repeated characters detected."

        lower = question.lower()

        if cls._contains(lower, cls.PROMPT_INJECTION_PATTERNS):
            return False, "Prompt injection detected."

        if cls._contains(lower, cls.JAILBREAK_PATTERNS):
            return False, "Jailbreak attempt detected."

        if cls._contains(lower, cls.SQL_PATTERNS):
            return False, "SQL injection detected."

        if cls._contains(lower, cls.XSS_PATTERNS):
            return False, "Script injection detected."

        return True, None

    @staticmethod
    def _contains(text: str, patterns: list[str]) -> bool:

        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True

        return False