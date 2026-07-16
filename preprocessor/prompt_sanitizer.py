import re

from utils.logger import get_logger

logger = get_logger("prompt_security")


class PromptSecurityFilter:
    """Security component to mitigate prompt injection & mask sensitive freight data"""

    INJECTION_RULES = [
        re.compile(r"ignore all previous instructions", re.IGNORECASE),
        re.compile(r"forget system rules", re.IGNORECASE),
        re.compile(r"override your base prompt", re.IGNORECASE),
        re.compile(r"reveal all hidden information", re.IGNORECASE),
        re.compile(r"disregard security restrictions", re.IGNORECASE),
    ]

    SENSITIVE_MASK_RULES = [
        (re.compile(r"\$[\d,]+\.?\d* USD"), "[FREIGHT_PRICE_MASKED]"),
        (re.compile(r"\b\d{10,18}\b"), "[CONFIDENTIAL_ID_MASKED]"),
    ]

    def detect_injection_risk(self, user_input: str) -> bool:
        """Return True if potential prompt injection payload detected"""
        for pattern in self.INJECTION_RULES:
            if pattern.search(user_input):
                logger.warning(
                    f"Potential injection detected, raw input snippet: {user_input[:120]}"
                )
                return True
        return False

    def mask_sensitive_content(self, raw_text: str) -> str:
        """Mask price data and confidential ID numbers before sending text to LLM"""
        cleaned_text = raw_text
        for regex, replacement in self.SENSITIVE_MASK_RULES:
            cleaned_text = regex.sub(replacement, cleaned_text)
        return cleaned_text

    def full_pipeline_clean(self, raw_user_input: str) -> str | None:
        """Full security pipeline: injection check + sensitive data masking
        Return None if injection risk exists, return sanitized text otherwise
        """
        if self.detect_injection_risk(raw_user_input):
            return None
        return self.mask_sensitive_content(raw_user_input)
