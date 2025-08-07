import re

def clean_text(text):
    """Basic PII redaction - extend as needed"""
    redactions = {
        r"\b\d{3}-\d{2}-\d{4}\b": "[REDACTED_SSN]",  # SSN format
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[REDACTED_EMAIL]",  # Email
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b": "[REDACTED_NAME]"  # Full name (First Last)
    }
    for pattern, replacement in redactions.items():
        text = re.sub(pattern, replacement, text)
    return text
