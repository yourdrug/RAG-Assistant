"""Count the complete rendered chat, including message framing and summary.

The default is a conservative UTF-8 estimate, not a model-specific tokenizer.
Callers with a local tokenizer may supply an exact message counter.
"""

import math


def estimate_text_tokens(text: str) -> int:
    """Use the same conservative estimate for context and chat message text."""
    return math.ceil(len(text.encode("utf-8")) / 3)


def estimate_message_tokens(messages: list) -> int:
    return 3 + sum(12 + estimate_text_tokens(str(message.content)) for message in messages)
