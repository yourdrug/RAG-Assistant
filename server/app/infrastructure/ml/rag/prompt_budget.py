"""Count the complete rendered chat, including message framing and summary.

The default is a conservative UTF-8 estimate, not a model-specific tokenizer.
Callers with a local tokenizer may supply an exact message counter.
"""

import math


def estimate_message_tokens(messages: list) -> int:
    return 3 + sum(12 + math.ceil(len(str(message.content).encode("utf-8")) / 3) for message in messages)
