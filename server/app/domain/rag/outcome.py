"""The outcome of one attempt to answer a question."""

from enum import StrEnum


class AnswerOutcome(StrEnum):
    GENERATED = "generated"
    GROUNDED = "grounded"
    CACHE_HIT = "cache_hit"
    NOT_FOUND = "not_found"
    OUT_OF_DOMAIN = "out_of_domain"
    INPUT_BLOCKED = "input_blocked"
    OUTPUT_BLOCKED = "output_blocked"
    ERROR = "error"
    CANCELLED = "cancelled"
