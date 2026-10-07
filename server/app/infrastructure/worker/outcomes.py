"""Expected resumable job outcomes, distinct from unexpected worker exceptions."""


class IncompleteEvaluation(ValueError):
    """All work finished, but some judge scores remain unavailable."""
