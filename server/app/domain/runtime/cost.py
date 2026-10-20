"""USD accounting in integer nanodollars; tariffs are versioned data."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING

from domain.shared.exceptions import BusinessRuleViolation

UNDISPATCHED_COST_SOURCE = "not_dispatched"


class BudgetError(BusinessRuleViolation):
    """A paid operation cannot safely be admitted."""


def usd_units(value: str | Decimal) -> int:
    try:
        amount = Decimal(value)
        if not amount.is_finite() or amount < 0:
            raise ValueError
        units = int((amount * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
        if units > 2**53 - 1:
            raise ValueError
        return units
    except (InvalidOperation, ValueError, OverflowError):
        raise BudgetError("Invalid USD amount") from None


@dataclass(frozen=True)
class CostRate:
    provider: str
    model: str
    version: int
    unit: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    context_tokens: int
    input_only: bool = False

    def reserve(self, input_tokens: int, output_tokens: int) -> int:
        if (
            self.unit != "tokens"
            or self.version < 1
            or input_tokens < 0
            or output_tokens < 0
            or (output_tokens == 0 and not self.input_only)
            or self.context_tokens < 1
            or input_tokens > self.context_tokens
            or output_tokens > self.context_tokens
        ):
            raise BudgetError("Unsupported billing unit or unbounded token budget")
        usd_units(self.input_usd_per_million)
        usd_units(self.output_usd_per_million)
        if self.input_usd_per_million <= 0 or (not self.input_only and self.output_usd_per_million <= 0):
            raise BudgetError("Paid token tariffs must be positive")
        return usd_units(
            (input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million)
            / 1_000_000
        )


@dataclass(frozen=True)
class ModelBillingContract:
    """Verified per-element/pair bound including template and special tokens."""

    unit: str
    operation: str
    input_limit: int
    allowed_parameters: frozenset[str]
    evidence: str
