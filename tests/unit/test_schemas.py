import pytest
from pydantic import ValidationError

from ada.schemas import (
    Criterion,
    CriterionScore,
    ValidationResult,
    validate_criteria_weights,
)


def test_criterion_weights_accept_sum_of_one():
    criteria = [
        Criterion(
            criterion_id="maintainability",
            name="Mantenibilidad",
            weight=0.6,
            description="Facilidad para modificar el sistema",
        ),
        Criterion(
            criterion_id="cost",
            name="Costo operativo",
            weight=0.4,
            description="Costo de operación y soporte",
        ),
    ]

    validate_criteria_weights(criteria)


def test_criterion_weights_reject_invalid_sum():
    criteria = [
        Criterion(
            criterion_id="maintainability",
            name="Mantenibilidad",
            weight=0.6,
            description="Facilidad para modificar el sistema",
        ),
        Criterion(
            criterion_id="cost",
            name="Costo operativo",
            weight=0.3,
            description="Costo de operación y soporte",
        ),
    ]

    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_criteria_weights(criteria)


def test_score_rejects_value_outside_one_to_five():
    with pytest.raises(ValidationError):
        CriterionScore(
            criterion_id="cost",
            weight=1.0,
            score=6,
            rationale="Invalid score",
        )


def test_valid_validation_result_cannot_contain_errors():
    with pytest.raises(ValidationError):
        ValidationResult(is_valid=True, errors=["Missing evidence"])
