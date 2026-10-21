import pytest

from ada.schemas import Criterion, CriterionScore
from ada.tools.scoring import calculate_weighted_score


def test_weighted_score_is_normalized():
    criteria = [
        Criterion(
            criterion_id="maintenance",
            name="Mantenibilidad",
            weight=0.6,
            description="Facilidad de mantenimiento",
        ),
        Criterion(
            criterion_id="cost",
            name="Costo",
            weight=0.4,
            description="Costo operativo",
        ),
    ]

    scores = [
        CriterionScore(
            criterion_id="maintenance",
            weight=0.6,
            score=4,
            rationale="Bajo acoplamiento",
        ),
        CriterionScore(
            criterion_id="cost",
            weight=0.4,
            score=3,
            rationale="Costo moderado",
        ),
    ]

    assert calculate_weighted_score(scores, criteria) == pytest.approx(0.72)


def test_rejects_missing_criterion():
    criteria = [
        Criterion(
            criterion_id="maintenance",
            name="Mantenibilidad",
            weight=0.6,
            description="Facilidad de mantenimiento",
        ),
        Criterion(
            criterion_id="cost",
            name="Costo",
            weight=0.4,
            description="Costo operativo",
        ),
    ]

    scores = [
        CriterionScore(
            criterion_id="maintenance",
            weight=0.6,
            score=4,
            rationale="Bajo acoplamiento",
        ),
    ]

    with pytest.raises(ValueError, match="exactly the configured criteria"):
        calculate_weighted_score(scores, criteria)