"""Analyst agent: compare architecture alternatives with weighted criteria.

The analyst owns only the `analysis_artifact` field of shared state.
Scores are derived from the use case (user request, system profile) and
the evidence retrieved by the researcher; Python validates the schema
and computes all totals deterministically.
"""

from ada.schemas import (
    AlternativeAnalysis,
    AnalysisArtifact,
    Criterion,
    CriterionScore,
)
from ada.tools.scoring import calculate_weighted_score, rank_alternatives

DEFAULT_ALTERNATIVES = [
    {"alternative_id": "monolith", "name": "Monolito"},
    {"alternative_id": "modular-monolith", "name": "Monolito modular"},
    {"alternative_id": "microservices", "name": "Microservicios"},
]

DEFAULT_CRITERIA = [
    {
        "criterion_id": "dev-speed",
        "name": "Velocidad de desarrollo",
        "weight": 0.25,
        "description": "Rapidez con la que el equipo puede entregar funcionalidad.",
    },
    {
        "criterion_id": "operational-cost",
        "name": "Simplicidad operativa",
        "weight": 0.25,
        "description": "Costo de desplegar, monitorear y depurar el sistema.",
    },
    {
        "criterion_id": "independent-deploys",
        "name": "Despliegues independientes",
        "weight": 0.15,
        "description": "Capacidad de desplegar componentes sin afectar a otros.",
    },
    {
        "criterion_id": "scalability",
        "weight": 0.15,
        "name": "Escalabilidad",
        "description": "Capacidad de escalar componentes según demanda.",
    },
    {
        "criterion_id": "team-alignment",
        "name": "Alineación con el equipo",
        "weight": 0.20,
        "description": "Ajuste entre límites de arquitectura y estructura del equipo.",
    },
]

# Señales por criterio: frases del caso de uso que empujan el puntaje del
# criterio hacia arriba o hacia abajo. Se buscan en la solicitud del
# usuario, el perfil del sistema y la evidencia recuperada.
CRITERIA_SIGNALS: dict[str, dict[str, list[str]]] = {
    "dev-speed": {
        "positive": ["entregar rápido", "presión", "plazo", "time to market", "mvp"],
        "negative": ["sin apuro", "plazo flexible"],
    },
    "operational-cost": {
        "negative": ["equipo de plataformas", "observabilidad", "alta escala"],
        "positive": ["equipo pequeño", "operación simple"],
    },
    "independent-deploys": {
        "positive": ["despliegues independientes", "release frecuente", "varios equipos"],
        "negative": ["no se requieren despliegues independientes"],
    },
    "scalability": {
        "positive": ["escalar", "alta demanda", "crecimiento", "picos"],
        "negative": ["carga predecible", "baja concurrencia"],
    },
    "team-alignment": {
        "positive": ["equipo pequeño", "un solo equipo"],
        "negative": ["muchos equipos", "organización grande"],
    },
    "coupling": {
        "positive": [
            "acoplamiento bajo",
            "desacoplar",
            "desacoplado",
            "mensajería",
            "mensajería asincrónica",
        ],
        "negative": ["acoplamiento fuerte", "llamadas directas", "acoplamiento"],
    },
    "resilience": {
        "positive": [
            "resiliencia",
            "tolerancia a fallos",
            "reintentos",
            "circuit breaker",
            "desacoplar",
        ],
        "negative": ["punto único de falla", "acoplamiento fuerte"],
    },
    "latency": {
        "positive": ["latencia baja", "respuesta inmediata", "baja latencia"],
        "negative": ["latencia", "espera", "bloqueante"],
    },
}

# Señales por alternativa: si aparecen en la evidencia recuperada, la
# alternativa recibe un empujón semántico en sus puntajes.
ALTERNATIVE_SIGNALS: dict[str, list[str]] = {
    "monolith": ["monolito", "monolith", "aplicación única"],
    "modular-monolith": ["monolito modular", "modular monolith", "módulos"],
    "microservices": ["microservicios", "microservices", "servicios independientes"],
    "azure-service-bus": ["service bus", "mensajería", "cola de mensajes", "asincrónica"],
    "sync-http": ["http", "llamada directa", "síncrona", "sincrona"],
}


def system_profile_text(system_profile: dict) -> str:
    """Flatten the system profile into a single text for signal matching."""
    return " ".join(
        str(value) for value in system_profile.values() if isinstance(value, (str, list))
    )


def _count_signals(text: str, phrases: list[str]) -> int:
    """Count how many signal phrases appear in the given text."""
    normalized = text.lower()
    return sum(1 for phrase in phrases if phrase in normalized)


def _criterion_terms(criterion: Criterion) -> list[str]:
    """Señales (positivas y negativas) y nombre de un criterio."""
    signals = CRITERIA_SIGNALS.get(criterion.criterion_id, {})
    phrases = list(signals.get("positive", [])) + list(signals.get("negative", []))
    return [criterion.name.lower(), *phrases]


def _pertinent_evidence_ids(
    research_artifact: dict,
    alternative_id: str,
    criterion: Criterion,
) -> list[str]:
    """Evidencia relacionada con la alternativa Y el criterio evaluado.

    Una evidencia es pertinente para una puntuación solo si menciona
    señales de la alternativa (p. ej. "service bus") y además términos
    relacionados con el criterio evaluado (sus señales o su nombre).
    """
    evidence = research_artifact.get("evidence", []) or []
    alternative_signals = ALTERNATIVE_SIGNALS.get(alternative_id, [])
    if not alternative_signals:
        return []
    criterion_phrases = _criterion_terms(criterion)
    matched = []
    for item in evidence:
        text = f"{item.get('title', '')} {item.get('excerpt', '')}".lower()
        mentions_alternative = any(signal in text for signal in alternative_signals)
        mentions_criterion = any(phrase in text for phrase in criterion_phrases)
        if mentions_alternative and mentions_criterion:
            matched.append(item.get("document_id", ""))
    return sorted({doc_id for doc_id in matched if doc_id})


def derive_scores_from_evidence(
    user_request: str,
    system_profile: dict,
    criteria: list[Criterion],
    alternatives: list[dict],
    research_artifact: dict,
) -> dict[str, list[dict]]:
    """Derive per-criterion scores from the use case and the evidence.

    Deterministic keyword-signal scoring:
    - Signals found in the user request / system profile push each
      criterion score up or down.
    - Evidence excerpts matching alternative keywords push that
      alternative's scores up (semantic support from the corpus).
    - Criteria with no evidence backing are marked `is_assumption=True`
      and the uncertainty is reported in the artifact's data gaps.
    """
    request_text = f"{user_request} {system_profile_text(system_profile)}".lower()

    scores_by_alternative: dict[str, list[dict]] = {}

    for alternative in alternatives:
        alternative_id = alternative["alternative_id"]
        alternative_text = " ".join(
            str(value) for value in alternative.values() if isinstance(value, str)
        ).lower()
        scores: list[dict] = []

        for criterion in criteria:
            signals = CRITERIA_SIGNALS.get(criterion.criterion_id, {})
            positive_hits = _count_signals(
                f"{request_text} {alternative_text}", signals.get("positive", [])
            )
            negative_hits = _count_signals(
                f"{request_text} {alternative_text}", signals.get("negative", [])
            )

            # Neutral baseline: 3. Signals push within the 1..5 range.
            score = 3 + positive_hits - negative_hits
            score = min(max(score, 1), 5)

            # Respaldo por puntuación: solo la evidencia realmente
            # relacionada con la alternativa Y el criterio cuenta como
            # soporte trazable.
            pertinent_ids = _pertinent_evidence_ids(
                research_artifact, alternative_id, criterion
            )

            # Corpus support: evidence pertinent to this alternative AND
            # criterion nudges the score up by one (bounded).
            if pertinent_ids:
                score = min(score + 1, 5)

            rationale = _build_rationale(
                criterion=criterion,
                score=score,
                positive_hits=positive_hits,
                negative_hits=negative_hits,
                has_evidence=bool(pertinent_ids),
            )

            scores.append(
                {
                    "criterion_id": criterion.criterion_id,
                    "weight": criterion.weight,
                    "score": score,
                    "rationale": rationale,
                    "evidence_ids": pertinent_ids,
                    "is_assumption": not pertinent_ids,
                }
            )

        scores_by_alternative[alternative_id] = scores

    return scores_by_alternative


def _count_signals(text: str, signals: list[str]) -> int:
    """Count how many signal phrases appear in the text."""
    return sum(1 for signal in signals if signal in text)


def _build_rationale(
    positive_hits: int,
    negative_hits: int,
    score: int,
    criterion: Criterion,
    has_evidence: bool,
) -> str:
    """Build a deterministic rationale for a criterion score."""
    parts = [f"Puntaje {score}/5 para '{criterion.name}'."]
    if positive_hits:
        parts.append(f"Señales a favor del caso: {positive_hits}.")
    if negative_hits:
        parts.append(f"Señales en contra del caso: {negative_hits}.")
    if has_evidence:
        parts.append("Apoyado en evidencia del corpus local.")
    else:
        parts.append("Sin evidencia directa: estimación sujeta a revisión.")
    return " ".join(parts)


def build_analysis(
    criteria: list[Criterion],
    alternatives: list[dict],
    scores_by_alternative: dict[str, list[CriterionScore]],
) -> AnalysisArtifact:
    """Build a validated analysis from proposed scores.

    Python recalculates every weighted score with `calculate_weighted_score`
    and derives the ranking with `rank_alternatives`; the LLM never computes
    totals.
    """
    if not alternatives:
        raise ValueError("At least two alternatives are required.")

    analyses: list[AlternativeAnalysis] = []

    for alternative in alternatives:
        alternative_id = alternative["alternative_id"]
        proposed_scores = scores_by_alternative.get(alternative_id)
        if proposed_scores is None:
            raise ValueError(f"Missing proposed scores for alternative '{alternative_id}'.")

        # Accept both CriterionScore instances and plain dicts.
        validated_scores = [
            score if isinstance(score, CriterionScore) else CriterionScore.model_validate(score)
            for score in proposed_scores
        ]

        weighted_score = calculate_weighted_score(validated_scores, criteria)

        analyses.append(
            AlternativeAnalysis(
                alternative_id=alternative_id,
                name=alternative["name"],
                scores=validated_scores,
                weighted_score=weighted_score,
                advantages=alternative.get("advantages", []),
                disadvantages=alternative.get("disadvantages", []),
                risks=alternative.get("risks", []),
                assumptions=alternative.get("assumptions", []),
            )
        )

    ranking = rank_alternatives(analyses)

    best_id = ranking[0]
    best = next(a for a in analyses if a.alternative_id == best_id)
    # Empates honestos: si el top comparte puntaje, se declara empate.
    tied = [
        a.name
        for a in analyses
        if abs(a.weighted_score - best.weighted_score) <= 1e-6
    ]
    if len(tied) > 1:
        summary = (
            f"Empate entre {len(tied)} alternativas con puntaje ponderado "
            f"{best.weighted_score:.6f}: {', '.join(tied)}. El ranking refleja "
            f"el orden de presentación, no una diferencia real."
        )
    else:
        summary = (
            f"'{best.name}' obtiene el mayor puntaje ponderado "
            f"({best.weighted_score:.3f}) sobre {len(analyses)} alternativas "
            f"evaluadas con {len(criteria)} criterios ponderados."
        )

    data_gaps = [
        f"Criterio '{score.criterion_id}' sin evidencia directa: "
        "estimación sujeta a revisión."
        for alternative in analyses
        for score in alternative.scores
        if score.is_assumption
    ]

    return AnalysisArtifact(
        alternatives=analyses,
        ranking=ranking,
        data_gaps=data_gaps,
        qualitative_summary=summary,
    )
