"""Extracción determinística de la especificación de decisión.

Convierte la consulta libre del usuario en una especificación estructurada
y validada (pregunta de decisión, alternativas solicitadas, criterios y
restricciones). No usa LLM: la extracción es determinística y auditable.

Política honesta: si las alternativas no se pueden extraer de la consulta,
`requested_alternatives` queda vacío y el resto del sistema NO debe
inventar alternativas como si hubieran sido solicitadas.
"""

import re

from pydantic import Field, model_validator

from ada.schemas import Criterion, StrictModel


class RequestedAlternative(StrictModel):
    """Alternativa explícita extraída de la consulta del usuario."""

    alternative_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # Frases (minúsculas) que identifican esta alternativa en textos.
    signals: list[str] = Field(min_length=1)


class DecisionSpec(StrictModel):
    """Especificación estructurada de la decisión solicitada."""

    decision_question: str = Field(min_length=1)
    requested_alternatives: list[RequestedAlternative] = Field(default_factory=list)
    requested_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    # True solo si las alternativas salieron explícitamente de la consulta.
    alternatives_are_explicit: bool = False

    @model_validator(mode="after")
    def explicit_requires_alternatives(self):
        if self.alternatives_are_explicit and not self.requested_alternatives:
            raise ValueError(
                "alternatives_are_explicit requires at least one requested alternative."
            )
        return self


# Vocabulario de dominio: mapea nombres técnicos a identificadores estables.
# Cada entrada define las señales léxicas que la reconocen en texto libre.
KNOWN_ALTERNATIVES: dict[str, dict] = {
    "azure-service-bus": {
        "alternative_id": "azure-service-bus",
        "name": "Azure Service Bus",
        "signals": [
            "azure service bus",
            "service bus",
            "mensajería asíncrona",
            "mensajeria asincrona",
            "cola de mensajes",
            "message broker",
        ],
    },
    "sync-http": {
        "alternative_id": "sync-http",
        "name": "HTTP síncrono",
        "signals": [
            "http síncrono",
            "http sincrono",
            "comunicación http",
            "comunicacion http",
            "llamadas http",
            "http directo",
        ],
    },
    "monolith": {
        "alternative_id": "monolith",
        "name": "Monolito",
        "signals": ["monolito", "monolith"],
    },
    "modular-monolith": {
        "alternative_id": "modular-monolith",
        "name": "Monolito modular",
        "signals": ["monolito modular", "modular monolith"],
    },
    "microservices": {
        "alternative_id": "microservices",
        "name": "Microservicios",
        "signals": ["microservicios", "microservices"],
    },
}

# Criterios reconocibles en la consulta (texto normalizado -> criterion_id).
KNOWN_CRITERIA: dict[str, str] = {
    "acoplamiento": "coupling",
    "acoplamiento entre servicios": "coupling",
    "resiliencia": "resilience",
    "latencia": "latency",
    "complejidad operativa": "operational-cost",
    "simplicidad operativa": "operational-cost",
    "escalabilidad": "scalability",
    "velocidad de desarrollo": "dev-speed",
    "despliegues independientes": "independent-deploys",
}

# Metadatos por criterion_id: nombre legible, descripción y peso base.
# El peso base es relativo; build_criteria() lo normaliza para que la suma
# sea exactamente 1.0.
CRITERIA_METADATA: dict[str, dict] = {
    "coupling": {
        "name": "Acoplamiento",
        "description": "Grado de dependencia técnica entre servicios.",
        "weight": 0.20,
    },
    "resilience": {
        "name": "Resiliencia",
        "description": "Capacidad de absorber fallos parciales sin degradar el servicio.",
        "weight": 0.20,
    },
    "latency": {
        "name": "Latencia",
        "description": "Impacto en la latencia percibida por el consumidor.",
        "weight": 0.15,
    },
    "operational-cost": {
        "name": "Complejidad operativa",
        "description": "Costo de desplegar, monitorear y depurar el sistema.",
        "weight": 0.15,
    },
    "scalability": {
        "name": "Escalabilidad",
        "description": "Capacidad de escalar componentes según demanda.",
        "weight": 0.15,
    },
    "dev-speed": {
        "name": "Velocidad de desarrollo",
        "description": "Rapidez con la que el equipo puede entregar funcionalidad.",
        "weight": 0.10,
    },
    "independent-deploys": {
        "name": "Despliegues independientes",
        "description": "Capacidad de desplegar componentes sin afectar a otros.",
        "weight": 0.10,
    },
}

# Conectores de comparación explícita entre dos alternativas.
_COMPARISON_PATTERNS = [
    r"\b(?P<first>[\w\sáéíóúñü-]+?)\s+(?:vs\.?|versus|o\b|y\b)\s+(?P<second>[\w\sáéíóúñü-]+?)(?=[,.;:¿?!]|$)",
    r"\bcompar(?:á|a|ar|e)\s+(?P<first>[\w\sáéíóúñü-]+?)\s+(?:con|y|e)\s+(?P<second>[\w\sáéíóúñü-]+?)(?=[,.;:¿?!]|$)",
    r"\bevalu(?:a|á|ar|e)\s+(?P<first>[\w\sáéíóúñü-]+?)\s+(?:y|con)\s+(?P<second>[\w\sáéíóúñü-]+?)(?=[,.;:¿?!]|$)",
]

# Palabras que no deben capturarse como parte de un nombre de alternativa.
_STOPWORDS = {
    "necesito", "decidir", "cómo", "como", "implementar", "compará", "compara",
    "comparar", "evaluando", "evaluar", "evaluá", "evalua", "justificá",
    "justifica", "indicá", "indica", "compará", "la", "el", "los", "las",
    "un", "una", "unos", "unas", "de", "del", "al", "a", "en", "con", "para",
    "por", "que", "y", "o", "vs", "versus", "entre", "arquitectura",
    "arquitectura .net", "net", "azure", "desplegada", "azure .net",
}


def _normalize(text: str) -> str:
    """Minimizar y colapsar espacios para matching determinístico."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _slugify(name: str) -> str:
    """Convertir un nombre en identificador estable."""
    slug = re.sub(r"[^a-z0-9]+", "-", _normalize(name)).strip("-")
    return slug or "alternativa"


def _match_known_alternatives(text: str) -> list[RequestedAlternative]:
    """Detectar alternativas conocidas por sus señales léxicas.

    Excluye menciones que forman parte de frases de contexto (p. ej.
    "comunicación entre microservicios" no solicita evaluar Microservicios).
    """
    normalized = _normalize(text)
    found: list[RequestedAlternative] = []
    for spec in KNOWN_ALTERNATIVES.values():
        for signal in spec["signals"]:
            if signal not in normalized:
                continue
            # Una señal precedida de "entre" o "dentro de" indica contexto
            # (p. ej. "entre microservicios"), no una alternativa solicitada.
            head = normalized[: normalized.index(signal)]
            if re.search(r"\b(entre|dentro de)\s+$", head):
                continue
            found.append(
                RequestedAlternative(
                    alternative_id=spec["alternative_id"],
                    name=spec["name"],
                    signals=spec["signals"],
                )
            )
            break
    return found


def _match_known_criteria(text: str) -> list[str]:
    """Detectar criterios conocidos mencionados en la consulta."""
    normalized = _normalize(text)
    return [
        criterion_id
        for phrase, criterion_id in KNOWN_CRITERIA.items()
        if phrase in normalized
    ]


def extract_decision(user_request: str) -> DecisionSpec:
    """Extraer la especificación de decisión de la consulta del usuario.

    Determinístico y honesto: si no se detectan al menos dos alternativas
    conocidas, `requested_alternatives` queda vacío y
    `alternatives_are_explicit` es False; el resto del sistema no debe
    presentar alternativas genéricas como si hubieran sido solicitadas.
    """
    candidates = _match_known_alternatives(user_request)
    # Desduplicar por id conservando el orden de aparición.
    unique: dict[str, RequestedAlternative] = {}
    for alternative in candidates:
        unique.setdefault(alternative.alternative_id, alternative)

    alternatives = list(unique.values())
    explicit = len(alternatives) >= 2

    criteria = _match_known_criteria(user_request)

    return DecisionSpec(
        decision_question=user_request.strip(),
        requested_alternatives=alternatives if explicit else [],
        alternatives_are_explicit=explicit,
        requested_criteria=criteria,
    )


def build_criteria(requested_ids: list[str]) -> list[Criterion]:
    """Construir objetos Criterion válidos a partir de identificadores.

    Usa CRITERIA_METADATA para nombre y descripción; normaliza los pesos
    base para que sumen exactamente 1.0 (tolerancia de scoring). Lanza
    ValueError si algún identificador no tiene metadatos: los criterios
    nunca se inventan.
    """
    if not requested_ids:
        raise ValueError("At least one criterion id is required.")

    unknown = [
        criterion_id
        for criterion_id in requested_ids
        if criterion_id not in CRITERIA_METADATA
    ]
    if unknown:
        raise ValueError(f"Unknown criterion ids: {', '.join(unknown)}.")

    total_base = sum(CRITERIA_METADATA[c]["weight"] for c in requested_ids)
    weights = [
        round(CRITERIA_METADATA[criterion_id]["weight"] / total_base, 6)
        for criterion_id in requested_ids
    ]
    # Ajuste de residuo: el último criterio absorbe la diferencia de
    # redondeo para que la suma sea exactamente 1.0 (tolerancia 1e-6).
    weights[-1] = round(weights[-1] + (1.0 - sum(weights)), 6)
    return [
        Criterion(
            criterion_id=criterion_id,
            name=CRITERIA_METADATA[criterion_id]["name"],
            weight=weight,
            description=CRITERIA_METADATA[criterion_id]["description"],
        )
        for criterion_id, weight in zip(requested_ids, weights)
    ]
