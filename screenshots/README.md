# Capturas de evidencia (Pre-entrega 7)

Este directorio contiene las capturas reales del dashboard de Arize
Phoenix tomadas durante la prueba de aceptación. **No se incluyen
capturas simuladas**: cada imagen debe corresponder a una ejecución real
contra servicios activos.

## Capturas requeridas

| Archivo esperado | Contenido | Cómo obtenerla |
|------------------|-----------|----------------|
| `traza_completa.png` | Traza completa de un job ADA (supervisor → researcher → analyst → validator → synthesizer) con spans anidados de llamadas al LLM | Phoenix UI → Projects → `ada-api` → abrir una sesión/traza de un job DONE |
| `cinco_concurrentes.png` | Las 5 ejecuciones de la prueba de carga, cada una como sesión separada | Ejecutar `scripts/load_test.py` y refrescar el dashboard |
| `latencia_p95.png` | Métrica de latencia (p95 o media) mostrada por el dashboard | Phoenix UI → vista de latencia del proyecto |
| `costo_tokens.png` | Tokens de entrada/salida y costo por ejecución | Phoenix UI → detalle de un span de LLM (atributos `llm.token_count.*`) |

## Estado actual

- [x] `traza_completa.png` — capturada (job real con Groq: LangGraph → hitl_approval → supervisor → ChatGroq → researcher → analyst → validator → synthesizer)
- [x] `cinco_concurrentes.png` — capturada (5 traces LangGraph consecutivos de la prueba de carga, con tokens y costo por ejecución)
- [x] `latencia_p95.png` — capturada (vista Metrics: Trace latency con percentiles)
- [x] `latencia_p95_1.png` — capturada (vista Metrics alternativa: percentiles P50–P99.9, Cost y Token usage)
- [x] `costo_tokens.png` — capturada (tokens `llm.token_count.*` y costo del span LLM)

## Capturas complementarias

| Archivo | Contenido |
|---------|-----------|
| `traza_output_final.png` | Span raíz `LangGraph` con el output final del job (informe de decisión) |
| `spans_general.png` | Vista Spans del proyecto con métricas agregadas (Total Traces, Total Cost, Latency P50/P99) |
| `traces_listado.png` | Listado de traces del proyecto |

## Procedimiento para capturar

1. Levantar los servicios: `docker compose up --build -d`
2. Verificar salud: `curl http://localhost:8000/health` (debe dar `"redis": true`)
3. Abrir Phoenix: http://localhost:6006 → proyecto `ada-api`
4. Ejecutar la prueba de carga: `uv run python scripts/load_test.py`
5. Tomar las 4 capturas del dashboard y guardarlas aquí con los nombres de la tabla.
