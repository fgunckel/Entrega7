"""Prueba de carga: 5 peticiones concurrentes contra la API de ADA.

Envía 5 solicitudes simultáneas a POST /tasks (escenario NO crítico, sin
HITL), hace polling de GET /tasks/{id} hasta un estado terminal y reporta
latencia por trabajo y global, con p95 calculado por interpolación
(método estándar de numpy/statistics para muestras pequeñas).

Uso:
    uv run python scripts/load_test.py [--url http://localhost:8000] [--timeout 300]

Método del p95: con n=5 muestras se usa `statistics.quantiles(data, n=20,
method='inclusive')`, equivalente al percentil lineal interpolado de
numpy.percentile (método 'linear'): p95 = x[4] + 0.95*(x[5]-x[4]) sobre
los datos ordenados. Es el método estándar y correcto para muestras
pequeñas; el valor se compara con la métrica de latencia del dashboard
de Phoenix (ver README, sección de evidencias).
"""

import argparse
import asyncio
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 300.0  # segundos: timeout máximo de polling por job

# Escenario NO crítico: la prueba de aceptación no debe quedar bloqueada
# esperando aprobaciones humanas (la ruta HITL se valida en tests aparte).
REQUEST = {
    "user_request": (
        "Decidir si una API .NET debe ser monolito modular o microservicios. "
        "El equipo es pequeño y el dominio está en evolución."
    ),
    "critical": False,
}


async def run_one(client: httpx.AsyncClient, index: int, base_url: str, timeout_s: float) -> dict:
    """Send one task and poll until a terminal state; return timing + status."""
    started = time.perf_counter()
    try:
        response = await client.post(f"{base_url}/tasks", json=REQUEST, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        return {"index": index, "job_id": None, "status": "HTTP_ERROR",
                "error": str(error), "seconds": round(time.perf_counter() - started, 2),
                "usage": None}

    job_id = response.json()["job_id"]
    deadline = time.monotonic() + timeout_s

    status = None
    error = None
    usage = None
    while time.perf_counter() < deadline:
        await asyncio.sleep(1.0)
        try:
            detail = await client.get(f"{base_url}/tasks/{job_id}", timeout=10.0)
            detail.raise_for_status()
        except httpx.HTTPError:
            error = "HTTP error durante el polling"
            continue
        data = detail.json()
        status = data.get("status")
        if status in ("DONE", "FAILED"):
            error = data.get("error")
            usage = data.get("usage")
            break
    else:
        status = "TIMEOUT"

    elapsed = time.perf_counter() - started
    return {
        "index": index,
        "job_id": job_id,
        "status": status,
        "error": error,
        "seconds": round(elapsed, 2),
        "usage": usage,
    }


async def main() -> None:
    """Launch 5 concurrent tasks and report latency statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL de la API")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Timeout máximo de polling por job (segundos)")
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        print(f"Lanzando 5 peticiones concurrentes contra {args.url} ...")
        batch_started = time.perf_counter()
        results = await asyncio.gather(
            *(run_one(client, i, args.url, args.timeout) for i in range(1, 6))
        )
    batch_seconds = time.perf_counter() - batch_started

    print("\nResultados:")
    for result in results:
        marker = {"DONE": "OK ", "FAILED": "FAIL", "TIMEOUT": "TOUT"}.get(
            result["status"], "ERR "
        )
        print(
            f"  [{marker}] #{result['index']} {str(result['job_id'])[:8]}…  "
            f"{result['seconds']}s  {result['error'] or ''}"
        )

    ok = [r for r in results if r["status"] == "DONE"]
    seconds_list = [r["seconds"] for r in results if r["status"] == "DONE"]
    if not seconds_list:
        print("\nNingún job terminó en DONE; revisar la API y el worker.")
        return

    # p95 con interpolación lineal (equivalente a numpy.percentile 'linear').
    # Para n=5: p95 = x[4] + 0.95 * (x[5] - x[4]) sobre datos ordenados.
    p95 = statistics.quantiles(seconds_list, n=20, method="inclusive")[-1]
    print()
    print(f"jobs DONE      : {len(ok)}/5")
    print(f"latencia media : {statistics.mean(seconds_list):.2f} s")
    print(f"latencia p95   : {p95:.2f} s  (interpolación lineal, n={len(seconds_list)})")
    print(f"latencia max   : {max(seconds_list):.2f} s")
    print(f"latencia global: {batch_seconds:.2f} s")

    # Costo por ejecución (cuando el proveedor informó tokens).
    print("\nCosto por ejecución (tokens reportados por el proveedor):")
    total_cost = 0.0
    any_cost = False
    for result in results:
        usage = result.get("usage") or {}
        if not usage.get("tokens_available"):
            print(
                f"  #{result['index']}: dato no disponible "
                f"({usage.get('cost_note') or 'sin reporte de tokens'})"
            )
            continue
        cost = usage.get("estimated_cost_usd")
        if cost is None:
            print(
                f"  #{result['index']}: tokens {usage.get('total_tokens')} "
                f"({usage.get('input_tokens')} in / {usage.get('output_tokens')} out), "
                f"costo no disponible ({usage.get('cost_note')})"
            )
            continue
        any_cost = True
        total_cost += cost
        print(
            f"  #{result['index']}: ${cost:.6f}  "
            f"[{usage.get('total_tokens')} tokens: {usage.get('input_tokens')} in / "
            f"{usage.get('output_tokens')} out, modelo {usage.get('model')}]"
        )
    if any_cost:
        print(f"  TOTAL estimado: ${total_cost:.6f}")
    print(
        "\nNota: costo estimado con tarifas configurables (ADA_MODEL_RATES); "
        "Phoenix calcula el costo automáticamente solo para modelos soportados."
    )
    print("Verificar en Phoenix (http://localhost:6006) que las 5 ejecuciones")
    print("aparecen como sesiones separadas con sus trazas completas.")


if __name__ == "__main__":
    asyncio.run(main())
