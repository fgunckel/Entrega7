"""Entry point: python -m app.worker

Long-lived worker process: consumes the Redis Streams (new jobs and HITL
resumes) and executes the ADA graph. Run it alongside the API (see
docker-compose.yml and README).
"""

import asyncio

from app.worker import run_worker

if __name__ == "__main__":
    asyncio.run(run_worker())
