"""Servidor Redis en memoria (fakeredis TCP) para pruebas locales sin Docker.

Uso: uv run python .coda/tmp/fake_redis_server.py [puerto]

Mantiene un TcpFakeServer de fakeredis escuchando en 127.0.0.1:<puerto>.
API y worker se conectan con redis://localhost:<puerto>/0 como si fuera
Redis real. Solo para desarrollo/pruebas: los datos no persisten.
"""

import sys

from fakeredis import TcpFakeServer

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6390
    server = TcpFakeServer(("127.0.0.1", port), server_type="redis")
    print(f"fakeredis TCP escuchando en 127.0.0.1:{port}", flush=True)
    server.serve_forever()
