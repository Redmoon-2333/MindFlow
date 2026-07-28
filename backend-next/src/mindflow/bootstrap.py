"""CLI helper that obtains a one-time UI bootstrap ticket from the local API."""

from __future__ import annotations

import asyncio

import httpx

from mindflow.config import Settings, get_settings


async def request_bootstrap_url(settings: Settings) -> str:
    """Authenticate as the local launcher and return a fragment-based UI URL."""
    token = settings.token_path.read_text(encoding="utf-8").strip()
    # Never send the root launcher token to a configured hostname. The server
    # is local-only; wildcard/IPv4 binds are reached through IPv4 loopback,
    # while explicit IPv6 binds use the bracketed loopback literal.
    host = "[::1]" if settings.host in {"::", "::1", "[::1]"} else "127.0.0.1"
    base_url = f"http://{host}:{settings.port}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{base_url}/api/v1/auth/bootstrap/ticket",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
    ticket = str(response.json()["ticket"])
    return f"{base_url}/#bootstrap={ticket}"


async def _async_main() -> None:
    print(await request_bootstrap_url(get_settings()))


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
