"""Alembic environment configuration.

Critical settings:
  - render_as_batch=True: required for SQLite ALTER TABLE support
    (SQLite does not support ALTER COLUMN; batch mode recreates tables)
  - target_metadata = None (no declarative Base yet; migrations use
    raw schema operations from the architecture doc)

URL resolution:
  When ``sqlalchemy.url`` in alembic.ini is left empty (the default),
  env.py derives a synchronous SQLite URL from the application's
  ``Settings`` object (see :func:`_resolve_sync_db_url`).  This lets
  ``uv run python -m alembic upgrade head`` work without any
  user configuration — the URL is sourced from the same resolution
  chain (env var → .env → defaults) that the running app uses.

  Explicit ``-x`` config paths or a non-empty ``sqlalchemy.url`` in
  the ini file take precedence over the automatic resolution.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No declarative Base yet — using raw operations
target_metadata = None


# ── URL resolution ──────────────────────────────────────────────────────

_ASYNC_TO_SYNC_DRIVERS = {
    "sqlite+aiosqlite": "sqlite+pysqlite",
    "postgresql+asyncpg": "postgresql+psycopg",
}


def _resolve_sync_db_url() -> str | None:
    """Derive a synchronous database URL from application ``Settings``.

    Called only when ``sqlalchemy.url`` in the ini section is empty
    (the default committed state).  Returns ``None`` so callers can
    distinguish "explicit empty from ini" from "not yet resolved".

    The conversion replaces async driver names (e.g. ``+aiosqlite``)
    with their sync counterparts so that Alembic's synchronous
    ``Engine`` can connect.
    """
    from mindflow.config import get_settings

    settings = get_settings()
    url = settings.db_url

    for async_driver, sync_driver in _ASYNC_TO_SYNC_DRIVERS.items():
        if url.startswith(async_driver):
            return url.replace(async_driver, sync_driver, 1)
    return url


# If the ini file does not supply a URL, resolve it from Settings.
_ini_url = config.get_main_option("sqlalchemy.url")
if not _ini_url:
    _resolved = _resolve_sync_db_url()
    if _resolved is not None:
        config.set_main_option("sqlalchemy.url", _resolved)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given SQL rather than
    actually executing it.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine from the config URL and associates a connection
    with the migration context. render_as_batch=True is critical for
    SQLite compatibility.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
