from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from mindflow.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_data_dir():
    db_path = settings.database_url.replace("sqlite:///", "")
    if not Path(db_path).is_absolute():
        db_path = str(_PROJECT_ROOT / db_path)
    data_dir = Path(db_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


_ensure_data_dir()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db():
    _ensure_data_dir()
    from mindflow.models.schemas import User, ActivityLog, FocusSession, DailyReport  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
