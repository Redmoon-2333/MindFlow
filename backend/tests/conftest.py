import pytest
from datetime import datetime, timezone

from mindflow.models.database import SessionLocal, init_db, Base, engine
from mindflow.models.schemas import User


@pytest.fixture(scope="function")
def db_session():
    init_db()
    db = SessionLocal()
    try:
        yield db
        db.rollback()
    finally:
        db.close()


@pytest.fixture(scope="function")
def test_user(db_session):
    user = db_session.query(User).first()
    if user is None:
        user = User(username="test_user", preferences={})
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user
