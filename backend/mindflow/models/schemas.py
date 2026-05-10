from datetime import datetime, date, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Date,
    ForeignKey, Index, JSON, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from mindflow.models.database import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    preferences = Column(JSON, nullable=True)

    activities = relationship("ActivityLog", back_populates="user", lazy="dynamic")
    focus_sessions = relationship("FocusSession", back_populates="user", lazy="dynamic")
    daily_reports = relationship("DailyReport", back_populates="user", lazy="dynamic")


class ActivityLog(Base):
    __tablename__ = "activity_logs"
    __table_args__ = (
        Index("idx_activity_user_time", "user_id", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=_utcnow)
    process_name = Column(String(255), nullable=False)
    window_title = Column(Text, nullable=True)
    window_class = Column(String(255), nullable=True)
    duration_seconds = Column(Float, default=0.0)
    is_idle = Column(Integer, default=0)

    user = relationship("User", back_populates="activities")


class FocusSession(Base):
    __tablename__ = "focus_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    focus_score = Column(Float, nullable=True)
    session_type = Column(String(50), nullable=True)
    dominant_app = Column(String(255), nullable=True)

    user = relationship("User", back_populates="focus_sessions")


class DailyReport(Base):
    __tablename__ = "daily_reports"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_daily_report_user_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    total_focus_minutes = Column(Float, default=0.0)
    total_distraction_minutes = Column(Float, default=0.0)
    focus_score = Column(Float, default=0.0)
    top_apps = Column(JSON, nullable=True)
    switch_frequency = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="daily_reports")
