import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.models.entities import Base

logger = logging.getLogger(__name__)

# Determine engine parameters based on database type (SQLite vs PostgreSQL)
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

connect_args = {"check_same_thread": False} if is_sqlite else {"connect_timeout": 5}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key enforcement for SQLite."""
    if is_sqlite:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables for SQLite or PostgreSQL."""
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_schema_columns()
        logger.info(f"Database initialized successfully using: {settings.DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _ensure_schema_columns() -> None:
    """Add missing columns for databases created before the current models."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    column_specs = {
        "exam_attempts": {
            "time_spent_seconds": "INTEGER DEFAULT 0",
            "duration_seconds": "INTEGER DEFAULT 0",
            "is_auto_submitted": "BOOLEAN DEFAULT FALSE",
            "ai_insights": "TEXT",
        },
        "answers": {
            "is_marked_review": "BOOLEAN DEFAULT FALSE",
            "time_spent_seconds": "INTEGER DEFAULT 0",
        },
    }

    with engine.begin() as conn:
        for table_name, columns in column_specs.items():
            if table_name not in existing_tables:
                continue

            existing_columns = {
                column["name"]
                for column in inspector.get_columns(table_name)
            }

            for column_name, column_definition in columns.items():
                if column_name in existing_columns:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"))


def test_connection() -> bool:
    """Verify database connection health."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for handling database sessions safely."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        session.close()
