import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.models.entities import Base

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_session_factory = None
_sqlite_pragma_attached = False


def _is_sqlite() -> bool:
    return settings.DATABASE_URL.startswith("sqlite")


def get_engine() -> Engine:
    """Create the SQLAlchemy engine lazily so config errors can render in Streamlit."""
    global _engine, _session_factory, _sqlite_pragma_attached
    if _engine is not None:
        return _engine

    is_sqlite = _is_sqlite()
    connect_args = {"check_same_thread": False} if is_sqlite else {"connect_timeout": 5}
    _engine = create_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.DEBUG,
    )
    _session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
        expire_on_commit=False,
    )

    if is_sqlite and not _sqlite_pragma_attached:
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            """Enable foreign key enforcement for SQLite."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        _sqlite_pragma_attached = True

    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        get_engine()
    return _session_factory


def init_db() -> None:
    """Initialize database tables for SQLite or PostgreSQL."""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        _ensure_schema_columns()
        logger.info(f"Database initialized successfully using: {settings.DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _ensure_schema_columns() -> None:
    """Add missing columns for databases created before the current models."""
    engine = get_engine()
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
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for handling database sessions safely."""
    session: Session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        session.close()
