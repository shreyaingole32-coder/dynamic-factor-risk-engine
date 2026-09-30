"""
Database connection and session factory supporting SQLite and PostgreSQL.
"""

import os
from typing import Generator
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base


def load_db_config(config_path: str = "configs/default.yaml") -> dict:
    """Load database settings from configuration file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("database", {})
    return {}


def get_db_engine(config_path: str = "configs/default.yaml"):
    """
    Create SQLAlchemy Engine based on configuration.
    Defaults to SQLite if PostgreSQL connection fails or dialect is set to sqlite.
    """
    db_cfg = load_db_config(config_path)
    dialect = db_cfg.get("dialect", "sqlite")

    if dialect == "postgresql":
        pg = db_cfg.get("postgres", {})
        url = f"postgresql://{pg.get('user', 'postgres')}:{pg.get('password', '')}@{pg.get('host', 'localhost')}:{pg.get('port', 5432)}/{pg.get('dbname', 'factor_risk_db')}"
        try:
            engine = create_engine(url, pool_pre_ping=True)
            # Test connection
            with engine.connect():
                return engine
        except Exception:
            print("[DB Connection Warning] PostgreSQL unavailable. Falling back to SQLite.")

    # Fallback / Default SQLite
    sqlite_path = db_cfg.get("sqlite_path", "data/processed/dynamic_factor_risk.db")
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
    return engine


def init_db(engine=None, config_path: str = "configs/default.yaml"):
    """Initialize database tables."""
    if engine is None:
        engine = get_db_engine(config_path)
    Base.metadata.create_all(bind=engine)
    return engine


def get_session_factory(engine=None, config_path: str = "configs/default.yaml") -> sessionmaker:
    """Return sessionmaker for database operations."""
    if engine is None:
        engine = get_db_engine(config_path)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
