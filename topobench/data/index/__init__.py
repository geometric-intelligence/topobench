"""Structure index backends for transductive learning."""

from topobench.data.index.base import AbstractIndexBackend
from topobench.data.index.sqlite_backend import SQLiteIndexBackend

__all__ = ["AbstractIndexBackend", "SQLiteIndexBackend"]
