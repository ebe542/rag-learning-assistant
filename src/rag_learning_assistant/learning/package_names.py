"""Shared SQLite name reservations for package lifecycle records."""

import sqlite3
from uuid import UUID


def initialize_name_registry(connection: sqlite3.Connection) -> None:
    """Create the cross-table package-name registry."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS package_names (
            name TEXT PRIMARY KEY COLLATE NOCASE,
            owner_kind TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            UNIQUE (owner_kind, owner_id)
        )
        """
    )


def ensure_name_reservation(
    connection: sqlite3.Connection,
    *,
    name: str,
    owner_kind: str,
    owner_id: UUID,
) -> None:
    """Reserve a name or accept its existing reservation by the same owner."""

    row = connection.execute(
        "SELECT owner_kind, owner_id FROM package_names WHERE name = ? COLLATE NOCASE",
        (name,),
    ).fetchone()
    identity = (owner_kind, str(owner_id))
    if row is not None:
        if tuple(row) == identity:
            return
        raise ValueError(f"Learning package already exists: {name}")
    connection.execute(
        "INSERT INTO package_names (name, owner_kind, owner_id) VALUES (?, ?, ?)",
        (name, *identity),
    )


def release_name_reservation(
    connection: sqlite3.Connection,
    *,
    owner_kind: str,
    owner_id: UUID,
) -> None:
    """Release only the reservation owned by the supplied lifecycle record."""

    connection.execute(
        "DELETE FROM package_names WHERE owner_kind = ? AND owner_id = ?",
        (owner_kind, str(owner_id)),
    )
