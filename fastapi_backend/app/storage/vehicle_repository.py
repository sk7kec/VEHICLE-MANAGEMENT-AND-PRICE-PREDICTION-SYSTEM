"""
Vehicle Repository — JSON-based persistent storage

Provides thread-safe CRUD operations on a local JSON file.
Does NOT claim multi-process transactional guarantees.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import json
import uuid
import threading
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class VehicleRepository:
    """Thread-safe JSON file storage for vehicle records."""

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self):
        """Create data directory and file if they don't exist."""
        directory = os.path.dirname(self._filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self._filepath):
            self._write_records([])
            logger.info(f"Created new storage file: {self._filepath}")

    def _read_records(self) -> List[Dict[str, Any]]:
        """Read all records from storage. Handles corrupted files gracefully."""
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Storage file is empty, returning empty list")
                    return []
                data = json.loads(content)
                if not isinstance(data, list):
                    logger.warning("Storage file does not contain a list, resetting")
                    return []
                return data
        except FileNotFoundError:
            logger.warning("Storage file not found, returning empty list")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted storage file: {e}. Creating backup and resetting.")
            backup_path = self._filepath + f".corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                os.rename(self._filepath, backup_path)
                logger.info(f"Corrupted file backed up to: {backup_path}")
            except OSError:
                pass
            return []

    def _write_records(self, records: List[Dict[str, Any]]):
        """Write records to storage with safe-write (write-then-rename)."""
        directory = os.path.dirname(self._filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp_path = self._filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomic-ish rename (on Windows, need to remove target first)
            if os.path.exists(self._filepath):
                os.replace(tmp_path, self._filepath)
            else:
                os.rename(tmp_path, self._filepath)
        except Exception:
            # Clean up tmp file if rename failed
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _generate_id(self) -> str:
        """Generate a unique vehicle ID."""
        return f"VH-{uuid.uuid4().hex[:8].upper()}"

    def _now_iso(self) -> str:
        """Current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all vehicle records."""
        with self._lock:
            return self._read_records()

    def get_by_id(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a vehicle by ID. Returns None if not found."""
        with self._lock:
            records = self._read_records()
            for record in records:
                if record.get("id") == vehicle_id:
                    return record
            return None

    def create(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new vehicle record."""
        now = self._now_iso()
        record = {
            "id": self._generate_id(),
            **vehicle_data,
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            records = self._read_records()
            records.append(record)
            self._write_records(records)

        logger.info(f"Created vehicle: {record['id']}")
        return record

    def update(self, vehicle_id: str, vehicle_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace all vehicle fields (PUT). Returns updated record or None if not found."""
        with self._lock:
            records = self._read_records()
            for i, record in enumerate(records):
                if record.get("id") == vehicle_id:
                    updated = {
                        "id": vehicle_id,
                        **vehicle_data,
                        "created_at": record.get("created_at", self._now_iso()),
                        "updated_at": self._now_iso(),
                    }
                    records[i] = updated
                    self._write_records(records)
                    logger.info(f"Updated vehicle: {vehicle_id}")
                    return updated
            return None

    def patch(self, vehicle_id: str, patch_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Partially update vehicle fields (PATCH). Returns updated record or None."""
        with self._lock:
            records = self._read_records()
            for i, record in enumerate(records):
                if record.get("id") == vehicle_id:
                    for key, value in patch_data.items():
                        if value is not None:
                            record[key] = value
                    record["updated_at"] = self._now_iso()
                    records[i] = record
                    self._write_records(records)
                    logger.info(f"Patched vehicle: {vehicle_id}")
                    return record
            return None

    def delete(self, vehicle_id: str) -> bool:
        """Delete a vehicle record. Returns True if deleted, False if not found."""
        with self._lock:
            records = self._read_records()
            new_records = [r for r in records if r.get("id") != vehicle_id]
            if len(new_records) == len(records):
                return False
            self._write_records(new_records)
            logger.info(f"Deleted vehicle: {vehicle_id}")
            return True

    def count(self) -> int:
        """Return total number of vehicle records."""
        with self._lock:
            return len(self._read_records())
