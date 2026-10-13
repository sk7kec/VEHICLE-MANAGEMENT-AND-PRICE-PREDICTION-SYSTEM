"""
Vehicle Service — Business logic for vehicle CRUD operations.

Author: Sahishnakumaar S S (24ADR134)
"""

import logging
from typing import List, Optional, Dict, Any

from ..storage.vehicle_repository import VehicleRepository

logger = logging.getLogger(__name__)


class VehicleService:
    """Business logic layer for vehicle operations."""

    def __init__(self, repository: VehicleRepository):
        self._repo = repository

    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Get all vehicle records."""
        return self._repo.get_all()

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get a vehicle by ID."""
        return self._repo.get_by_id(vehicle_id)

    def create_vehicle(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new vehicle record."""
        # Ensure previous_owners has a default
        if "previous_owners" not in vehicle_data or vehicle_data["previous_owners"] is None:
            vehicle_data["previous_owners"] = 0
        return self._repo.create(vehicle_data)

    def update_vehicle(self, vehicle_id: str, vehicle_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace a vehicle record (PUT)."""
        if "previous_owners" not in vehicle_data or vehicle_data["previous_owners"] is None:
            vehicle_data["previous_owners"] = 0
        return self._repo.update(vehicle_id, vehicle_data)

    def patch_vehicle(self, vehicle_id: str, patch_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Partially update a vehicle record (PATCH)."""
        # Filter out None values
        filtered = {k: v for k, v in patch_data.items() if v is not None}
        if not filtered:
            # Return existing record unchanged if no fields to update
            return self._repo.get_by_id(vehicle_id)
        return self._repo.patch(vehicle_id, filtered)

    def delete_vehicle(self, vehicle_id: str) -> bool:
        """Delete a vehicle record."""
        return self._repo.delete(vehicle_id)

    def count_vehicles(self) -> int:
        """Get total count of vehicles."""
        return self._repo.count()
