# sim_run_core_move_ship_types.py
"""
Ship transport types and state definitions.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any
import simpy


class ShipState(Enum):
    """Ship state machine states."""
    IDLE = "IDLE"
    LOADING = "LOADING"
    WAITING_FOR_PRODUCT = "WAITING_FOR_PRODUCT"
    IN_TRANSIT = "IN_TRANSIT"
    WAITING_FOR_BERTH = "WAITING_FOR_BERTH"
    UNLOADING = "UNLOADING"
    WAITING_FOR_SPACE = "WAITING_FOR_SPACE"
    ERROR = "ERROR"


@dataclass
class ShipContext:
    """
    Context object holding all ship state and configuration.
    Passed between state handlers to maintain state across function calls.
    """
    env: simpy.Environment
    stores: Dict[str, simpy.Container]
    port_berths: Dict[str, simpy.Resource]
    log_func: Callable
    store_rates: Dict[str, Tuple[float, float]]

    # Route configuration
    speed_knots: float = 10.0
    n_holds: int = 1
    payload_per_hold: float = 25000.0
    route_group: str = "Ship"
    berth_info: Dict[str, dict] = field(default_factory=dict)
    nm_distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    default_load_rate: float = 500.0
    default_unload_rate: float = 500.0
    max_wait_product_h: float = 24.0
    min_utilization: float = 0.60

    # Vessel identity
    vessel_id: int = 1
    origin_location: str = ""

    # Current state
    current_location: str = ""
    state: ShipState = ShipState.IDLE
    cargo: Dict[str, float] = field(default_factory=dict)
    chosen_itinerary: Optional[List[Dict]] = None
    itinerary_idx: int = 0
    current_route_id: Optional[str] = None

    # Berth tracking
    active_berth: Optional[simpy.Resource] = None
    active_berth_req: Optional[Any] = None

    # Wait tracking
    total_waited_at_location: float = 0.0
    last_prob_wait_location: Optional[str] = None
    planned_load_remaining: float = 0.0

    # External references
    demand_rates: Dict[str, float] = field(default_factory=dict)
    sole_supplier_stores: Optional[set] = None
    production_rates: Optional[Dict[str, float]] = None
    store_capacity_map: Optional[Dict[str, float]] = None
    sim: Any = None
    t_state: Optional[dict] = None
    itineraries: List[List[Dict]] = field(default_factory=list)

    def log_state_change(self, new_state: ShipState, location: str = None):
        """Log a state change event."""
        self.log_func(
            process="ShipState",
            event="StateChange",
            location=location or self.current_location,
            equipment="Ship",
            product=None,
            qty=None,
            unmet_demand=0.0,
            from_store=None,
            from_level=None,
            to_store=None,
            to_level=None,
            route_id=self.current_route_id or self.route_group,
            vessel_id=self.vessel_id,
            ship_state=new_state.value,
            override_time_h=self.env.now
        )

