# sim_run_grok_core_move_ship.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import simpy
import random

from sim_run_grok_core import TransportRoute


def _get_berth(env, port_berths: Dict[str, simpy.Resource], route: TransportRoute, location: str):
    # Initialize berth resource lazily in the passed dictionary
    if location not in port_berths:
        cap = 1
        if getattr(route, 'berth_info', None):
            try:
                cap = int((route.berth_info.get(location) or {}).get('berths', 1) or 1)
            except Exception:
                cap = 1
        port_berths[location] = simpy.Resource(env, capacity=max(1, cap))
    return port_berths[location]


def _arrival_extra_wait(env, log_func: Callable, route: TransportRoute, location: str):
    info = (getattr(route, 'berth_info', {}) or {}).get(location, {})
    pocc = float(info.get('p_occupied', 0.0) or 0.0)
    pilot_in_h = float(info.get('pilot_in_h', 0.0) or 0.0)
    if pocc > 0.0 and random.random() < pocc and pilot_in_h > 0:
        log_func("BerthOccupiedWait", mode="SHIP", location=location, wait_h=pilot_in_h, p_occupied=pocc)
        return env.timeout(pilot_in_h)
    return env.timeout(0)


def _pilot(env, log_func: Callable, route: TransportRoute, location: str, phase: str):
    info = (getattr(route, 'berth_info', {}) or {}).get(location, {})
    hours = float(info.get('pilot_in_h' if phase == 'in' else 'pilot_out_h', 0.0) or 0.0)
    if hours > 0:
        log_func("PilotIn" if phase == 'in' else "PilotOut", mode="SHIP", location=location, hours=hours)
        return env.timeout(hours)
    return env.timeout(0)


def _store_rate(store_rates: Dict[str, Tuple[float, float]], store_key: str, default_rate: float):
    try:
        if store_key in store_rates:
            # rates: (load, unload). Return whichever is not None, else default
            return float(store_rates[store_key][0] or default_rate), float(store_rates[store_key][1] or default_rate)
    except Exception:
        pass
    return default_rate, default_rate


def _nm(route: TransportRoute, a: str, b: str) -> float:
    if getattr(route, 'nm_distance', None):
        try:
            return float(route.nm_distance.get((a, b), 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def transporter(env, route: TransportRoute,
                stores: Dict[str, simpy.Container],
                port_berths: Dict[str, simpy.Resource],
                log_func: Callable,
                store_rates: Dict[str, Tuple[float, float]],
                require_full: bool = True):
    """
    Itinerary-based SHIP transporter.
    Refactored to be pure: explicitly accepts port_berths and store_rates.
    """
    # Fallback path if no itineraries present
    if not getattr(route, 'itineraries', None):
        while True:
            if route.payload_t <= 0:
                yield env.timeout(1)
                continue
            yield env.timeout(1)
        return

    # Initialize vessel state
    speed_knots = float(getattr(route, 'speed_knots', 0.0) or 0.0)
    payload_t = float(getattr(route, 'payload_t', 0.0) or 0.0)

    # Determine the starting location
    def start_loc_from_it(it):
        for st in it:
            if st.get('kind') == 'start':
                return st.get('location')
        return None

    current_location = start_loc_from_it(route.itineraries[0]) or route.origin_location
    cargo = {}

    while True:
        # Pick an itinerary that starts at current_location
        candidate_its = [it for it in route.itineraries if (start_loc_from_it(it) or '') == current_location]
        if not candidate_its:
            current_location = start_loc_from_it(route.itineraries[0]) or current_location
            yield env.timeout(1)
            continue

        chosen = None
        # Feasibility check
        for it in candidate_its:
            origin = start_loc_from_it(it)
            total_stock = 0.0
            for st in it:
                if st.get('kind') == 'load' and st.get('location') == origin and st.get('store_key'):
                    sk = st['store_key']
                    total_stock += float(stores[sk].level)
            if not require_full or total_stock + 1e-6 >= payload_t:
                chosen = it
                break
        if chosen is None:
            yield env.timeout(1)
            continue

        log_func("ShipChooseItinerary", mode="SHIP", route_group=getattr(route, 'route_group', None),
                 route_id=chosen[1].get('route_id') if len(chosen) > 1 else None, start=current_location)

        idx = 0
        while idx < len(chosen) and chosen[idx].get('kind') != 'start':
            idx += 1
        if idx < len(chosen) and chosen[idx].get('kind') == 'start':
            idx += 1

        while idx < len(chosen):
            step = chosen[idx]
            k = step.get('kind')
            if k == 'sail':
                a = step.get('from') or current_location
                b = step.get('to')
                if not b:
                    idx += 1
                    continue
                nm = _nm(route, a, b)
                if nm <= 0 or speed_knots <= 0:
                    log_func("warn_missing_distance", mode="SHIP", frm=a, to=b)
                    break
                hours = nm / max(speed_knots, 1e-6)
                log_func("Sailed", mode="SHIP", frm=a, to=b, nm=nm, hours=hours, speed_knots=speed_knots)
                yield env.timeout(hours)
                current_location = b
                idx += 1
                continue
            elif k in ('load', 'unload'):
                loc = step.get('location') or current_location
                store_key = step.get('store_key')
                product = step.get('product')
                if not store_key or store_key not in stores:
                    idx += 1
                    continue

                berth = _get_berth(env, port_berths, route, loc)
                yield _arrival_extra_wait(env, log_func, route, loc)

                with berth.request() as req:
                    yield req
                    yield _pilot(env, log_func, route, loc, 'in')

                    load_rate, unload_rate = _store_rate(store_rates, store_key,
                                                         route.load_rate_tph if k == 'load' else route.unload_rate_tph)
                    if k == 'load':
                        cont = stores[store_key]
                        remaining_cap = max(0.0, payload_t - sum(cargo.values()))
                        if remaining_cap > 1e-6 and cont.level > 1e-6:
                            qty = min(remaining_cap, float(cont.level))
                            eff_rate = max(float(load_rate), 1e-6)
                            t_h = qty / eff_rate
                            yield env.timeout(t_h)
                            yield cont.get(qty)
                            cargo[product] = cargo.get(product, 0.0) + qty
                            log_func("ShipLoad", mode="SHIP", location=loc, product=product, store_key=store_key,
                                     qty=qty, rate_tph=eff_rate)
                    else:
                        # unload
                        cont = stores[store_key]
                        carried = float(cargo.get(product, 0.0))
                        if carried > 1e-6:
                            room = float(cont.capacity - cont.level)
                            if room > 1e-6:
                                qty = min(carried, room)
                                eff_rate = max(float(unload_rate), 1e-6)
                                t_h = qty / eff_rate
                                yield env.timeout(t_h)
                                yield cont.put(qty)
                                cargo[product] = carried - qty
                                log_func("ShipUnload", mode="SHIP", location=loc, product=product, store_key=store_key,
                                         qty=qty, rate_tph=eff_rate)

                    yield _pilot(env, log_func, route, loc, 'out')
                idx += 1
                continue
            else:
                idx += 1
                continue
        yield env.timeout(0.01)