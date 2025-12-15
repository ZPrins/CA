# sim_run_grok_core_move_ship.py
from __future__ import annotations
import simpy
import random

from sim_run_grok_core import TransportRoute


def _get_berth(sim, route: TransportRoute, location: str):
    # Initialize berth resource lazily based on route.berth_info
    if location not in sim.port_berths:
        cap = 1
        if getattr(route, 'berth_info', None):
            try:
                cap = int((route.berth_info.get(location) or {}).get('berths', 1) or 1)
            except Exception:
                cap = 1
        sim.port_berths[location] = simpy.Resource(sim.env, capacity=max(1, cap))
    return sim.port_berths[location]


def _arrival_extra_wait(sim, route: TransportRoute, location: str):
    info = (getattr(route, 'berth_info', {}) or {}).get(location, {})
    pocc = float(info.get('p_occupied', 0.0) or 0.0)
    pilot_in_h = float(info.get('pilot_in_h', 0.0) or 0.0)
    if pocc > 0.0 and random.random() < pocc and pilot_in_h > 0:
        sim.log("BerthOccupiedWait", mode="SHIP", location=location, wait_h=pilot_in_h, p_occupied=pocc)
        return sim.env.timeout(pilot_in_h)
    return sim.env.timeout(0)


def _pilot(sim, route: TransportRoute, location: str, phase: str):
    info = (getattr(route, 'berth_info', {}) or {}).get(location, {})
    hours = float(info.get('pilot_in_h' if phase == 'in' else 'pilot_out_h', 0.0) or 0.0)
    if hours > 0:
        sim.log("PilotIn" if phase == 'in' else "PilotOut", mode="SHIP", location=location, hours=hours)
        return sim.env.timeout(hours)
    return sim.env.timeout(0)


def _store_rate(sim, store_key: str, default_rate: float):
    try:
        rates = sim.settings.get('store_rates') or {}
        if store_key in rates:
            # rates: (load, unload)
            # caller decides which component; here we return whichever is not None, else default
            return float(rates[store_key][0] or default_rate), float(rates[store_key][1] or default_rate)
    except Exception:
        pass
    return default_rate, default_rate


def _nm(sim, route: TransportRoute, a: str, b: str) -> float:
    if getattr(route, 'nm_distance', None):
        try:
            return float(route.nm_distance.get((a, b), 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def transporter(sim, route: TransportRoute):
    """
    Itinerary-based SHIP transporter with berth concurrency, pilot times, and sailing times.
    If no itineraries are provided, falls back to simple TRAIN-like behavior.
    Uses sim.env, sim.stores, sim.log, sim.port_berths, and per-store rates from settings.
    """
    # Fallback path if no itineraries present
    if not getattr(route, 'itineraries', None):
        while True:
            if route.payload_t <= 0:
                yield sim.env.timeout(1)
                continue
            # Choose any available origin/dest stores? We don't have them for SHIP enriched routes; just idle
            yield sim.env.timeout(1)
        return

    # Initialize vessel state
    speed_knots = float(getattr(route, 'speed_knots', 0.0) or 0.0)
    payload_t = float(getattr(route, 'payload_t', 0.0) or 0.0)
    require_full = bool(sim.settings.get("require_full_payload", True))

    # Determine the starting location from first itinerary's start, else route.origin_location
    def start_loc_from_it(it):
        for st in it:
            if st.get('kind') == 'start':
                return st.get('location')
        return None

    current_location = start_loc_from_it(route.itineraries[0]) or route.origin_location

    # Cargo manifest on board
    cargo = {}

    while True:
        # Pick an itinerary that starts at current_location
        candidate_its = [it for it in route.itineraries if (start_loc_from_it(it) or '') == current_location]
        if not candidate_its:
            # If none match, reset to the first itinerary's start
            current_location = start_loc_from_it(route.itineraries[0]) or current_location
            yield sim.env.timeout(1)
            continue

        chosen = None
        # Simple feasibility: check if at least one load step at origin can proceed (stock available)
        for it in candidate_its:
            # compute available load at origin
            origin = start_loc_from_it(it)
            total_stock = 0.0
            # look through steps at origin (load kind with location=origin)
            for st in it:
                if st.get('kind') == 'load' and st.get('location') == origin and st.get('store_key'):
                    sk = st['store_key']
                    total_stock += float(sim.stores[sk].level)
            if not require_full or total_stock + 1e-6 >= payload_t:
                chosen = it
                break
        if chosen is None:
            # wait for stock
            yield sim.env.timeout(1)
            continue

        # Execute chosen itinerary
        sim.log("ShipChooseItinerary", mode="SHIP", route_group=getattr(route, 'route_group', None), route_id=chosen[1].get('route_id') if len(chosen) > 1 else None, start=current_location)

        idx = 0
        # identify start marker
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
                nm = _nm(sim, route, a, b)
                if nm <= 0 or speed_knots <= 0:
                    # missing distance or speed — skip this itinerary instance
                    sim.log("warn_missing_distance", mode="SHIP", frm=a, to=b)
                    break
                hours = nm / max(speed_knots, 1e-6)
                sim.log("Sailed", mode="SHIP", frm=a, to=b, nm=nm, hours=hours, speed_knots=speed_knots)
                yield sim.env.timeout(hours)
                current_location = b
                idx += 1
                continue
            elif k in ('load', 'unload'):
                loc = step.get('location') or current_location
                store_key = step.get('store_key')
                product = step.get('product')
                if not store_key or store_key not in sim.stores:
                    # unresolved store; skip
                    idx += 1
                    continue
                berth = _get_berth(sim, route, loc)
                # arrival-based extra wait
                yield _arrival_extra_wait(sim, route, loc)
                # request berth
                with berth.request() as req:
                    yield req
                    # pilot in
                    yield _pilot(sim, route, loc, 'in')
                    # perform operation
                    load_rate, unload_rate = _store_rate(sim, store_key, route.load_rate_tph if k == 'load' else route.unload_rate_tph)
                    if k == 'load':
                        # amount to load this cycle
                        cont = sim.stores[store_key]
                        remaining_cap = max(0.0, payload_t - sum(cargo.values()))
                        if require_full and remaining_cap + 1e-6 < payload_t:
                            # starting mid-itinerary; allow partial unless setting forbids; here we allow proceeding
                            pass
                        if remaining_cap <= 1e-6 or cont.level <= 1e-6:
                            # nothing to do
                            pass
                        else:
                            # Load up to rate-limited amount; we simulate in one chunk using time = qty / rate
                            qty = min(remaining_cap, float(cont.level))
                            eff_rate = max(float(load_rate), 1e-6)
                            t_h = qty / eff_rate
                            yield sim.env.timeout(t_h)
                            yield cont.get(qty)
                            cargo[product] = cargo.get(product, 0.0) + qty
                            sim.log("ShipLoad", mode="SHIP", location=loc, product=product, store_key=store_key, qty=qty, rate_tph=eff_rate)
                    else:
                        # unload
                        cont = sim.stores[store_key]
                        carried = float(cargo.get(product, 0.0))
                        if carried <= 1e-6:
                            pass
                        else:
                            room = float(cont.capacity - cont.level)
                            if room <= 1e-6:
                                pass
                            else:
                                qty = min(carried, room)
                                eff_rate = max(float(unload_rate), 1e-6)
                                t_h = qty / eff_rate
                                yield sim.env.timeout(t_h)
                                yield cont.put(qty)
                                cargo[product] = carried - qty
                                sim.log("ShipUnload", mode="SHIP", location=loc, product=product, store_key=store_key, qty=qty, rate_tph=eff_rate)
                    # pilot out
                    yield _pilot(sim, route, loc, 'out')
                idx += 1
                continue
            else:
                # unknown or start marker
                idx += 1
                continue
        # End of itinerary — we loop; current_location is where we ended (Return Location)
        # Loop continues to select next itinerary starting at current_location
        yield sim.env.timeout(0.01)
