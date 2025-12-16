# sim_run_core_move_ship.py
from __future__ import annotations
from typing import Callable, Dict, Tuple
import simpy
import random
from sim_run_types import TransportRoute


def _get_berth(env, port_berths: Dict[str, simpy.Resource], route: TransportRoute, location: str):
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
        return env.timeout(pilot_in_h)
    return env.timeout(0)


def _pilot(env, log_func: Callable, route: TransportRoute, location: str, phase: str):
    info = (getattr(route, 'berth_info', {}) or {}).get(location, {})
    hours = float(info.get('pilot_in_h' if phase == 'in' else 'pilot_out_h', 0.0) or 0.0)
    if hours > 0:
        return env.timeout(hours)
    return env.timeout(0)


def _store_rate(store_rates: Dict[str, Tuple[float, float]], store_key: str, default_rate: float):
    try:
        if store_key in store_rates:
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
    if not getattr(route, 'itineraries', None):
        yield env.timeout(1)
        return

    speed_knots = float(getattr(route, 'speed_knots', 0.0) or 0.0)
    payload_t = float(getattr(route, 'payload_t', 0.0) or 0.0)
    route_grp = getattr(route, 'route_group', 'Ship')

    def start_loc_from_it(it):
        for st in it:
            if st.get('kind') == 'start': return st.get('location')
        return None

    current_location = start_loc_from_it(route.itineraries[0]) or route.origin_location
    cargo = {}

    while True:
        candidate_its = [it for it in route.itineraries if (start_loc_from_it(it) or '') == current_location]
        if not candidate_its:
            current_location = start_loc_from_it(route.itineraries[0]) or current_location
            yield env.timeout(1)
            continue

        chosen = None
        for it in candidate_its:
            origin = start_loc_from_it(it)
            total_stock = 0.0
            for st in it:
                if st.get('kind') == 'load' and st.get('location') == origin and st.get('store_key'):
                    sk = st['store_key']
                    if sk in stores: total_stock += float(stores[sk].level)
            if not require_full or total_stock + 1e-6 >= payload_t:
                chosen = it
                break

        if chosen is None:
            yield env.timeout(1)
            continue

        idx = 0
        while idx < len(chosen) and chosen[idx].get('kind') != 'start': idx += 1
        if idx < len(chosen) and chosen[idx].get('kind') == 'start': idx += 1

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
                hours = nm / max(speed_knots, 1e-6)
                yield env.timeout(hours)
                current_location = b
                idx += 1
                continue

            elif k in ('load', 'unload'):
                loc = step.get('location') or current_location
                store_key = step.get('store_key')
                product = step.get('product') or step.get('Product_Class')

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

                            yield cont.get(qty)
                            from_bal = cont.level
                            yield env.timeout(qty / eff_rate)

                            cargo[product] = cargo.get(product, 0.0) + qty

                            log_func(
                                process="Move",
                                event="Load",
                                location=loc,
                                equipment="Ship",
                                product=product,
                                qty=qty,
                                from_store=store_key,
                                from_level=from_bal,
                                to_store=None,
                                to_level=None,
                                route_id=route_grp
                            )

                    else:  # unload
                        cont = stores[store_key]
                        carried = float(cargo.get(product, 0.0))
                        if carried > 1e-6:
                            room = float(cont.capacity - cont.level)
                            if room > 1e-6:
                                qty = min(carried, room)
                                eff_rate = max(float(unload_rate), 1e-6)

                                yield env.timeout(qty / eff_rate)
                                yield cont.put(qty)
                                to_bal = cont.level

                                cargo[product] = carried - qty

                                log_func(
                                    process="Move",
                                    event="Unload",
                                    location=loc,
                                    equipment="Ship",
                                    product=product,
                                    qty=qty,
                                    from_store=None,
                                    from_level=None,
                                    to_store=store_key,
                                    to_level=to_bal,
                                    route_id=route_grp
                                )

                    yield _pilot(env, log_func, route, loc, 'out')
                idx += 1
                continue
            else:
                idx += 1
                continue
        yield env.timeout(0.01)