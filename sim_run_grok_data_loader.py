# data_loader.py
import pandas as pd
from sim_run_grok_core import StoreConfig, ProductionCandidate, MakeUnit, TransportRoute, Demand

EXCEL_FILE = "generated_model_inputs.xlsx"

def load_data():
    store_df = pd.read_excel(EXCEL_FILE, sheet_name="Store")
    make_df = pd.read_excel(EXCEL_FILE, sheet_name="Make")
    move_df = pd.read_excel(EXCEL_FILE, sheet_name="Move")
    deliver_df = pd.read_excel(EXCEL_FILE, sheet_name="Deliver")
    network_df = pd.read_excel(EXCEL_FILE, sheet_name="Network")

    settings = {
        "horizon_days": 365,
        "random_opening": True,
        "random_seed": None,
    }

    # Stores
    store_lookup = {}
    stores = []
    for _, row in store_df.iterrows():
        loc = row["Location"]
        eq = row["Equipment Name"]
        prod = row["Input"]
        if pd.isna(prod):
            continue
        key = f"{prod}|{loc}|{eq}|{prod}"
        store_lookup[(loc, eq, prod)] = key
        stores.append(StoreConfig(
            key=key,
            capacity=float(row["Silo Max Capacity"]),
            opening_low=float(row["Silo Opening Stock (Low)"]),
            opening_high=float(row["Silo Opening Stock (High)"])
        ))

    # Makes - with proper multi-store output
    makes = []
    for _, make_row in make_df.iterrows():
        loc = make_row["Location"]
        eq = make_row["Equipment Name"]
        inp = make_row["Input"] if pd.notna(make_row["Input"]) else None
        outp = make_row["Output"]

        in_key = store_lookup.get((loc, eq, inp)) if inp else None

        # Find output stores from Network
        output_candidates = []
        make_rows = network_df[
            (network_df["Process"] == "Make") &
            (network_df["Location"] == loc) &
            (network_df["Equipment Name"] == eq) &
            (network_df["Output"] == outp)
        ]
        for _, nr in make_rows.iterrows():
            next_eq = nr["Next Equipment"]
            next_loc = nr["Next Location"]
            key = store_lookup.get((next_loc, next_eq, outp))
            if key:
                output_candidates.append(key)

        if not output_candidates:
            for (sl, se, sp) in store_lookup:
                if sl == loc and sp == outp:
                    output_candidates.append(store_lookup[(sl, se, sp)])
            output_candidates = list(set(output_candidates))

        candidates = []
        for out_key in output_candidates:
            candidates.append(ProductionCandidate(
                product=outp,
                out_store_key=out_key,
                in_store_key=in_key,
                rate_tph=float(make_row["Mean Production Rate (Tons/hr)"]),
                consumption_pct=float(make_row.get("Consumption %", 1.0))
            ))

        makes.append(MakeUnit(
            location=loc,
            equipment=eq,
            candidates=candidates
        ))

    # Moves - flexible
    moves = []
    for _, row in move_df.iterrows():
        product = row["Product"]
        origin_loc = row["Location"]
        dest_loc = row["Next Location"]

        origin_stores = [key for (loc, eq, prod) in store_lookup if loc == origin_loc and prod == product]
        dest_stores = [key for (loc, eq, prod) in store_lookup if loc == dest_loc and prod == product]

        if not origin_stores or not dest_stores:
            print(f"Warning: Incomplete route {product} {origin_loc} → {dest_loc}")
            continue

        payload = row["#Parcels"] * row["Capacity Per Parcel"]
        n_units = 99 if pd.isna(row["#Equipment \n(99-unlimited)"]) or row["#Equipment \n(99-unlimited)"] >= 99 else int(row["#Equipment \n(99-unlimited)"])

        moves.append(TransportRoute(
            product=product,
            origin_location=origin_loc,
            dest_location=dest_loc,
            origin_stores=origin_stores,
            dest_stores=dest_stores,
            n_units=n_units,
            payload_t=float(payload),
            load_rate_tph=float(row["Load Rate (Ton/hr)"] or 500),
            unload_rate_tph=float(row["Unload Rate (Ton/Hr)"] or 400),
            to_min=float(row["Travel to Time (Min)"]),
            back_min=float(row["Travel back Time (Min)"])
        ))

    # Demands
    demands = []
    for _, row in deliver_df.iterrows():
        loc = row["Location"]
        prod = row["Input"]
        possible = [key for (l, e, p) in store_lookup if l == loc and p == prod]
        if possible:
            demands.append(Demand(store_key=possible[0], rate_per_hour=float(row["Demand per Location/Hour"])))

    return settings, stores, makes, moves, demands