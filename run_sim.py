"""
Run the SimPy model built from a generated workbook without using the command line.

Usage:
- Edit settings in sim_config.py
- Double-click this file (Windows), or run `python run_sim.py`

This script will:
1) Load Config from sim_config.py
2) Ensure an input workbook exists (auto-generate from 'Model Inputs.xlsx' if configured)
3) Build the model via sim_from_generated.build_simpy_from_generated
4) Run the environment for the configured horizon
5) Print a summary and write optional CSV outputs
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


def _load_config():
    """Load Config from sim_config.py. Supports either a Config class or a pre-instantiated `config` object."""
    try:
        import sim_config as cfg_mod  # type: ignore
    except Exception as e:
        print("ERROR: Could not import sim_config.py in the current directory.")
        print("Make sure sim_config.py exists next to this file.\n")
        raise

    cfg = None
    if hasattr(cfg_mod, "Config"):
        try:
            cfg = cfg_mod.Config()  # type: ignore
        except Exception:
            cfg = None
    if cfg is None and hasattr(cfg_mod, "config"):
        cfg = getattr(cfg_mod, "config")

    if cfg is None:
        raise RuntimeError("sim_config.py must define a Config dataclass or a `config` instance.")
    return cfg


def _ensure_generated_if_needed(cfg) -> Path:
    in_path = Path(cfg.resolve_in_path())
    if in_path.exists():
        return in_path

    if not getattr(cfg, "auto_generate", False):
        raise FileNotFoundError(f"Input workbook not found: {in_path}")

    # Attempt to generate a normalized workbook from source using supply_chain_viz helpers
    try:
        from supply_chain_viz import prepare_inputs_generate  # type: ignore
    except Exception:
        raise RuntimeError(
            "Could not import prepare_inputs_generate from supply_chain_viz.py. "
            "Ensure the file exists and dependencies are installed."
        )

    src = Path(cfg.resolve_source_model())
    out = Path(cfg.resolve_generated_output())
    print(f"Input workbook '{in_path.name}' not found. Auto-generating from '{src.name}' → '{out.name}'...")
    summary = prepare_inputs_generate(src, out)
    print("Generation summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    if not out.exists():
        raise FileNotFoundError(f"Failed to generate workbook at: {out}")
    return out


def _determine_horizon_hours(cfg, meta) -> float:
    # 1) Override via config
    days_override = getattr(cfg, "override_days", None)
    if days_override is not None:
        try:
            return float(days_override) * 24.0
        except Exception:
            pass

    # 2) Use settings object if provided by builder
    settings = meta.get("settings") if isinstance(meta, dict) else None
    if settings is not None:
        # Try best-known patterns
        try:
            if hasattr(settings, "horizon_hours"):
                return float(settings.horizon_hours())  # type: ignore
        except Exception:
            pass
        try:
            days = getattr(settings, "days", None)
            if days is not None:
                return float(days) * 24.0
        except Exception:
            pass

    # 3) Fallback
    return 30.0 * 24.0


def _collect_summary(comps, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    # Store levels
    stores = getattr(comps, "stores", None)
    if stores:
        try:
            summary["stores"] = {str(k): float(getattr(v, "level", 0.0)) for k, v in stores.items()}
        except Exception:
            # Fallback: best-effort repr
            summary["stores"] = {str(k): getattr(v, "level", None) for k, v in stores.items()}

    # Unmet demand
    unmet = getattr(comps, "unmet_demand", None)
    if unmet is not None:
        try:
            if isinstance(unmet, dict):
                summary["unmet_demand"] = {str(k): float(v) for k, v in unmet.items()}
            else:
                summary["unmet_demand"] = float(unmet)
        except Exception:
            summary["unmet_demand"] = unmet

    # Route stats (optional)
    route_stats = getattr(comps, "route_stats", None)
    if route_stats is not None:
        summary["route_stats"] = route_stats

    # Warnings from meta (if provided)
    if isinstance(meta, dict) and "warnings" in meta and meta["warnings"]:
        try:
            # Ensure list of dicts
            if isinstance(meta["warnings"], list):
                summary["warnings"] = list(meta["warnings"])  # shallow copy
        except Exception:
            pass

    return summary


def _write_csvs(cfg, summary: Dict[str, Any]) -> None:
    out_dir = Path(cfg.resolve_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # lazy import
    except Exception:
        pd = None

    # Stores
    if "stores" in summary:
        rows = [(k, v) for k, v in summary["stores"].items()]
        if pd is not None:
            df = pd.DataFrame(rows, columns=["Store", "Level"])
            df.to_csv(out_dir / "store_levels.csv", index=False)
        else:
            with open(out_dir / "store_levels.csv", "w", encoding="utf-8") as f:
                f.write("Store,Level\n")
                for k, v in rows:
                    f.write(f"{k},{v}\n")

    # Unmet demand
    if "unmet_demand" in summary:
        unmet = summary["unmet_demand"]
        if isinstance(unmet, dict):
            rows = [(k, v) for k, v in unmet.items()]
            if pd is not None:
                df = pd.DataFrame(rows, columns=["Key", "Unmet"])
                df.to_csv(out_dir / "unmet_demand.csv", index=False)
            else:
                with open(out_dir / "unmet_demand.csv", "w", encoding="utf-8") as f:
                    f.write("Key,Unmet\n")
                    for k, v in rows:
                        f.write(f"{k},{v}\n")
        else:
            with open(out_dir / "unmet_demand.csv", "w", encoding="utf-8") as f:
                f.write("TotalUnmet\n")
                f.write(f"{unmet}\n")

    # Route stats if present and tabular
    if "route_stats" in summary and isinstance(summary["route_stats"], dict):
        try:
            rs = summary["route_stats"]
            # Expecting dict[name] -> dict of metrics
            keys = sorted({k2 for v in rs.values() if isinstance(v, dict) for k2 in v.keys()})
            with open(out_dir / "route_stats.csv", "w", encoding="utf-8") as f:
                f.write(",".join(["Route"] + keys) + "\n")
                for route, metrics in rs.items():
                    row = [str(route)] + [str(metrics.get(k, "")) for k in keys]
                    f.write(",".join(row) + "\n")
        except Exception:
            pass

    # Warnings (variable columns). Write union of keys across items.
    if "warnings" in summary and isinstance(summary["warnings"], list) and summary["warnings"]:
        rows = summary["warnings"]
        # compute union of keys
        all_keys = []
        key_set = set()
        for item in rows:
            if isinstance(item, dict):
                for k in item.keys():
                    if k not in key_set:
                        key_set.add(k)
                        all_keys.append(k)
        # Prefer type/message first if present
        preferred = [k for k in ["type", "message"] if k in key_set]
        other_keys = [k for k in all_keys if k not in preferred]
        header = preferred + other_keys
        # Write CSV
        out_path = out_dir / "warnings.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for item in rows:
                if isinstance(item, dict):
                    vals = [str(item.get(k, "")) for k in header]
                else:
                    vals = [str(item)]
                f.write(",".join(vals).replace("\n", " ") + "\n")



def main():
    try:
        from sim_from_generated import build_simpy_from_generated  # type: ignore
    except Exception as e:
        print("ERROR: sim_from_generated.py is missing or failed to import.")
        print(e)
        _pause_if_needed(True)
        sys.exit(1)

    # Load config
    try:
        cfg = _load_config()
    except Exception as e:
        print("\nFATAL: Failed to load configuration:")
        print(e)
        _pause_if_needed(True)
        sys.exit(2)

    # Ensure workbook exists (generate if needed)
    try:
        xlsx_path = _ensure_generated_if_needed(cfg)
    except Exception as e:
        print("\nFATAL: Workbook not available:")
        print(e)
        _pause_if_needed(getattr(cfg, "pause_on_finish", True))
        sys.exit(3)

    # Build model
    try:
        comps, meta = build_simpy_from_generated(xlsx_path, product_class=getattr(cfg, "product_class", None))
    except Exception as e:
        print("\nFATAL: Failed to build the model from the workbook:")
        print(e)
        _pause_if_needed(getattr(cfg, "pause_on_finish", True))
        sys.exit(4)

    # Determine horizon
    try:
        horizon_hours = _determine_horizon_hours(cfg, meta)
    except Exception:
        horizon_hours = 30.0 * 24.0

    # Run
    env = getattr(comps, "env", None) or meta.get("env") if isinstance(meta, dict) else None
    if env is None:
        print("\nFATAL: The model did not expose a SimPy environment (`env`).")
        _pause_if_needed(getattr(cfg, "pause_on_finish", True))
        sys.exit(5)

    print(f"Running simulation for {horizon_hours} hours...")
    try:
        env.run(until=horizon_hours)
    except Exception as e:
        print("\nFATAL: Simulation run failed:")
        print(e)
        _pause_if_needed(getattr(cfg, "pause_on_finish", True))
        sys.exit(6)

    # Summarize
    summary = _collect_summary(comps, meta)

    print("\n=== Simulation Summary ===")
    stores = summary.get("stores", {})
    if stores:
        print("Ending store levels (units):")
        for k, v in sorted(stores.items()):
            print(f"  {k}: {v}")

    unmet = summary.get("unmet_demand")
    if unmet is not None:
        if isinstance(unmet, dict):
            total_unmet = sum(float(v) for v in unmet.values())
            print(f"Total unmet demand: {total_unmet}")
        else:
            print(f"Total unmet demand: {unmet}")

    # Warnings
    warns = summary.get("warnings", [])
    if isinstance(warns, list) and warns:
        # Group by type
        counts: dict[str, int] = {}
        for w in warns:
            t = w.get("type") if isinstance(w, dict) else str(type(w))
            t = str(t) if t is not None else "(unknown)"
            counts[t] = counts.get(t, 0) + 1
        print("\nWarnings during build/run:")
        for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {t}: {c}")
        # Show a few samples
        print("  (first 3 examples)")
        for i, w in enumerate(warns[:3]):
            if isinstance(w, dict):
                msg = w.get("message", "")
                ctx = {k: v for k, v in w.items() if k not in {"type", "message"}}
                print(f"    - {w.get('type','')}: {msg} | {ctx}")
            else:
                print(f"    - {w}")

    # Outputs
    if getattr(cfg, "write_csvs", False):
        try:
            _write_csvs(cfg, summary)
            print(f"\nCSV outputs written to: {Path(cfg.resolve_out_dir()).resolve()}")
        except Exception as e:
            print("Warning: Failed to write CSV outputs:", e)

    # Open folder
    if getattr(cfg, "open_folder_after", False):
        try:
            out_dir = Path(cfg.resolve_out_dir()).resolve()
            if os.name == "nt":
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            else:
                # Fallback for non-Windows
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(out_dir)])
        except Exception:
            pass

    _pause_if_needed(getattr(cfg, "pause_on_finish", True))


def _pause_if_needed(pause: bool) -> None:
    if pause:
        try:
            input("\nPress Enter to close...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
