"""
Run the SimPy model built from a generated workbook without using the command line.

Usage:
- Edit settings in sim_config.py
- Double-click this file (Windows), or run `python sim_run.py`

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
from datetime import datetime


def _now_str() -> str:
    try:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return ''


def _log(cfg, msg: str) -> None:
    try:
        if not getattr(cfg, 'verbose_logging', True):
            return
        prefix = f"[{_now_str()}] " if getattr(cfg, 'log_with_timestamps', True) else ''
        print(prefix + str(msg))
    except Exception:
        # Best-effort logging; never fail
        try:
            print(str(msg))
        except Exception:
            pass


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

    # Route stats if present (objects or dicts). Serialize sensibly.
    if "route_stats" in summary and isinstance(summary["route_stats"], dict):
        try:
            from dataclasses import asdict, is_dataclass  # type: ignore
        except Exception:
            def is_dataclass(_):
                return False
            def asdict(x):  # type: ignore
                return x  # best-effort
        try:
            rs = summary["route_stats"]

            def _norm(obj):
                # Convert RouteStats or arbitrary objects to a plain dict of simple fields
                try:
                    if isinstance(obj, dict):
                        d = dict(obj)
                    elif is_dataclass(obj):
                        d = asdict(obj)
                    elif hasattr(obj, "__dict__"):
                        d = {k: getattr(obj, k) for k in vars(obj).keys() if not k.startswith("_")}
                    else:
                        d = {"value": obj}
                except Exception:
                    d = {"value": str(obj)}
                # Derived metrics (if available)
                try:
                    trips = float(d.get("trips_completed", 0) or 0)
                    tons = float(d.get("tons_moved", 0) or 0)
                    d["avg_ton_per_trip"] = (tons / trips) if trips else 0.0
                except Exception:
                    pass
                return d

            normed = {route: _norm(metrics) for route, metrics in rs.items()}
            # Union of all metric keys to build header
            key_set = set()
            for m in normed.values():
                key_set.update(m.keys())
            keys = sorted(key_set)

            with open(out_dir / "route_stats.csv", "w", encoding="utf-8") as f:
                f.write(",".join(["Route"] + keys) + "\n")
                for route, metrics in normed.items():
                    route_str = str(route).replace("→", "->")
                    row = [route_str] + [str(metrics.get(k, "")) for k in keys]
                    f.write(",".join(row) + "\n")
        except Exception:
            # best effort; don't fail the whole run on export problems
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


def _write_action_log(cfg, meta: Dict[str, Any]) -> None:
    """Write a detailed per-hour action log (text and CSV) to sim_outputs.
    Expects meta['action_log'] to be a list of dicts with at least: event, time_h, qty_t, product_class, product, etc.
    """
    try:
        entries = list(meta.get("action_log", [])) if isinstance(meta, dict) else []
    except Exception:
        entries = []
    if not entries:
        return

    out_dir = Path(cfg.resolve_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV (structured)
    csv_cols = [
        "hour", "time_h", "event", "product_class", "location", "equipment", "process",
        "product", "qty_t", "units", "src_location", "src_equipment", "dst_location", "dst_equipment", "duration_h",
    ]
    try:
        import csv as _csv
        with open(out_dir / "sim_log.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=csv_cols)
            w.writeheader()
            for e in entries:
                hour = int(float(e.get("time_h", 0.0)) // 1) + 1
                row = {k: e.get(k, "") for k in csv_cols}
                row.update({"hour": hour})
                w.writerow(row)
    except Exception:
        # best effort
        pass

    # Text (human-friendly)
    def fmt_qty(q):
        try:
            return f"{round(float(q)):,}"
        except Exception:
            return str(q)

    by_hour: Dict[int, list[dict]] = {}
    for e in entries:
        hour = int(float(e.get("time_h", 0.0)) // 1) + 1
        by_hour.setdefault(hour, []).append(e)

    lines: list[str] = []
    for hour in sorted(by_hour.keys()):
        lines.append(f"Hour {hour}:")
        for e in by_hour[hour]:
            ev = str(e.get("event", ""))
            prod = str(e.get("product", "")).upper()
            if ev == "Produced":
                loc = e.get("location", "")
                eq = e.get("equipment", "")
                lines.append(f"  {loc} {eq} Produced {fmt_qty(e.get('qty_t'))} TON {prod}")
            elif ev == "Consumed":
                loc = e.get("location", "")
                eq = e.get("equipment", "")
                lines.append(f"  {loc} {eq} Consumed {fmt_qty(e.get('qty_t'))} TON {prod}")
            elif ev == "Loaded":
                src_loc = e.get("src_location", "")
                src_eq = e.get("src_equipment", "")
                dst_loc = e.get("dst_location", "")
                dst_eq = e.get("dst_equipment", "")
                lines.append(f"  {src_loc} {src_eq} Loaded {fmt_qty(e.get('qty_t'))} TON {prod} to {dst_loc} {dst_eq}")
            elif ev == "Unloaded":
                src_loc = e.get("src_location", "")
                src_eq = e.get("src_equipment", "")
                dst_loc = e.get("dst_location", "")
                dst_eq = e.get("dst_equipment", "")
                lines.append(f"  {dst_loc} {dst_eq} Unloaded {fmt_qty(e.get('qty_t'))} TON {prod} from {src_loc} {src_eq}")
            elif ev == "Transit":
                src_loc = e.get("src_location", "")
                src_eq = e.get("src_equipment", "")
                dst_loc = e.get("dst_location", "")
                dst_eq = e.get("dst_equipment", "")
                dur = e.get("duration_h", "")
                dur_str = f" in {dur} h" if dur not in (None, "") else ""
                lines.append(f"  {src_loc} {src_eq} Transit {fmt_qty(e.get('qty_t'))} TON {prod} to {dst_loc} {dst_eq}{dur_str}")
            elif ev == "Delivered":
                loc = e.get("location", "")
                lines.append(f"  {loc} Delivered {fmt_qty(e.get('qty_t'))} TON {prod}")
            elif ev == "Unmet":
                loc = e.get("location", "")
                lines.append(f"  {loc} Unmet {fmt_qty(e.get('qty_t'))} TON {prod}")
            else:
                lines.append(f"  {ev}: {e}")
        lines.append("")

    try:
        (out_dir / "sim_log.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _write_inventory_snapshots(cfg, meta: Dict[str, Any]) -> None:
    """Write periodic inventory snapshots to sim_outputs/inventory_daily.csv.
    Expects meta['inventory_snapshots'] to be a list of dicts with keys such as:
    day, time_h, product_class, location, equipment, input, store_key, level, capacity, fill_pct
    """
    try:
        entries = list(meta.get("inventory_snapshots", [])) if isinstance(meta, dict) else []
    except Exception:
        entries = []
    if not entries:
        return
    out_dir = Path(cfg.resolve_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "day", "time_h", "product_class", "location", "equipment", "input",
        "store_key", "level", "capacity", "fill_pct",
    ]
    # Try pandas for convenience; else write CSV manually
    try:
        import pandas as pd  # type: ignore
        rows = []
        for e in entries:
            rows.append({k: e.get(k, None) for k in cols})
        df = pd.DataFrame(rows, columns=cols)
        # Keep rows sorted by time then by store_key for readability
        df = df.sort_values(["time_h", "store_key"]).reset_index(drop=True)
        df.to_csv(out_dir / "inventory_daily.csv", index=False)
    except Exception:
        try:
            with open(out_dir / "inventory_daily.csv", "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for e in sorted(entries, key=lambda x: (float(x.get("time_h", 0.0)), str(x.get("store_key", "")))):
                    row = [str(e.get(k, "")) for k in cols]
                    # replace commas/newlines for safety
                    row = [c.replace("\n", " ").replace(",", " ") for c in row]
                    f.write(",".join(row) + "\n")
        except Exception:
            pass


def _assemble_inmemory_frames(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build in-memory pandas DataFrames for Make output and Store inventory.
    Returns a dict with keys: 'make_output_df' (may be None), 'store_inventory_df' (may be None).
    This works even if no CSV outputs are written.
    """
    frames: Dict[str, Any] = {"make_output_df": None, "store_inventory_df": None}
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return frames

    # Action log → Produced events (Make output)
    try:
        log_entries = list(meta.get("action_log", [])) if isinstance(meta, dict) else []
    except Exception:
        log_entries = []
    if log_entries:
        try:
            df_log = pd.DataFrame(log_entries)
            if not df_log.empty and "event" in df_log.columns:
                df_prod = df_log[df_log["event"].astype(str).str.upper() == "PRODUCED"].copy()
                if not df_prod.empty:
                    # Normalize columns and compute hour index
                    if "time_h" in df_prod.columns:
                        df_prod["time_h"] = pd.to_numeric(df_prod["time_h"], errors="coerce").fillna(0.0)
                        df_prod["hour"] = (df_prod["time_h"] // 1).astype(int) + 1
                    # Ensure expected keys exist
                    for col in ["product_class", "location", "equipment", "product", "qty_t"]:
                        if col not in df_prod.columns:
                            df_prod[col] = None
                    df_prod["qty_t"] = pd.to_numeric(df_prod["qty_t"], errors="coerce").fillna(0.0)
                    frames["make_output_df"] = df_prod
        except Exception:
            pass

    # Inventory snapshots → Store inventory balance
    try:
        inv_entries = list(meta.get("inventory_snapshots", [])) if isinstance(meta, dict) else []
    except Exception:
        inv_entries = []
    if inv_entries:
        try:
            df_inv = pd.DataFrame(inv_entries)
            if not df_inv.empty:
                # Normalize
                for c in ["time_h", "level", "capacity", "fill_pct"]:
                    if c in df_inv.columns:
                        df_inv[c] = pd.to_numeric(df_inv[c], errors="coerce")
                if "time_h" in df_inv.columns:
                    df_inv["hour"] = (df_inv["time_h"] // 1).astype(int) + 1
                frames["store_inventory_df"] = df_inv
        except Exception:
            pass

    return frames


def _render_plot_task(task: dict, save_path: str | None = None) -> str | None:
    """Top-level worker to render a single plot (picklable for ProcessPool on Windows)."""
    try:
        import matplotlib
        # Force non-interactive backend if saving
        if save_path is not None:
            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass
        import matplotlib.pyplot as _plt
        kind = task.get("kind")
        title = task.get("title", "")
        if kind == "make":
            x = task.get("x", [])
            cum = task.get("cum", [])
            per = task.get("per", [])
            x_is_hour = bool(task.get("x_is_hour", False))
            fig, ax1 = _plt.subplots(figsize=(9, 4.5))
            ax1.plot(x, cum, color="tab:blue", label="Cumulative Produced (t)")
            ax1.set_title(title)
            ax1.set_xlabel("Hour" if x_is_hour else "Time (h)")
            ax1.set_ylabel("Cumulative Tons Produced", color="tab:blue")
            ax1.tick_params(axis='y', labelcolor="tab:blue")
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            # Secondary series as a LINE (not bars)
            ax2.plot(x, per, color="tab:orange", linestyle="--", linewidth=1.5, label="Per Period Produced (t)")
            ax2.set_ylabel("Per-Period Tons Produced", color="tab:orange")
            ax2.tick_params(axis='y', labelcolor="tab:orange")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")
            fig.tight_layout()
        elif kind == "store":
            x = task.get("x", [])
            y = task.get("y", [])
            fig, ax = _plt.subplots(figsize=(9, 4.5))
            ax.plot(x, y, label="Inventory Level (t)")
            ax.set_title(title)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Inventory (t)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
        else:
            return None
        if save_path:
            _plt.savefig(save_path, dpi=120)
            _plt.close(fig)
            return save_path
        else:
            _plt.show()
            _plt.close(fig)
            return None
    except Exception as e:
        try:
            print("Plot task failed:", e)
        except Exception:
            pass
        return None


def _plot_output_graphs(cfg, frames: Dict[str, Any], horizon_h: float | None = None) -> None:
    """Draw graphs from in-memory DataFrames for:
    - Make: quantity produced over time (per equipment, grouped by product_class and location)
    - Store: inventory level over time (per equipment, grouped by product_class and location)

    Supports two modes controlled by config:
    - Interactive (default False when saving images): show plots inline serially.
    - Parallel save (default True): render PNGs in parallel to speed up when many plots are needed.
    """
    # Try matplotlib import early
    try:
        import matplotlib
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("Plotting skipped: matplotlib not available. Install with `pip install matplotlib`.", e)
        return

    make_df = frames.get("make_output_df")
    inv_df = frames.get("store_inventory_df")

    # Helpers: filename-safe
    import re
    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "")).strip("_") or "na"

    # Build plotting tasks so we can run in parallel without passing DataFrames
    tasks: list[dict] = []

    # Assemble Make tasks
    if make_df is not None:
        try:
            import pandas as _pd  # local alias
            grp_keys = ["product_class", "location", "equipment"]
            for (pc, loc, eq), g in make_df.groupby(grp_keys):
                g = g.copy()
                if "hour" not in g.columns and "time_h" in g.columns:
                    try:
                        g["time_h"] = _pd.to_numeric(g["time_h"], errors="coerce").fillna(0.0)
                        g["hour"] = (g["time_h"] // 1).astype(int) + 1
                    except Exception:
                        pass
                if "hour" in g.columns:
                    per_period = g.groupby("hour", as_index=False)["qty_t"].sum().sort_values("hour")
                    x = per_period["hour"].tolist()
                    x_is_hour = True
                else:
                    g = g.sort_values("time_h")
                    per_period = g.groupby("time_h", as_index=False)["qty_t"].sum()
                    x = per_period["time_h"].tolist()
                    x_is_hour = False
                y_per = per_period["qty_t"].astype(float).tolist()
                # Pad to full horizon if requested
                try:
                    if bool(getattr(cfg, "plot_full_horizon", True)) and horizon_h is not None:
                        H = int(max(1, int(horizon_h)))
                        if x_is_hour:
                            # Build a dense hourly vector 1..H
                            per_map = {int(xx): float(vv) for xx, vv in zip(x, y_per)}
                            x = list(range(1, H + 1))
                            y_per = [per_map.get(hh, 0.0) for hh in x]
                        else:
                            # time_h series: ensure final point at horizon with zero per-period increment
                            if len(x) == 0 or float(x[-1]) < float(horizon_h):
                                x = x + [float(horizon_h)]
                                y_per = y_per + [0.0]
                except Exception:
                    pass
                # cumulative
                cum_vals = []
                total = 0.0
                for v in y_per:
                    total += float(v)
                    cum_vals.append(total)
                # If hourly and padded out, cum_vals already extend; if time_h padded with final point, cum_vals also extended
                title = f"Make Output — {pc or '(all)'} @ {loc or '(loc)'} — {eq or '(equip)'}"
                tasks.append({
                    "kind": "make",
                    "x": x,
                    "cum": cum_vals,
                    "per": y_per,
                    "x_is_hour": x_is_hour,
                    "title": title,
                    "pc": str(pc or ""),
                    "loc": str(loc or ""),
                    "eq": str(eq or ""),
                })
        except Exception as e:
            print("Warning: failed to prepare Make plots:", e)

    # Assemble Store tasks
    if inv_df is not None:
        try:
            grp_keys = ["product_class", "location", "equipment"]
            tmp = inv_df.copy()
            for k in grp_keys:
                if k not in tmp.columns:
                    tmp[k] = None
            for (pc, loc, eq), g in tmp.groupby(grp_keys):
                if "level" not in g.columns:
                    continue
                g = g.sort_values("time_h")
                x = g["time_h"].astype(float).tolist()
                y = g["level"].astype(float).tolist()
                # Pad to full horizon if requested: carry forward last known level to horizon_h
                try:
                    if bool(getattr(cfg, "plot_full_horizon", True)) and horizon_h is not None:
                        if len(x) == 0:
                            # No points; add flat line at 0 from 0 to horizon
                            x = [0.0, float(horizon_h)]
                            y = [0.0, 0.0]
                        else:
                            last_x = float(x[-1])
                            last_y = float(y[-1]) if len(y)>0 else 0.0
                            if last_x < float(horizon_h):
                                x = x + [float(horizon_h)]
                                y = y + [last_y]
                except Exception:
                    pass
                title = f"Store Inventory — {pc or '(all)'} @ {loc or '(loc)'} — {eq or '(equip)'}"
                tasks.append({
                    "kind": "store",
                    "x": x,
                    "y": y,
                    "title": title,
                    "pc": str(pc or ""),
                    "loc": str(loc or ""),
                    "eq": str(eq or ""),
                })
        except Exception as e:
            print("Warning: failed to prepare Store plots:", e)

    if not tasks:
        return

    out_dir = Path(getattr(cfg, "resolve_out_dir")()).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Decide execution mode
    use_parallel = bool(getattr(cfg, "plot_parallel", True))
    save_images = bool(getattr(cfg, "plot_save_images", True))

    if save_images:
        # Build save paths
        for t in tasks:
            safe_name = f"{_slug(t.get('pc',''))}__{_slug(t.get('loc',''))}__{_slug(t.get('eq',''))}"
            sub = "make" if t.get("kind") == "make" else "store"
            t["save_path"] = str(plots_dir / f"{sub}__{safe_name}.png")

    # Execute
    completed = 0
    saved = 0
    if use_parallel and save_images:
        # Prefer processes for CPU-bound render; fall back to threads
        import concurrent.futures as _fut
        workers = getattr(cfg, "plot_workers", None)
        if not workers or int(workers) <= 0:
            import os as _os
            workers = min(32, max(1, (_os.cpu_count() or 2)))
        try:
            with _fut.ProcessPoolExecutor(max_workers=int(workers)) as ex:
                futs = [ex.submit(_render_plot_task, t, t.get("save_path")) for t in tasks]
                for f in _fut.as_completed(futs):
                    completed += 1
                    res = f.result()
                    if res:
                        saved += 1
        except Exception as e:
            print("ProcessPool failed (falling back to threads):", e)
            try:
                with _fut.ThreadPoolExecutor(max_workers=int(workers)) as ex:
                    futs = [ex.submit(_render_plot_task, t, t.get("save_path")) for t in tasks]
                    for f in _fut.as_completed(futs):
                        completed += 1
                        res = f.result()
                        if res:
                            saved += 1
            except Exception as e2:
                print("ThreadPool also failed; rendering serially:", e2)
                for t in tasks:
                    res = _render_plot_task(t, t.get("save_path"))
                    completed += 1
                    if res:
                        saved += 1
    else:
        # Serial interactive or serial save
        for t in tasks:
            res = _render_plot_task(t, t.get("save_path") if save_images else None)
            completed += 1
            if res:
                saved += 1

    if save_images:
        print(f"Saved {saved}/{completed} plot images to: {plots_dir}")
    else:
        print(f"Rendered {completed} plots interactively.")


def _write_frozen_model(cfg, meta: Dict[str, Any]) -> None:
    """Write a standalone Python file that contains the exact SimPy model that was built
    from the Excel inputs and executed. This file can be run independently without Excel.
    Output path: sim_outputs/frozen_sim_model.py
    """
    export = None
    try:
        export = meta.get("export") if isinstance(meta, dict) else None
    except Exception:
        export = None
    if not export:
        return

    out_dir = Path(cfg.resolve_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frozen_sim_model.py"

    import json
    settings = export.get("settings", {})
    stores = export.get("stores", [])
    makes = export.get("makes", [])
    moves = export.get("moves", [])
    deliveries = export.get("deliveries", [])

    # Build a minimal, readable Python script.
    script = []
    script.append("# Auto-generated by sim_run.py — standalone SimPy model frozen from Excel inputs\n")
    script.append("from __future__ import annotations\n")
    script.append("import math\n")
    script.append("import random\n")
    script.append("from typing import Dict\n")
    script.append("import simpy\n\n")

    script.append(f"SETTINGS = {json.dumps(settings, indent=2)}\n\n")
    script.append(f"STORES = {json.dumps(stores, indent=2)}\n\n")
    script.append(f"MAKES = {json.dumps(makes, indent=2)}\n\n")
    script.append(f"MOVES = {json.dumps(moves, indent=2)}\n\n")
    script.append(f"DELIVERIES = {json.dumps(deliveries, indent=2)}\n\n")

    script.append("def build():\n")
    script.append("    env = simpy.Environment()\n")
    script.append("    stores: Dict[str, simpy.Container] = {}\n")
    script.append("    unmet: Dict[str, float] = {}\n")
    script.append("    action_log: list[dict] = []\n\n")
    script.append("    def log_action(event: str, t: float, details: dict):\n")
    script.append("        try:\n")
    script.append("            entry = {\"event\": event, \"time_h\": float(t)}\n")
    script.append("            entry.update(details)\n")
    script.append("            action_log.append(entry)\n")
    script.append("        except Exception:\n")
    script.append("            pass\n\n")

    # Stores
    script.append("    # Stores\n")
    script.append("    # Optionally seed RNG for reproducible random openings\n")
    script.append("    try:\n")
    script.append("        seed = SETTINGS.get('random_seed', None)\n")
    script.append("        if seed is not None:\n")
    script.append("            random.seed(int(seed))\n")
    script.append("    except Exception:\n")
    script.append("        pass\n")
    script.append("    rand_open = bool(SETTINGS.get('random_opening', True))\n")
    script.append("    for s in STORES:\n")
    script.append("        cap = float(s.get('capacity', 0.0))\n")
    script.append("        lo = s.get('opening_low', None)\n")
    script.append("        hi = s.get('opening_high', None)\n")
    script.append("        key = s['store_key']\n")
    script.append("        opening = float(s.get('opening', 0.0))\n")
    script.append("        try:\n")
    script.append("            if rand_open and (lo is not None or hi is not None):\n")
    script.append("                lo_v = float(0.0 if lo is None else lo)\n")
    script.append("                hi_v = float(0.0 if hi is None else hi)\n")
    script.append("                if hi_v < lo_v:\n")
    script.append("                    lo_v, hi_v = hi_v, lo_v\n")
    script.append("                opening = float(random.uniform(lo_v, hi_v))\n")
    script.append("        except Exception:\n")
    script.append("            # keep provided opening\n")
    script.append("            pass\n")
    script.append("        cap = max(0.0, float(cap))\n")
    script.append("        opening = max(0.0, float(opening))\n")
    script.append("        if cap <= 0.0 and opening > 0.0:\n")
    script.append("            cap = opening\n")
    script.append("        if opening > cap:\n")
    script.append("            opening = cap\n")
    script.append("        stores[key] = simpy.Container(env, init=opening, capacity=cap)\n\n")

    # Makes
    script.append("    # Make producer processes\n")
    script.append("    def _choose_candidate(rule: str, cands: list[dict]):\n")
    script.append("        best_i = None\n")
    script.append("        best_metric = None\n")
    script.append("        for i, c in enumerate(cands):\n")
    script.append("            skey = c['out_store_key']\n")
    script.append("            cont = stores.get(skey)\n")
    script.append("            if cont is None:\n")
    script.append("                continue\n")
    script.append("            level = float(cont.level)\n")
    script.append("            cap = float(cont.capacity) if getattr(cont, 'capacity', None) is not None else None\n")
    script.append("            if rule == 'min_level' or cap in (None, 0.0):\n")
    script.append("                metric = level\n")
    script.append("            else:\n")
    script.append("                try:\n")
    script.append("                    metric = level / max(cap, 1e-9)\n")
    script.append("                except Exception:\n")
    script.append("                    metric = 1.0\n")
    script.append("            if best_i is None or metric < best_metric:\n")
    script.append("                best_i = i\n")
    script.append("                best_metric = metric\n")
    script.append("        return 0 if best_i is None else best_i\n\n")

    script.append("    def _producer_multi(env, candidates, in_key, rate_tph, cons_pct, step_h, rule):\n")
    script.append("        while True:\n")
    script.append("            idx = _choose_candidate(rule, candidates)\n")
    script.append("            tgt = candidates[idx]\n")
    script.append("            out_key = tgt['out_store_key']\n")
    script.append("            qty_out = max(0.0, float(rate_tph)) * float(step_h)\n")
    script.append("            dst = stores.get(out_key)\n")
    script.append("            if dst is not None and qty_out > 0:\n")
    script.append("                room = dst.capacity - dst.level if dst.capacity is not None else qty_out\n")
    script.append("                qty_out = max(0.0, min(qty_out, room))\n")
    script.append("            else:\n")
    script.append("                qty_out = 0.0\n")
    script.append("            qty_in = qty_out * float(cons_pct) if in_key else 0.0\n")
    script.append("            if in_key and qty_out > 0:\n")
    script.append("                src = stores.get(in_key)\n")
    script.append("                if src is not None and qty_in > 0:\n")
    script.append("                    take = min(src.level, qty_in)\n")
    script.append("                    if take > 0:\n")
    script.append("                        yield src.get(take)\n")
    script.append("                    if qty_in > 0 and take < qty_in:\n")
    script.append("                        scale = take/qty_in if qty_in > 0 else 0.0\n")
    script.append("                        qty_out *= scale\n")
    script.append("            if dst is not None and qty_out > 0:\n")
    script.append("                yield dst.put(qty_out)\n")
    script.append("            yield env.timeout(float(step_h))\n\n")

    script.append("    def _producer_single(env, out_key, in_key, rate_tph, cons_pct, step_h):\n")
    script.append("        while True:\n")
    script.append("            qty_out = max(0.0, float(rate_tph)) * float(step_h)\n")
    script.append("            qty_in = qty_out * float(cons_pct) if in_key else 0.0\n")
    script.append("            if in_key:\n")
    script.append("                src = stores.get(in_key)\n")
    script.append("                if src is not None and qty_in > 0:\n")
    script.append("                    take = min(src.level, qty_in)\n")
    script.append("                    if take > 0:\n")
    script.append("                        yield src.get(take)\n")
    script.append("                    if qty_in > 0 and take < qty_in:\n")
    script.append("                        scale = take/qty_in if qty_in > 0 else 0.0\n")
    script.append("                        qty_out *= scale\n")
    script.append("            dst = stores.get(out_key)\n")
    script.append("            if dst is not None and qty_out > 0:\n")
    script.append("                room = dst.capacity - dst.level if dst.capacity is not None else qty_out\n")
    script.append("                put_amt = min(qty_out, max(room, 0))\n")
    script.append("                if put_amt > 0:\n")
    script.append("                    yield dst.put(put_amt)\n")
    script.append("            yield env.timeout(float(step_h))\n\n")

    script.append("    for m in MAKES:\n")
    script.append("        if 'candidates' in m and m['candidates']:\n")
    script.append("            rule = m.get('choice_rule', SETTINGS.get('make_output_choice', 'min_fill_pct'))\n")
    script.append("            env.process(_producer_multi(env, m['candidates'], m.get('in_store_key'), m['mean_rate_tph'], m['consumption_pct'], m['step_hours'], rule))\n")
    script.append("        else:\n")
    script.append("            env.process(_producer_single(env, m.get('out_store_key'), m.get('in_store_key'), m['mean_rate_tph'], m['consumption_pct'], m['step_hours']))\n\n")

    # Moves
    script.append("    # Transporter (move) processes\n")
    script.append("    def _transporter(env, origin_key, dest_key, payload_t, load_rate_tph, unload_rate_tph, to_min, back_min, step_h):\n")
    script.append("        def _parse_store_key(k: str):\n")
    script.append("            try:\n")
    script.append("                pc_v, loc_v, eq_v, inp_v = str(k).split('|')\n")
    script.append("                return pc_v, loc_v, eq_v, inp_v\n")
    script.append("            except Exception:\n")
    script.append("                return '', '', '', ''\n")
    script.append("        pc_o, loc_o, eq_o, inp = _parse_store_key(origin_key)\n")
    script.append("        pc_d, loc_d, eq_d, _ = _parse_store_key(dest_key)\n")
    script.append("        is_train = ('TRAIN' in str(eq_o).upper()) or ('TRAIN' in str(eq_d).upper())\n")
    script.append("        while True:\n")
    script.append("            if load_rate_tph <= 0 or payload_t <= 0:\n")
    script.append("                yield env.timeout((float(to_min)+float(back_min))/60.0 if (to_min or back_min) else float(step_h))\n")
    script.append("                continue\n")
    script.append("            origin = stores.get(origin_key)\n")
    script.append("            dest = stores.get(dest_key)\n")
    script.append("            if origin is None or dest is None:\n")
    script.append("                yield env.timeout(float(step_h))\n")
    script.append("                continue\n")
    script.append("            # For TRAIN operations, require full payload available at origin AND full room at destination before loading\n")
    script.append("            if is_train:\n")
    script.append("                have_stock = float(origin.level) >= float(payload_t)\n")
    script.append("                have_room = (dest.capacity - dest.level) >= float(payload_t) if dest.capacity is not None else True\n")
    script.append("                if not (have_stock and have_room):\n")
    script.append("                    yield env.timeout(float(step_h))\n")
    script.append("                    continue\n")
    script.append("            load_time_h = float(payload_t) / max(float(load_rate_tph), 1e-9)\n")
    script.append("            if is_train:\n")
    script.append("                take = float(payload_t)\n")
    script.append("            else:\n")
    script.append("                take = min(origin.level, float(payload_t))\n")
    script.append("            if take > 0:\n")
    script.append("                yield env.timeout(load_time_h)\n")
    script.append("                yield origin.get(take)\n")
    script.append("                # Log load\n")
    script.append("                try:\n")
    script.append("                    log_action(\n")
    script.append("                        'Loaded',\n")
    script.append("                        env.now,\n")
    script.append("                        {\n")
    script.append("                            'product_class': pc_o,\n")
    script.append("                            'product': inp,\n")
    script.append("                            'qty_t': float(take),\n")
    script.append("                            'units': 'TON',\n")
    script.append("                            'src_location': loc_o,\n")
    script.append("                            'src_equipment': eq_o,\n")
    script.append("                            'dst_location': loc_d,\n")
    script.append("                            'dst_equipment': eq_d,\n")
    script.append("                        },\n")
    script.append("                    )\n")
    script.append("                except Exception:\n")
    script.append("                    pass\n")
    script.append("            else:\n")
    script.append("                yield env.timeout(float(step_h))\n")
    script.append("                continue\n")
    script.append("            # Transit to destination\n")
    script.append("            try:\n")
    script.append("                log_action(\n")
    script.append("                    'Transit',\n")
    script.append("                    env.now,\n")
    script.append("                    {\n")
    script.append("                        'product_class': pc_o,\n")
    script.append("                        'product': inp,\n")
    script.append("                        'qty_t': float(take),\n")
    script.append("                        'units': 'TON',\n")
    script.append("                        'src_location': loc_o,\n")
    script.append("                        'src_equipment': eq_o,\n")
    script.append("                        'dst_location': loc_d,\n")
    script.append("                        'dst_equipment': eq_d,\n")
    script.append("                        'duration_h': float(to_min)/60.0,\n")
    script.append("                    },\n")
    script.append("                )\n")
    script.append("            except Exception:\n")
    script.append("                pass\n")
    script.append("            yield env.timeout(float(to_min)/60.0)\n")
    script.append("            unload_time_h = float(take) / max(float(unload_rate_tph), 1e-9) if unload_rate_tph else 0.0\n")
    script.append("            if is_train:\n")
    script.append("                put_amt = float(take)\n")
    script.append("            else:\n")
    script.append("                room = dest.capacity - dest.level if dest.capacity is not None else take\n")
    script.append("                put_amt = min(take, max(room, 0))\n")
    script.append("            if put_amt > 0 and unload_rate_tph > 0:\n")
    script.append("                yield env.timeout(unload_time_h)\n")
    script.append("                yield dest.put(put_amt)\n")
    script.append("                # Log unload\n")
    script.append("                try:\n")
    script.append("                    log_action(\n")
    script.append("                        'Unloaded',\n")
    script.append("                        env.now,\n")
    script.append("                        {\n")
    script.append("                            'product_class': pc_o,\n")
    script.append("                            'product': inp,\n")
    script.append("                            'qty_t': float(put_amt),\n")
    script.append("                            'units': 'TON',\n")
    script.append("                            'src_location': loc_o,\n")
    script.append("                            'src_equipment': eq_o,\n")
    script.append("                            'dst_location': loc_d,\n")
    script.append("                            'dst_equipment': eq_d,\n")
    script.append("                        },\n")
    script.append("                    )\n")
    script.append("                except Exception:\n")
    script.append("                    pass\n")
    script.append("            yield env.timeout(float(back_min)/60.0)\n\n")

    script.append("    for mv in MOVES:\n")
    script.append("        n = int(max(1, mv.get('n_units', 1)))\n")
    script.append("        for _ in range(n):\n")
    script.append("            env.process(_transporter(env, mv['origin_key'], mv['dest_key'], mv['payload_t'], mv['load_rate_tph'], mv['unload_rate_tph'], mv['to_min'], mv['back_min'], mv['step_hours']))\n\n")

    # Deliveries
    script.append("    # Consumers (deliveries)\n")
    script.append("    def _consumer(env, key, rate, step_h):\n")
    script.append("        cont = stores.get(key)\n")
    script.append("        while True:\n")
    script.append("            if cont is None:\n")
    script.append("                yield env.timeout(float(step_h))\n")
    script.append("                continue\n")
    script.append("            take = min(cont.level, float(rate))\n")
    script.append("            if take > 0:\n")
    script.append("                yield cont.get(take)\n")
    script.append("            short = float(rate) - float(take)\n")
    script.append("            if short > 0:\n")
    script.append("                unmet[key] = unmet.get(key, 0.0) + short\n")
    script.append("            yield env.timeout(float(step_h))\n\n")

    script.append("    for d in DELIVERIES:\n")
    script.append("        env.process(_consumer(env, d['store_key'], d['rate_per_step'], d['step_hours']))\n\n")

    script.append("    return env, stores, unmet\n\n")

    # Main runner in frozen script
    script.append("def main(hours: float | None = None):\n")
    script.append("    env, stores, unmet = build()\n")
    script.append("    horizon_h = hours if hours is not None else float(SETTINGS.get('horizon_days', 30))*24.0\n")
    script.append("    env.run(until=horizon_h)\n")
    script.append("    print('=== Frozen Simulation Summary ===')\n")
    script.append("    print('Ending store levels:')\n")
    script.append("    for k in sorted(stores.keys()):\n")
    script.append("        print(f'  {k}: {stores[k].level}')\n")
    script.append("    if unmet:\n")
    script.append("        total_unmet = sum(float(v) for v in unmet.values())\n")
    script.append("        print(f'Total unmet demand: {total_unmet}')\n")
    script.append("\n")
    script.append("if __name__ == '__main__':\n")
    script.append("    main()\n")

    out_path.write_text("".join(script), encoding="utf-8")



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
        _log(cfg, "Loaded configuration from sim_config.py")
    except Exception as e:
        print("\nFATAL: Failed to load configuration:")
        print(e)
        _pause_if_needed(True)
        sys.exit(2)

    # Ensure workbook exists (generate if needed)
    try:
        in_name = str(getattr(cfg, 'in_xlsx', 'generated_model_inputs.xlsx'))
        _log(cfg, f"Checking input workbook: {in_name}")
        xlsx_path = _ensure_generated_if_needed(cfg)
        _log(cfg, f"Reading workbook: {Path(xlsx_path).name}")
    except Exception as e:
        print("\nFATAL: Workbook not available:")
        print(e)
        _pause_if_needed(getattr(cfg, "pause_on_finish", True))
        sys.exit(3)

    # Build model
    try:
        _log(cfg, "Compiling SimPy model from workbook (building components)...")
        comps, meta = build_simpy_from_generated(xlsx_path, product_class=getattr(cfg, "product_class", None))
        # Try to describe components
        try:
            stores = getattr(comps, 'stores', None)
            makes = getattr(comps, 'makes', None)
            moves = getattr(comps, 'moves', None)
            deliveries = getattr(comps, 'deliveries', None)
            def _count(x):
                try:
                    if x is None:
                        return 0
                    if hasattr(x, 'items'):
                        return len(list(x.items()))
                    return len(list(x))
                except Exception:
                    return 0
            _log(cfg, f"Added stores: {_count(stores)}, producers: {_count(makes)}, movers: {_count(moves)}, deliveries: {_count(deliveries)}")
        except Exception:
            pass
        _log(cfg, "Model compilation complete.")
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

    _log(cfg, f"Starting simulation for {horizon_hours} hours...")
    try:
        # Progress loop
        step_pct = max(1, int(getattr(cfg, 'progress_step_pct', 10) or 10))
        step_pct = min(step_pct, 50)
        total = float(horizon_hours)
        # Build unique checkpoints in hours
        checkpoints = []
        for p in range(step_pct, 100 + step_pct, step_pct):
            pct = min(p, 100)
            t = total * (pct / 100.0)
            checkpoints.append((pct, t))
        seen_pct = set()
        last_target = 0.0
        for pct, t in checkpoints:
            if pct in seen_pct:
                continue
            seen_pct.add(pct)
            target = max(last_target, float(t))
            if target <= last_target:
                continue
            env.run(until=target)
            _log(cfg, f"Simulation {pct}% complete")
            last_target = target
        if last_target < total:
            env.run(until=total)
            _log(cfg, "Simulation 100% complete")
        else:
            if 100 not in seen_pct:
                _log(cfg, "Simulation 100% complete")
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
            _log(cfg, "Writing CSV outputs...")
            _write_csvs(cfg, summary)
            print(f"\nCSV outputs written to: {Path(cfg.resolve_out_dir()).resolve()}")
        except Exception as e:
            print("Warning: Failed to write CSV outputs:", e)

    # Detailed per-hour log
    if getattr(cfg, "write_log", True):
        try:
            _log(cfg, "Writing detailed simulation log (text + CSV)...")
            _write_action_log(cfg, meta)
            print(f"Detailed simulation log written to: {Path(cfg.resolve_out_dir()).resolve() / 'sim_log.txt'}")
        except Exception as e:
            print("Warning: Failed to write detailed log:", e)

    # Frozen model source code (standalone script)
    if getattr(cfg, "write_model_source", True):
        try:
            _log(cfg, "Writing frozen model source (standalone) ...")
            _write_frozen_model(cfg, meta)
            out_dir_path = Path(cfg.resolve_out_dir()).resolve()
            print(f"Frozen model source written to: {out_dir_path / 'frozen_sim_model.py'}")
        except Exception as e:
            print("Warning: Failed to write frozen model source:", e)

    # Daily inventory snapshots CSV
    if getattr(cfg, "write_daily_snapshots", True):
        try:
            _log(cfg, "Writing inventory snapshots CSV...")
            _write_inventory_snapshots(cfg, meta)
            print(f"Inventory snapshots written to: {Path(cfg.resolve_out_dir()).resolve() / 'inventory_daily.csv'}")
        except Exception as e:
            print("Warning: Failed to write inventory snapshots:", e)

    # In-memory analysis and plots (independent of file outputs)
    try:
        _log(cfg, "Assembling in-memory data frames for analysis/graphs...")
        frames = _assemble_inmemory_frames(meta)
        # attach to meta for programmatic access after run
        try:
            if isinstance(meta, dict):
                meta["inmemory_frames"] = frames
        except Exception:
            pass
        if getattr(cfg, "plot_output_graphs", True):
            _log(cfg, "Generating output graphs...")
            _plot_output_graphs(cfg, frames, horizon_hours)
    except Exception as e:
        print("Warning: failed to build/plot in-memory outputs:", e)

    # Open folder
    if getattr(cfg, "open_folder_after", False):
        try:
            out_dir = Path(cfg.resolve_out_dir()).resolve()
            _log(cfg, f"Opening output folder: {out_dir}")
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
