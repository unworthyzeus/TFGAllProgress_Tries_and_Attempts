"""Interactive viewer for histogram CSV files.

Modes:
    - per-sample
    - aggregate (filtered)
    - same-topology multi-height groups

Filters:
    city, city_type_3, city_type_6, expert_mode, metric, expert, altitude range,
    and "only samples that belong to multi-height topology groups".
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ROOT = Path(r"c:/TFG/TFGPractice")
DEFAULT_CFG = (
    ROOT
    / "TFGSeventyFifthTry75"
    / "experiments"
    / "seventyfifth_try75_experts"
    / "try75_expert_allcity_los.yaml"
)
DEFAULT_KIND_ORDER = [
    "target_los",
    "target_nlos",
    "pred_los",
    "pred_nlos",
    "target_delay_spread",
    "pred_delay_spread",
    "target_angular_spread",
    "pred_angular_spread",
]
KIND_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#7f7f7f",
    "#4daf4a",
    "#984ea3",
]
PANEL_BORDER_PALETTE = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#bcbd22",
    "#8c564b",
]
GROUP_TEMPLATES = {
    1: [
        [
            "target_los",
            "target_nlos",
            "pred_los",
            "pred_nlos",
            "target_delay_spread",
            "pred_delay_spread",
            "target_angular_spread",
            "pred_angular_spread",
        ]
    ],
    2: [
        ["target_los", "target_nlos", "pred_los", "pred_nlos"],
        ["target_delay_spread", "pred_delay_spread", "target_angular_spread", "pred_angular_spread"],
    ],
    3: [
        ["target_los", "pred_los"],
        ["target_nlos", "pred_nlos"],
        ["target_delay_spread", "pred_delay_spread", "target_angular_spread", "pred_angular_spread"],
    ],
    4: [
        ["target_los", "pred_los"],
        ["target_nlos", "pred_nlos"],
        ["target_delay_spread", "pred_delay_spread"],
        ["target_angular_spread", "pred_angular_spread"],
    ],
    6: [
        ["target_los", "pred_los"],
        ["target_nlos", "pred_nlos"],
        ["target_delay_spread"],
        ["pred_delay_spread"],
        ["target_angular_spread"],
        ["pred_angular_spread"],
    ],
    8: [[kind] for kind in DEFAULT_KIND_ORDER],
}


def load_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    bin_cols = [c for c in df.columns if c.startswith("b") and c[1:].lstrip("-").isdigit()]
    bin_cols.sort(key=lambda c: int(c[1:]))
    centers = np.array([int(c[1:]) + 0.5 for c in bin_cols], dtype=np.float64)
    return df, centers, bin_cols


def pivot_sample(
    df: pd.DataFrame,
    bin_cols: list[str],
    city: str,
    sample: str,
    kinds: list[str],
) -> dict[str, np.ndarray]:
    sub = df[(df["city"] == city) & (df["sample"] == sample)]
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        rows = sub[sub["kind"] == kind]
        if rows.empty:
            out[kind] = np.zeros(len(bin_cols), dtype=np.int64)
        else:
            out[kind] = rows[bin_cols].to_numpy(dtype=np.int64).sum(axis=0)
    return out


def aggregate(df: pd.DataFrame, bin_cols: list[str], kinds: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        sub = df[df["kind"] == kind]
        if sub.empty:
            out[kind] = np.zeros(len(bin_cols), dtype=np.int64)
        else:
            out[kind] = sub[bin_cols].to_numpy(dtype=np.int64).sum(axis=0)
    return out


def resolve_hdf5_path(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    hdf5_rel = str(cfg.get("data", {}).get("hdf5_path", "")).strip()
    if not hdf5_rel:
        raise RuntimeError(f"Could not find data.hdf5_path in {config_path}")

    candidate = (config_path.parent / hdf5_rel).resolve()
    if candidate.exists():
        return candidate

    fallback = ROOT / "datasets" / Path(hdf5_rel).name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Unable to resolve HDF5 path from config. Tried: "
        f"{candidate} and {fallback}"
    )


def extract_scalar(grp: h5py.Group, name: str, default: float = float("nan")) -> float:
    if name not in grp:
        return default
    return float(np.asarray(grp[name][()]).reshape(()))


def build_topology_groups(
    sample_keys: list[str],
    hdf5_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sample_to_group: dict[str, str] = {}

    with h5py.File(hdf5_path, "r") as h5:
        for key in sample_keys:
            city, sample = key.split("/", 1)
            if city not in h5 or sample not in h5[city]:
                continue
            grp = h5[city][sample]
            if "topology_map" not in grp:
                continue
            topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
            topo_hash = hashlib.sha1(topo.tobytes()).hexdigest()
            altitude_m = extract_scalar(grp, "uav_height")
            groups.setdefault((city, topo_hash), []).append(
                {
                    "city": city,
                    "sample": sample,
                    "sample_key": key,
                    "altitude_m": altitude_m,
                }
            )

    out_groups: dict[str, dict[str, Any]] = {}
    for (city, topo_hash), items in groups.items():
        items.sort(key=lambda item: (item["altitude_m"], item["sample"]))
        heights = {round(float(item["altitude_m"]), 6) for item in items}
        if len(items) < 2 or len(heights) < 2:
            continue
        group_id = f"{city}/{topo_hash}"
        out_groups[group_id] = {
            "group_id": group_id,
            "city": city,
            "topology_hash": topo_hash,
            "items": items,
            "altitude_span_m": float(items[-1]["altitude_m"] - items[0]["altitude_m"]),
        }
        for item in items:
            sample_to_group[item["sample_key"]] = group_id

    return out_groups, sample_to_group


class Viewer:
    def __init__(self, root: tk.Tk, csv_path: Path, config_path: Path | None, hdf5_path: Path | None) -> None:
        self.root = root
        self.csv_path = csv_path
        self.df, self.centers, self.bin_cols = load_csv(csv_path)
        if "city_type_3" not in self.df.columns:
            self.df["city_type_3"] = self.df.get("city_type", "")
        if "city_type_6" not in self.df.columns:
            self.df["city_type_6"] = ""
        if "expert_mode" not in self.df.columns:
            self.df["expert_mode"] = ""
        if "metric" not in self.df.columns:
            self.df["metric"] = "path_loss"
        # The CSV is very wide, so defragment before adding derived columns.
        self.df = self.df.copy()
        self.df["altitude_m"] = pd.to_numeric(self.df["altitude_m"], errors="coerce")
        sample_key = self.df["city"].astype(str).str.cat(self.df["sample"].astype(str), sep="/")
        self.df = pd.concat([self.df, sample_key.rename("sample_key")], axis=1)

        kinds_in_csv = sorted([k for k in self.df["kind"].dropna().astype(str).unique().tolist() if k])
        self.kinds = [k for k in DEFAULT_KIND_ORDER if k in kinds_in_csv] + [k for k in kinds_in_csv if k not in DEFAULT_KIND_ORDER]
        self.kind_colors = {k: KIND_PALETTE[i % len(KIND_PALETTE)] for i, k in enumerate(self.kinds)}
        self._entries: list[dict[str, Any]] = []

        self.topology_groups: dict[str, dict[str, Any]] = {}
        self.sample_to_group: dict[str, str] = {}
        self.topology_status = "Multi-height topology groups unavailable"
        self._load_topology_groups(config_path=config_path, hdf5_path=hdf5_path)

        self._build_ui()
        self._apply_filters()

    def _load_topology_groups(self, config_path: Path | None, hdf5_path: Path | None) -> None:
        try:
            resolved_hdf5 = hdf5_path if hdf5_path is not None else resolve_hdf5_path(config_path or DEFAULT_CFG)
            sample_keys = sorted(self.df["sample_key"].dropna().astype(str).unique().tolist())
            self.topology_groups, self.sample_to_group = build_topology_groups(sample_keys, resolved_hdf5)
            self.topology_status = f"Loaded {len(self.topology_groups)} multi-height topology groups"
        except Exception as exc:
            self.topology_groups = {}
            self.sample_to_group = {}
            self.topology_status = f"Multi-height topology groups unavailable: {exc}"

    def _build_ui(self) -> None:
        self.root.title(f"Histograms viewer - {self.csv_path.name}")
        self.root.geometry("1500x940")
        self.root.minsize(1300, 820)

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        filters = ttk.LabelFrame(top, text="Filters")
        filters.pack(fill=tk.X, pady=(0, 6))

        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(3, weight=1)
        filters.columnconfigure(5, weight=1)
        filters.columnconfigure(7, weight=1)
        filters.columnconfigure(9, weight=1)
        filters.columnconfigure(11, weight=1)

        cities = ["(all)"] + sorted(self.df["city"].dropna().astype(str).unique().tolist())
        ctypes3 = ["(all)"] + sorted([c for c in self.df["city_type_3"].dropna().astype(str).unique().tolist() if c])
        ctypes6 = ["(all)"] + sorted([c for c in self.df["city_type_6"].dropna().astype(str).unique().tolist() if c])
        modes = ["(all)"] + sorted([m for m in self.df["expert_mode"].dropna().astype(str).unique().tolist() if m])
        metrics = ["(all)"] + sorted([m for m in self.df["metric"].dropna().astype(str).unique().tolist() if m])

        self._expert_col = "expert_id" if "expert_id" in self.df.columns else ("expert" if "expert" in self.df.columns else None)
        experts_vals = ["(all)"]
        if self._expert_col:
            experts_vals += sorted(
                [
                    e
                    for e in self.df[self._expert_col].dropna().astype(str).unique().tolist()
                    if e and e != "ground_truth"
                ]
            )

        alt = self.df["altitude_m"].dropna()
        lo_alt = float(np.floor(alt.min())) if not alt.empty else 0.0
        hi_alt = float(np.ceil(alt.max())) if not alt.empty else 150.0

        self.var_city = tk.StringVar(value="(all)")
        self.var_ctype3 = tk.StringVar(value="(all)")
        self.var_ctype6 = tk.StringVar(value="(all)")
        self.var_mode_filter = tk.StringVar(value="(all)")
        self.var_metric = tk.StringVar(value="(all)")
        self.var_expert = tk.StringVar(value="(all)")
        self.var_alt_lo = tk.DoubleVar(value=lo_alt)
        self.var_alt_hi = tk.DoubleVar(value=hi_alt)
        self.var_only_multi_height = tk.BooleanVar(value=False)
        self.var_mode = tk.StringVar(value="per_sample")
        self.var_panels = tk.StringVar(value="4")
        self.var_logy = tk.BooleanVar(value=False)
        self.var_density = tk.BooleanVar(value=False)
        self.var_xrange = tk.StringVar(value="fit_nonzero")

        ttk.Label(filters, text="City:").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        city_combo = ttk.Combobox(filters, textvariable=self.var_city, values=cities, width=18, state="readonly")
        city_combo.grid(row=0, column=1, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="City type (3):").grid(row=0, column=2, sticky="e", padx=3, pady=3)
        c3_combo = ttk.Combobox(filters, textvariable=self.var_ctype3, values=ctypes3, width=18, state="readonly")
        c3_combo.grid(row=0, column=3, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="City type (6):").grid(row=0, column=4, sticky="e", padx=3, pady=3)
        c6_combo = ttk.Combobox(filters, textvariable=self.var_ctype6, values=ctypes6, width=20, state="readonly")
        c6_combo.grid(row=0, column=5, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="Expert mode:").grid(row=0, column=6, sticky="e", padx=3, pady=3)
        mode_combo = ttk.Combobox(filters, textvariable=self.var_mode_filter, values=modes, width=14, state="readonly")
        mode_combo.grid(row=0, column=7, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="Metric:").grid(row=0, column=8, sticky="e", padx=3, pady=3)
        metric_combo = ttk.Combobox(filters, textvariable=self.var_metric, values=metrics, width=18, state="readonly")
        metric_combo.grid(row=0, column=9, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="Expert:").grid(row=0, column=10, sticky="e", padx=3, pady=3)
        expert_combo = ttk.Combobox(filters, textvariable=self.var_expert, values=experts_vals, width=20, state="readonly")
        expert_combo.grid(row=0, column=11, sticky="ew", padx=3, pady=3)

        ttk.Label(filters, text="Alt min:").grid(row=1, column=0, sticky="e", padx=3, pady=3)
        alt_lo_spin = ttk.Spinbox(filters, from_=lo_alt, to=hi_alt, increment=1.0, textvariable=self.var_alt_lo, width=8)
        alt_lo_spin.grid(row=1, column=1, sticky="w", padx=3, pady=3)

        ttk.Label(filters, text="Alt max:").grid(row=1, column=2, sticky="e", padx=3, pady=3)
        alt_hi_spin = ttk.Spinbox(filters, from_=lo_alt, to=hi_alt, increment=1.0, textvariable=self.var_alt_hi, width=8)
        alt_hi_spin.grid(row=1, column=3, sticky="w", padx=3, pady=3)

        ttk.Checkbutton(
            filters,
            text="Only same-topology multi-height samples",
            variable=self.var_only_multi_height,
            command=self._apply_filters,
        ).grid(row=1, column=4, columnspan=3, sticky="w", padx=3, pady=3)

        ttk.Label(filters, text="X range:").grid(row=1, column=7, sticky="e", padx=3, pady=3)
        xrange_combo = ttk.Combobox(
            filters,
            textvariable=self.var_xrange,
            values=["full_range", "fit_nonzero"],
            width=14,
            state="readonly",
        )
        xrange_combo.grid(row=1, column=8, sticky="w", padx=3, pady=3)

        ttk.Button(filters, text="Apply filters", command=self._apply_filters).grid(row=1, column=11, sticky="e", padx=3, pady=3)

        options = ttk.LabelFrame(top, text="View")
        options.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(options, text="Mode:").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        ttk.Radiobutton(options, text="per-sample", variable=self.var_mode, value="per_sample", command=self._apply_filters).grid(row=0, column=1, sticky="w", padx=3, pady=3)
        ttk.Radiobutton(options, text="aggregate (filtered)", variable=self.var_mode, value="aggregate", command=self._apply_filters).grid(row=0, column=2, sticky="w", padx=3, pady=3)
        ttk.Radiobutton(options, text="same-topology multi-height", variable=self.var_mode, value="height_group", command=self._apply_filters).grid(row=0, column=3, sticky="w", padx=3, pady=3)

        ttk.Label(options, text="Panels:").grid(row=0, column=4, sticky="e", padx=3, pady=3)
        panels_combo = ttk.Combobox(options, textvariable=self.var_panels, values=["1", "2", "3", "4", "6", "8"], width=5, state="readonly")
        panels_combo.grid(row=0, column=5, sticky="w", padx=3, pady=3)

        ttk.Checkbutton(options, text="log-y", variable=self.var_logy, command=self._redraw).grid(row=0, column=6, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(options, text="density (normalize)", variable=self.var_density, command=self._redraw).grid(row=0, column=7, sticky="w", padx=6, pady=3)
        ttk.Label(options, text=self.topology_status).grid(row=0, column=8, sticky="w", padx=10, pady=3)

        show_frame = ttk.LabelFrame(top, text="Show")
        show_frame.pack(fill=tk.X)
        self.var_kinds = {k: tk.BooleanVar(value=True) for k in self.kinds}
        for i, kind in enumerate(self.kinds):
            ttk.Checkbutton(show_frame, text=kind, variable=self.var_kinds[kind], command=self._redraw).grid(
                row=i // 4,
                column=i % 4,
                sticky="w",
                padx=4,
                pady=2,
            )

        for combo in [city_combo, c3_combo, c6_combo, mode_combo, metric_combo, expert_combo, xrange_combo, panels_combo]:
            combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())
        alt_lo_spin.bind("<Return>", lambda _e: self._apply_filters())
        alt_hi_spin.bind("<Return>", lambda _e: self._apply_filters())
        alt_lo_spin.bind("<FocusOut>", lambda _e: self._apply_filters())
        alt_hi_spin.bind("<FocusOut>", lambda _e: self._apply_filters())

        middle = ttk.Frame(self.root)
        middle.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=2)

        left = ttk.Frame(middle)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Selection:").pack(anchor="w")

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.Y, expand=False)

        self.listbox = tk.Listbox(list_frame, width=52, height=36, exportselection=False)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.listbox.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.listbox.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._redraw())

        right = ttk.Frame(middle)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10.8, 7.2), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="ready")
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def _filtered_df(self) -> pd.DataFrame:
        df = self.df
        city = self.var_city.get()
        ctype3 = self.var_ctype3.get()
        ctype6 = self.var_ctype6.get()
        mode_filter = self.var_mode_filter.get()
        metric = self.var_metric.get()
        lo = float(self.var_alt_lo.get())
        hi = float(self.var_alt_hi.get())

        if city != "(all)":
            df = df[df["city"] == city]
        if ctype3 != "(all)":
            df = df[df["city_type_3"] == ctype3]
        if ctype6 != "(all)":
            df = df[df["city_type_6"] == ctype6]
        if mode_filter != "(all)":
            df = df[df["expert_mode"] == mode_filter]
        if metric != "(all)":
            df = df[df["metric"] == metric]
        df = df[(df["altitude_m"].between(lo, hi)) | (df["altitude_m"].isna())]

        expert = self.var_expert.get()
        if expert != "(all)" and self._expert_col:
            expert_values = df[self._expert_col].astype(str)
            is_truth = expert_values.eq("ground_truth")
            df = df[is_truth | expert_values.eq(expert)]

        if self.var_only_multi_height.get() and self.sample_to_group:
            df = df[df["sample_key"].isin(self.sample_to_group.keys())]

        return df

    def _build_sample_entries(self, fdf: pd.DataFrame) -> list[dict[str, Any]]:
        keys = (
            fdf[["city", "sample", "altitude_m", "city_type_3", "city_type_6", "sample_key"]]
            .drop_duplicates()
            .sort_values(["city", "sample", "altitude_m"])
        )
        entries: list[dict[str, Any]] = []
        for _, row in keys.iterrows():
            alt = row["altitude_m"]
            alt_str = f"{alt:.0f}m" if pd.notna(alt) else "?m"
            entries.append(
                {
                    "type": "sample",
                    "city": row["city"],
                    "sample": row["sample"],
                    "sample_key": row["sample_key"],
                    "altitude_m": float(alt) if pd.notna(alt) else float("nan"),
                    "city_type_3": row["city_type_3"] or "",
                    "city_type_6": row["city_type_6"] or "",
                    "label": f"{row['city']}/{row['sample']}  [{alt_str}]  c3={row['city_type_3'] or ''}  c6={row['city_type_6'] or ''}",
                }
            )
        return entries

    def _build_group_entries(self, fdf: pd.DataFrame) -> list[dict[str, Any]]:
        if not self.topology_groups:
            return []
        filtered_keys = set(fdf["sample_key"].dropna().astype(str).unique().tolist())
        entries: list[dict[str, Any]] = []
        for group_id, group in self.topology_groups.items():
            items = [item for item in group["items"] if item["sample_key"] in filtered_keys]
            heights = {round(float(item["altitude_m"]), 6) for item in items}
            if len(items) < 2 or len(heights) < 2:
                continue
            items.sort(key=lambda item: (item["altitude_m"], item["sample"]))
            alt_min = float(items[0]["altitude_m"])
            alt_max = float(items[-1]["altitude_m"])
            label = (
                f"{group['city']}  hash={group['topology_hash'][:12]}  "
                f"n={len(items)}  alt={alt_min:.0f}-{alt_max:.0f}m  "
                f"samples={', '.join(item['sample'] for item in items)}"
            )
            entries.append(
                {
                    "type": "group",
                    "group_id": group_id,
                    "city": group["city"],
                    "topology_hash": group["topology_hash"],
                    "items": items,
                    "label": label,
                }
            )
        entries.sort(key=lambda item: (item["city"], item["topology_hash"]))
        return entries

    def _apply_filters(self) -> None:
        fdf = self._filtered_df()
        if self.var_mode.get() == "height_group":
            self._entries = self._build_group_entries(fdf)
            item_name = "groups"
        else:
            self._entries = self._build_sample_entries(fdf)
            item_name = "samples"

        self.listbox.delete(0, tk.END)
        for entry in self._entries:
            self.listbox.insert(tk.END, entry["label"])

        if self._entries:
            cur = self.listbox.curselection()
            sel_idx = cur[0] if cur and cur[0] < len(self._entries) else 0
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(sel_idx)
            self.listbox.activate(sel_idx)

        self.status.set(f"{len(self._entries)} {item_name} after filter")
        self._redraw()

    def _selected_kinds(self) -> list[str]:
        return [kind for kind in self.kinds if self.var_kinds[kind].get()]

    def _panel_groups(self, selected_kinds: list[str]) -> list[list[str]]:
        if self.var_mode.get() == "height_group":
            return [[kind] for kind in selected_kinds]
        panel_count = int(self.var_panels.get()) if self.var_panels.get().isdigit() else 4
        template = GROUP_TEMPLATES.get(panel_count, GROUP_TEMPLATES[4])
        groups: list[list[str]] = []
        for group in template:
            keep = [kind for kind in group if kind in selected_kinds]
            if keep:
                groups.append(keep)
        if not groups and selected_kinds:
            groups = [selected_kinds]
        return groups

    def _panel_title(self, kinds: list[str]) -> str:
        labels = {
            ("target_los", "pred_los"): "LoS (target vs pred)",
            ("target_nlos", "pred_nlos"): "NLoS (target vs pred)",
            ("target_delay_spread", "pred_delay_spread"): "Delay spread",
            ("target_angular_spread", "pred_angular_spread"): "Angular spread",
            ("target_delay_spread",): "Delay target",
            ("pred_delay_spread",): "Delay pred",
            ("target_angular_spread",): "Angular target",
            ("pred_angular_spread",): "Angular pred",
            ("target_delay_spread", "pred_delay_spread", "target_angular_spread", "pred_angular_spread"): "Spread metrics",
        }
        return labels.get(tuple(kinds), " + ".join(kinds))

    def _grid_for_panels(self, n_panels: int) -> tuple[int, int]:
        if n_panels <= 1:
            return 1, 1
        if n_panels <= 2:
            return 1, 2
        if n_panels <= 4:
            return 2, 2
        if n_panels <= 6:
            return 2, 3
        return 2, 4

    def _axis_limits(self, series: list[np.ndarray]) -> tuple[float, float]:
        if self.var_xrange.get() == "full_range" or not series:
            return self.centers[0] - 0.5, self.centers[-1] + 0.5
        nonzero_mask = np.zeros(len(self.centers), dtype=bool)
        for counts in series:
            nonzero_mask |= np.asarray(counts) > 0
        if not np.any(nonzero_mask):
            return self.centers[0] - 0.5, self.centers[-1] + 0.5
        idx = np.flatnonzero(nonzero_mask)
        lo_idx = max(0, int(idx[0]) - 1)
        hi_idx = min(len(self.centers) - 1, int(idx[-1]) + 1)
        return self.centers[lo_idx] - 0.5, self.centers[hi_idx] + 0.5

    def _draw_empty(self, title: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title(title)
        ax.axis("off")
        self.canvas.draw_idle()

    def _redraw(self) -> None:
        self.fig.clear()
        fdf = self._filtered_df()
        selected_kinds = self._selected_kinds()
        if not selected_kinds:
            self._draw_empty("No histogram kinds selected")
            return

        mode = self.var_mode.get()
        density = self.var_density.get()
        panel_groups = self._panel_groups(selected_kinds)
        rows, cols = self._grid_for_panels(len(panel_groups))
        axes = self.fig.subplots(rows, cols, squeeze=False).ravel().tolist()

        if mode == "per_sample":
            if not self._entries:
                self._draw_empty("No samples after filter")
                return
            sel = self.listbox.curselection()
            idx = sel[0] if sel else 0
            entry = self._entries[idx]
            hists = pivot_sample(fdf, self.bin_cols, entry["city"], entry["sample"], self.kinds)
            title = (
                f"{entry['city']}/{entry['sample']}  alt={entry['altitude_m']:.0f}m  "
                f"c3={entry['city_type_3']}  c6={entry['city_type_6']}"
            )

            for i, ax in enumerate(axes):
                if i >= len(panel_groups):
                    ax.set_visible(False)
                    continue
                kinds = panel_groups[i]
                plotted: list[np.ndarray] = []
                for kind in kinds:
                    counts = hists[kind].astype(np.float64)
                    if density and counts.sum() > 0:
                        counts = counts / counts.sum()
                    plotted.append(counts)
                    ax.plot(
                        self.centers,
                        counts,
                        drawstyle="steps-mid",
                        color=self.kind_colors[kind],
                        label=kind,
                        linewidth=1.35,
                    )
                edge_color = PANEL_BORDER_PALETTE[i % len(PANEL_BORDER_PALETTE)]
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor(edge_color)
                ax.set_title(self._panel_title(kinds), fontsize=10)
                ax.set_xlabel("value [dB]")
                ax.set_ylabel("fraction" if density else "pixel count")
                ax.set_xlim(*self._axis_limits(plotted))
                if self.var_logy.get():
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.28)
                ax.legend(loc="upper right", fontsize=8, frameon=True)

        elif mode == "aggregate":
            hists = aggregate(fdf, self.bin_cols, self.kinds)
            n_samples = int(fdf[["city", "sample"]].drop_duplicates().shape[0])
            title = f"Aggregate of {n_samples} samples (filtered)"

            for i, ax in enumerate(axes):
                if i >= len(panel_groups):
                    ax.set_visible(False)
                    continue
                kinds = panel_groups[i]
                plotted = []
                for kind in kinds:
                    counts = hists[kind].astype(np.float64)
                    if density and counts.sum() > 0:
                        counts = counts / counts.sum()
                    plotted.append(counts)
                    ax.plot(
                        self.centers,
                        counts,
                        drawstyle="steps-mid",
                        color=self.kind_colors[kind],
                        label=kind,
                        linewidth=1.35,
                    )
                edge_color = PANEL_BORDER_PALETTE[i % len(PANEL_BORDER_PALETTE)]
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor(edge_color)
                ax.set_title(self._panel_title(kinds), fontsize=10)
                ax.set_xlabel("value [dB]")
                ax.set_ylabel("fraction" if density else "pixel count")
                ax.set_xlim(*self._axis_limits(plotted))
                if self.var_logy.get():
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.28)
                ax.legend(loc="upper right", fontsize=8, frameon=True)

        else:
            if not self.topology_groups:
                self._draw_empty(self.topology_status)
                return
            if not self._entries:
                self._draw_empty("No same-topology multi-height groups after filter")
                return
            sel = self.listbox.curselection()
            idx = sel[0] if sel else 0
            entry = self._entries[idx]
            panel_groups = [[kind] for kind in selected_kinds]
            rows, cols = self._grid_for_panels(len(panel_groups))
            self.fig.clear()
            axes = self.fig.subplots(rows, cols, squeeze=False).ravel().tolist()
            density = self.var_density.get()
            line_palette = [KIND_PALETTE[i % len(KIND_PALETTE)] for i in range(max(len(entry["items"]), 1))]
            title = (
                f"{entry['city']} same-topology multi-height  "
                f"hash={entry['topology_hash'][:12]}  "
                f"n={len(entry['items'])}"
            )

            item_hists = []
            for item in entry["items"]:
                item_hists.append(
                    (
                        item,
                        pivot_sample(fdf, self.bin_cols, item["city"], item["sample"], self.kinds),
                    )
                )

            for i, ax in enumerate(axes):
                if i >= len(panel_groups):
                    ax.set_visible(False)
                    continue
                kind = panel_groups[i][0]
                plotted = []
                for j, (item, hists) in enumerate(item_hists):
                    counts = hists[kind].astype(np.float64)
                    if density and counts.sum() > 0:
                        counts = counts / counts.sum()
                    plotted.append(counts)
                    ax.plot(
                        self.centers,
                        counts,
                        drawstyle="steps-mid",
                        color=line_palette[j],
                        label=f"{item['sample']} [{float(item['altitude_m']):.0f}m]",
                        linewidth=1.35,
                    )
                edge_color = PANEL_BORDER_PALETTE[i % len(PANEL_BORDER_PALETTE)]
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor(edge_color)
                ax.set_title(self._panel_title([kind]), fontsize=10)
                ax.set_xlabel("value [dB]")
                ax.set_ylabel("fraction" if density else "pixel count")
                ax.set_xlim(*self._axis_limits(plotted))
                if self.var_logy.get():
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.28)
                ax.legend(loc="upper right", fontsize=7, frameon=True)

        self.fig.suptitle(title, fontsize=11)
        self.canvas.draw_idle()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).parent / "histograms.csv"))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--hdf5", default="")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run compute_histograms.py first.", file=sys.stderr)
        sys.exit(2)

    config_path = Path(args.config).resolve() if str(args.config).strip() else None
    hdf5_path = Path(args.hdf5).resolve() if str(args.hdf5).strip() else None

    root = tk.Tk()
    Viewer(root, csv_path, config_path, hdf5_path)
    root.mainloop()


if __name__ == "__main__":
    main()
