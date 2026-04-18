"""Interactive viewer for histograms CSV files.

Per-sample mode: for the selected sample, plot selected histogram kinds.

Aggregate mode: sum filtered samples' histograms and plot selected kinds.

Filters: city, city_type_3, city_type_6, expert_mode, metric, expert, altitude range.

Nothing is saved to disk; figures are drawn live into the Tk window.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DEFAULT_KIND_ORDER = [
    "target_los", "target_nlos", "pred_los", "pred_nlos",
    "target_delay_spread", "pred_delay_spread",
    "target_angular_spread", "pred_angular_spread",
]
KIND_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#17becf", "#bcbd22",
    "#e377c2", "#7f7f7f", "#4daf4a", "#984ea3",
]
PANEL_BORDER_PALETTE = [
    "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
    "#ff7f0e", "#17becf", "#bcbd22", "#8c564b",
]

GROUP_TEMPLATES = {
    1: [
        [
            "target_los", "target_nlos", "pred_los", "pred_nlos",
            "target_delay_spread", "pred_delay_spread",
            "target_angular_spread", "pred_angular_spread",
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
    8: [
        ["target_los"],
        ["target_nlos"],
        ["pred_los"],
        ["pred_nlos"],
        ["target_delay_spread"],
        ["pred_delay_spread"],
        ["target_angular_spread"],
        ["pred_angular_spread"],
    ],
}


def load_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    bin_cols = [c for c in df.columns if c.startswith("b") and c[1:].lstrip("-").isdigit()]
    bin_cols.sort(key=lambda c: int(c[1:]))
    centers = np.array([int(c[1:]) + 0.5 for c in bin_cols], dtype=np.float64)
    return df, centers, bin_cols


def pivot_sample(df: pd.DataFrame, bin_cols: list[str], city: str, sample: str, kinds: list[str]) -> dict[str, np.ndarray]:
    sub = df[(df["city"] == city) & (df["sample"] == sample)]
    out = {}
    for kind in kinds:
        rows = sub[sub["kind"] == kind]
        if rows.empty:
            out[kind] = np.zeros(len(bin_cols), dtype=np.int64)
        else:
            out[kind] = rows[bin_cols].to_numpy(dtype=np.int64).sum(axis=0)
    return out


def aggregate(df: pd.DataFrame, bin_cols: list[str], kinds: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for kind in kinds:
        sub = df[df["kind"] == kind]
        if sub.empty:
            out[kind] = np.zeros(len(bin_cols), dtype=np.int64)
        else:
            out[kind] = sub[bin_cols].to_numpy(dtype=np.int64).sum(axis=0)
    return out


class Viewer:
    def __init__(self, root: tk.Tk, csv_path: Path) -> None:
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
        kinds_in_csv = sorted([k for k in self.df["kind"].dropna().astype(str).unique().tolist() if k])
        self.kinds = [k for k in DEFAULT_KIND_ORDER if k in kinds_in_csv] + [k for k in kinds_in_csv if k not in DEFAULT_KIND_ORDER]
        self.kind_colors = {k: KIND_PALETTE[i % len(KIND_PALETTE)] for i, k in enumerate(self.kinds)}
        # altitude numeric
        self.df["altitude_m"] = pd.to_numeric(self.df["altitude_m"], errors="coerce")
        self._build_ui()
        self._apply_filters()

    def _build_ui(self) -> None:
        self.root.title(f"Histograms viewer — {self.csv_path.name}")
        self.root.geometry("1420x860")
        self.root.minsize(1220, 760)

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # city filter
        ttk.Label(top, text="City:").grid(row=0, column=0, sticky="e")
        cities = ["(all)"] + sorted(self.df["city"].dropna().unique().tolist())
        self.var_city = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_city, values=cities, width=18, state="readonly").grid(row=0, column=1, padx=4)

        # city_type_3 filter
        ttk.Label(top, text="City type (3):").grid(row=0, column=2, sticky="e")
        ctypes3 = ["(all)"] + sorted([c for c in self.df["city_type_3"].dropna().unique().tolist() if c])
        self.var_ctype3 = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_ctype3, values=ctypes3, width=18, state="readonly").grid(row=0, column=3, padx=4)

        # city_type_6 filter
        ttk.Label(top, text="City type (6):").grid(row=0, column=4, sticky="e")
        ctypes6 = ["(all)"] + sorted([c for c in self.df["city_type_6"].dropna().unique().tolist() if c])
        self.var_ctype6 = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_ctype6, values=ctypes6, width=22, state="readonly").grid(row=0, column=5, padx=4)

        # expert mode filter (3_experts / 6_experts)
        ttk.Label(top, text="Expert mode:").grid(row=0, column=6, sticky="e")
        modes = ["(all)"] + sorted([m for m in self.df["expert_mode"].dropna().unique().tolist() if m])
        self.var_mode_filter = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_mode_filter, values=modes, width=12, state="readonly").grid(row=0, column=7, padx=4)

        # metric filter
        ttk.Label(top, text="Metric:").grid(row=0, column=8, sticky="e")
        metrics = ["(all)"] + sorted([m for m in self.df["metric"].dropna().astype(str).unique().tolist() if m])
        self.var_metric = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_metric, values=metrics, width=16, state="readonly").grid(row=0, column=9, padx=4)

        # expert filter (for pred rows)
        ttk.Label(top, text="Expert:").grid(row=2, column=0, sticky="e", pady=4)
        expert_col = "expert_id" if "expert_id" in self.df.columns else ("expert" if "expert" in self.df.columns else None)
        self._expert_col = expert_col
        experts_vals = ["(all)"]
        if expert_col:
            experts_vals += sorted([e for e in self.df[expert_col].dropna().unique().tolist() if e and e != "ground_truth"])
        self.var_expert = tk.StringVar(value="(all)")
        ttk.Combobox(top, textvariable=self.var_expert, values=experts_vals, width=24, state="readonly").grid(row=2, column=1, columnspan=2, sticky="w", padx=4)

        # altitude range
        alt = self.df["altitude_m"].dropna()
        lo_alt = float(np.floor(alt.min())) if not alt.empty else 0.0
        hi_alt = float(np.ceil(alt.max())) if not alt.empty else 150.0
        ttk.Label(top, text="Alt min:").grid(row=1, column=4, sticky="e")
        self.var_alt_lo = tk.DoubleVar(value=lo_alt)
        ttk.Spinbox(top, from_=lo_alt, to=hi_alt, increment=1.0, textvariable=self.var_alt_lo, width=7).grid(row=1, column=5, padx=2)
        ttk.Label(top, text="Alt max:").grid(row=1, column=6, sticky="e")
        self.var_alt_hi = tk.DoubleVar(value=hi_alt)
        ttk.Spinbox(top, from_=lo_alt, to=hi_alt, increment=1.0, textvariable=self.var_alt_hi, width=7).grid(row=1, column=7, padx=2)

        # kinds to show
        ttk.Label(top, text="Show:").grid(row=1, column=8, sticky="e")
        self.var_kinds = {k: tk.BooleanVar(value=True) for k in self.kinds}
        kf = ttk.Frame(top)
        kf.grid(row=1, column=9, padx=2)
        for i, k in enumerate(self.kinds):
            r = i // 4
            c = i % 4
            ttk.Checkbutton(kf, text=k, variable=self.var_kinds[k], command=self._redraw).grid(row=r, column=c, padx=1, sticky="w")

        # log-y
        self.var_logy = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="log-y", variable=self.var_logy, command=self._redraw).grid(row=1, column=10, padx=6)

        # density
        self.var_density = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="density (normalize)", variable=self.var_density, command=self._redraw).grid(row=1, column=11, padx=6)

        # mode: per-sample / aggregate
        ttk.Label(top, text="Mode:").grid(row=2, column=0, sticky="e", pady=4)
        self.var_mode = tk.StringVar(value="per_sample")
        ttk.Radiobutton(top, text="per-sample", variable=self.var_mode, value="per_sample", command=self._redraw).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(top, text="aggregate (filtered)", variable=self.var_mode, value="aggregate", command=self._redraw).grid(row=2, column=2, sticky="w")

        ttk.Label(top, text="Panels:").grid(row=2, column=4, sticky="e", pady=4)
        self.var_panels = tk.StringVar(value="4")
        ttk.Combobox(
            top,
            textvariable=self.var_panels,
            values=["1", "2", "3", "4", "6", "8"],
            width=5,
            state="readonly",
        ).grid(row=2, column=5, sticky="w", padx=4)

        ttk.Button(top, text="Apply filters", command=self._apply_filters).grid(row=2, column=9, padx=6)

        # left: list of samples; right: matplotlib canvas
        middle = ttk.Frame(self.root)
        middle.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=2)

        left = ttk.Frame(middle)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Samples (filtered):").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=46, height=34, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._redraw())

        right = ttk.Frame(middle)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10.2, 6.4), dpi=100, constrained_layout=True)
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
        expert = self.var_expert.get() if hasattr(self, "var_expert") else "(all)"
        if expert != "(all)" and self._expert_col:
            # Keep ground_truth rows always; filter pred rows to the chosen expert.
            is_truth = df[self._expert_col].astype(str).eq("ground_truth")
            df = df[is_truth | df[self._expert_col].astype(str).eq(expert)]
        return df

    def _apply_filters(self) -> None:
        fdf = self._filtered_df()
        keys = fdf[["city", "sample", "altitude_m", "city_type_3", "city_type_6"]].drop_duplicates().sort_values(["city", "sample"])
        self.listbox.delete(0, tk.END)
        self._keys = []
        for _, row in keys.iterrows():
            alt = row["altitude_m"]
            alt_str = f"{alt:.0f}m" if pd.notna(alt) else "  ? m"
            c3 = row["city_type_3"] or ""
            c6 = row["city_type_6"] or ""
            self.listbox.insert(tk.END, f"{row['city']}/{row['sample']}  [{alt_str}]  c3={c3}  c6={c6}")
            self._keys.append((row["city"], row["sample"], float(alt) if pd.notna(alt) else float("nan"), c3, c6))
        self.status.set(f"{len(self._keys)} samples after filter")
        if self._keys and self.listbox.curselection() == ():
            self.listbox.selection_set(0)
        self._redraw()

    def _selected_kinds(self) -> list[str]:
        return [k for k in self.kinds if self.var_kinds[k].get()]

    def _panel_groups(self, selected_kinds: list[str], panel_count: int) -> list[list[str]]:
        template = GROUP_TEMPLATES.get(panel_count, GROUP_TEMPLATES[4])
        groups: list[list[str]] = []
        for g in template:
            keep = [k for k in g if k in selected_kinds]
            if keep:
                groups.append(keep)
        if not groups and selected_kinds:
            groups = [selected_kinds]
        return groups

    def _panel_title(self, kinds: list[str]) -> str:
        labels = {
            tuple(["target_los", "pred_los"]): "LoS (target vs pred)",
            tuple(["target_nlos", "pred_nlos"]): "NLoS (target vs pred)",
            tuple(["target_delay_spread", "pred_delay_spread"]): "Delay spread",
            tuple(["target_angular_spread", "pred_angular_spread"]): "Angular spread",
            tuple(["target_delay_spread"]): "Delay target",
            tuple(["pred_delay_spread"]): "Delay pred",
            tuple(["target_angular_spread"]): "Angular target",
            tuple(["pred_angular_spread"]): "Angular pred",
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

    def _redraw(self) -> None:
        self.fig.clear()
        fdf = self._filtered_df()
        mode = self.var_mode.get()
        if mode == "per_sample":
            if not self._keys:
                ax = self.fig.add_subplot(111)
                ax.set_title("no samples after filter")
                self.canvas.draw_idle()
                return
            sel = self.listbox.curselection()
            if not sel:
                sel = (0,)
            i = sel[0]
            city, sample, alt, ctype3, ctype6 = self._keys[i]
            hists = pivot_sample(fdf, self.bin_cols, city, sample, self.kinds)
            title = f"{city}/{sample}  alt={alt:.0f}m  c3={ctype3}  c6={ctype6}"
        else:
            hists = aggregate(fdf, self.bin_cols, self.kinds)
            title = f"Aggregate of {fdf[['city','sample']].drop_duplicates().shape[0]} samples (filtered)"

        density = self.var_density.get()
        selected_kinds = self._selected_kinds()
        if not selected_kinds:
            ax = self.fig.add_subplot(111)
            ax.set_title("No histogram kinds selected")
            self.canvas.draw_idle()
            return

        panel_count = int(self.var_panels.get()) if self.var_panels.get().isdigit() else 4
        groups = self._panel_groups(selected_kinds, panel_count)
        rows, cols = self._grid_for_panels(len(groups))
        axes = self.fig.subplots(rows, cols, squeeze=False).ravel().tolist()

        for i, ax in enumerate(axes):
            if i >= len(groups):
                ax.set_visible(False)
                continue

            kinds = groups[i]
            for kind in kinds:
                counts = hists[kind].astype(np.float64)
                if density and counts.sum() > 0:
                    counts = counts / counts.sum()
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
            if self.var_logy.get():
                ax.set_yscale("log")
            ax.grid(True, alpha=0.28)
            ax.legend(loc="upper right", fontsize=8, frameon=True)

        self.fig.suptitle(title, fontsize=11)
        self.canvas.draw_idle()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).parent / "histograms.csv"))
    args = ap.parse_args()
    p = Path(args.csv)
    if not p.exists():
        print(f"CSV not found: {p}", file=sys.stderr)
        print("Run compute_histograms.py first.", file=sys.stderr)
        sys.exit(2)
    root = tk.Tk()
    Viewer(root, p)
    root.mainloop()


if __name__ == "__main__":
    main()
