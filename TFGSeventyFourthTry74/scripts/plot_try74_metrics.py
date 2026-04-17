#!/usr/bin/env python3
"""Compatibility wrapper for the Try 74 plotter."""

from __future__ import annotations

from plot_try73_metrics import load_rows, main, plot_expert

__all__ = ["load_rows", "main", "plot_expert"]


if __name__ == "__main__":
    main()
