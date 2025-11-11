"""HTML report generation using Jinja2 and matplotlib."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader


@dataclass
class ReportContext:
    metrics: Dict[str, float]
    simulation: Dict[str, float]
    optimization: Dict[str, float]


def _plot_series(df: pd.DataFrame, column: str, path: pathlib.Path) -> str:
    plt.figure(figsize=(8, 3))
    plt.plot(df["timestamp"], df[column])
    plt.title(column.replace("_", " ").title())
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return str(path.name)


def generate_report(
    forecasts: pd.DataFrame,
    simulation: pd.DataFrame,
    optimization: pd.DataFrame,
    context: ReportContext,
    template_dir: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    template_env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = template_env.get_template("report.html")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assets_dir = output_path.parent / "assets"
    forecast_plot = _plot_series(forecasts, "yhat", assets_dir / "forecast.png")
    simulation_plot = _plot_series(simulation, "validations", assets_dir / "simulation.png")
    optimization_plot = _plot_series(optimization, "optimized_validations", assets_dir / "optimization.png")

    html = template.render(
        metrics=context.metrics,
        simulation=context.simulation,
        optimization=context.optimization,
        plots={
            "forecast": forecast_plot,
            "simulation": simulation_plot,
            "optimization": optimization_plot,
        },
    )

    output_path.write_text(html, encoding="utf-8")
