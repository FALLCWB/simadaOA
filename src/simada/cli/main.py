# simada -- Simulation of AID Adherence
# Copyright (C) 2026 Dr. Filipe Augusto da Luz Lemos, MSc. Ph.D.
# Contact: filipellemos@gmail.com | filipe@falleng.com.br | fadaluzl@syr.edu
#
# "Omnis enim res, quae dando non deficit, dum habetur et non datur,
#  nondum habetur quomodo habenda est." -- St. Augustine, De Doctrina Christiana
#  (A thing not diminished by being shared is not yet rightly possessed if only possessed and not shared.)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""simada CLI — command-line interface for running T1D simulations."""

from __future__ import annotations

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from simada import __version__
from simada.analysis.export import export_metrics_csv
from simada.core.config import load_config
from simada.pipeline.runner import SimulationRunner

app = typer.Typer(
    name="simada",
    help="simada — Simulation of AID Adherence. T1D behavioral simulation framework.",
    no_args_is_help=True,
)
console = Console()


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "simada").exists():
            return parent
    warnings.warn(
        f"Could not locate simada project root from {current}; "
        "falling back to cwd. Relative config paths may break.",
        stacklevel=2,
    )
    return current


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to YAML config file"),
    seed: int | None = typer.Option(None, help="Override seed from config"),
    dry_run: bool = typer.Option(False, help="Validate config without running"),
) -> None:
    """Run a simulation cohort from a YAML config file."""
    project_root = _find_project_root()

    sim_config = load_config(config)
    if seed is not None:
        sim_config.seed = seed

    if dry_run:
        console.print("[bold]Config validation:[/bold] OK")
        console.print(f"  Scenario: {sim_config.scenario.name}")
        console.print(f"  Duration: {sim_config.scenario.duration_days} days")
        console.print(f"  Seed: {sim_config.seed}")
        if sim_config.scenario.cohort:
            console.print(f"  Cohort size: {sim_config.scenario.cohort.size}")
            console.print(f"  Archetypes: {sim_config.scenario.cohort.archetype_distribution}")
        return

    runner = SimulationRunner(sim_config, project_root)
    runner.run()


@app.command()
def inspect(
    config: Path = typer.Argument(..., help="Path to YAML config file"),
) -> None:
    """Validate and display configuration details."""
    sim_config = load_config(config)

    table = Table(title="Simulation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Scenario", sim_config.scenario.name)
    table.add_row("Duration", f"{sim_config.scenario.duration_days} days")
    table.add_row("Start date", sim_config.scenario.start_date)
    table.add_row("Seed", str(sim_config.seed))
    table.add_row("Output", str(sim_config.output.directory))
    table.add_row("Formats", ", ".join(sim_config.output.formats))

    if sim_config.scenario.cohort:
        c = sim_config.scenario.cohort
        table.add_row("Cohort size", str(c.size))
        for arch, pct in c.archetype_distribution.items():
            table.add_row(f"  {arch}", f"{pct:.0%}")
        for reg, pct in c.insulin_regimen_distribution.items():
            table.add_row(f"  {reg}", f"{pct:.0%}")

    table.add_row("Exercise", str(sim_config.behavior.exercise.enabled))
    table.add_row("Stress", str(sim_config.behavior.stress.enabled))
    table.add_row("Alcohol", str(sim_config.behavior.alcohol.enabled))

    console.print(table)


@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Path to simulation run output directory"),
    output: Path | None = typer.Option(None, help="Save metrics CSV to this path"),
) -> None:
    """Compute and display glycemic metrics from simulation results."""
    summary = export_metrics_csv(results_dir, output)

    table = Table(title="Glycemic Metrics Summary")
    table.add_column("Patient", style="cyan")
    table.add_column("TIR %", justify="right")
    table.add_column("TBR L1 %", justify="right")
    table.add_column("TBR L2 %", justify="right")
    table.add_column("TAR L1 %", justify="right")
    table.add_column("TAR L2 %", justify="right")
    table.add_column("GMI", justify="right")
    table.add_column("CV %", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["patient"]),
            f"{row['tir']:.1f}",
            f"{row['tbr_l1']:.1f}",
            f"{row['tbr_l2']:.1f}",
            f"{row['tar_l1']:.1f}",
            f"{row['tar_l2']:.1f}",
            f"{row['gmi']:.2f}",
            f"{row['cv']:.1f}",
        )

    console.print(table)

    if output is not None:
        console.print(f"\nMetrics saved to: {output}")


@app.command()
def version() -> None:
    """Show simada version."""
    console.print(f"simada {__version__}")


if __name__ == "__main__":
    app()
