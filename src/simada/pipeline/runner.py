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

"""Simulation runner — top-level pipeline orchestrator.

Runs all simulation units built by ScenarioBuilder, collects results,
and writes them to disk via StreamingSink.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from simada.core.config import SimulationConfig, load_config
from simada.pipeline.streaming import StreamingSink
from simada.scenario.builder import ScenarioBuilder, SimulationUnit

console = Console()
_log = logging.getLogger(__name__)


class SimulationRunner:
    """Top-level orchestrator for running simulation batches.

    Usage::

        runner = SimulationRunner(config_path, project_root)
        output_dir = runner.run()
    """

    def __init__(self, config: SimulationConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root

    @classmethod
    def from_yaml(cls, config_path: Path, project_root: Path) -> SimulationRunner:
        """Create a runner from a YAML config file."""
        config = load_config(config_path)
        return cls(config, project_root)

    def run(self) -> Path:
        """Run the full simulation pipeline.

        Returns the output directory path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{self._config.seed}_{timestamp}"
        output_dir = self._project_root / self._config.output.directory / run_name

        console.print(f"[bold]simada[/bold] — Simulation of AID Adherence")
        console.print(f"Seed: {self._config.seed}")
        console.print(f"Scenario: {self._config.scenario.name}")
        console.print(f"Duration: {self._config.scenario.duration_days} days")
        console.print()

        # Build simulation units
        console.print("Building scenarios...", style="dim")
        builder = ScenarioBuilder(self._config, self._project_root)
        units = builder.build()
        console.print(f"  {len(units)} patient(s) configured")

        # Initialize output sink
        sink = StreamingSink(output_dir, self._config.output.formats)
        sink.save_config_snapshot(self._config.model_dump(mode="json"))

        # Run simulations
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Simulating...", total=len(units))

            failed = 0
            for unit in units:
                name = unit.patient_profile.simglucose_name
                arch = unit.patient_profile.archetype_id.value
                progress.update(task, description=f"[cyan]{name}[/cyan] ({arch})")

                try:
                    results = self._run_single(unit)
                    sink.write_patient_results(results, unit.patient_profile)
                except Exception as e:
                    failed += 1
                    # BUG 12: bare `except Exception` only printed the message
                    # via console.print, discarding the traceback. In non-TTY
                    # deployments (log files, CI) the root cause was invisible,
                    # making partial cohorts extremely hard to diagnose.
                    # We now log at ERROR with exc_info=True so the full
                    # traceback appears in structured logs, while still printing
                    # a human-readable summary to the Rich console.
                    _log.exception(
                        "Patient simulation failed: %s (%s): %s",
                        name,
                        arch,
                        e,
                    )
                    console.print(
                        f"  [red]FAILED[/red] {name} ({arch}): {e}",
                        highlight=False,
                    )
                progress.advance(task)

        output_path = sink.finalize()

        console.print()
        if failed:
            console.print(
                f"[yellow]Done with {failed} failure(s).[/yellow] "
                f"Results saved to: {output_path}"
            )
        else:
            console.print(f"[green]Done![/green] Results saved to: {output_path}")
        return output_path

    def _run_single(self, unit: SimulationUnit) -> pd.DataFrame:
        """Run a single patient simulation and return the results DataFrame.

        Calls simulate() + results() directly on the SimObj, bypassing
        simglucose's sim() wrapper which redundantly writes a CSV and
        prints to stdout on every patient.
        """
        unit.sim_obj.simulate()
        return unit.sim_obj.results()
