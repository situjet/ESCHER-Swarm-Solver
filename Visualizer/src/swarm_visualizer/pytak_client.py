"""PyTak client that streams game history to WinTAK."""

from __future__ import annotations

import asyncio
import configparser
from pathlib import Path
from typing import Iterable, List, Sequence

import pytak
from rich.console import Console
from rich.table import Table

from .config import PyTakRuntimeConfig, ScenarioBundle
from .cot import iter_history_as_cot
from .history import GameHistory

console = Console()


class HistoryWorker(pytak.QueueWorker):
    def __init__(
        self,
        tx_queue: asyncio.Queue,
        config: configparser.SectionProxy | dict,
        event_batches: Sequence[List[str]],
        step_delay: float,
        export_path: Path | None,
    ) -> None:
        super().__init__(tx_queue, config)
        self.event_batches = event_batches
        self.step_delay = step_delay
        self.export_path = export_path
        self._buffer: List[str] = []

    async def run(self, *args, **kwargs):  # noqa: D401
        """Push CoT events from the scenario into the TX queue."""
        for batch in self.event_batches:
            for cot_xml in batch:
                if self.export_path:
                    self._buffer.append(cot_xml)
                await self.put_queue(cot_xml.encode("utf-8"))
            await asyncio.sleep(self.step_delay)
        if self.export_path:
            Path(self.export_path).write_text("\n".join(self._buffer), encoding="utf-8")


class PyTakStreamer:
    def __init__(
        self,
        bundle: ScenarioBundle,
        runtime: PyTakRuntimeConfig,
    ) -> None:
        self.bundle = bundle
        self.runtime = runtime

    async def stream(self, history: GameHistory) -> None:
        batches = list(
            iter_history_as_cot(
                history.snapshots,
                grid=self.bundle.grid,
                geo=self.bundle.geo,
                runtime=self.runtime,
            )
        )
        if self.runtime.dry_run:
            self._print_dry_run(history.snapshots)
            self._export_batches(batches)
            return

        tx_queue: asyncio.Queue = asyncio.Queue()
        config_parser = configparser.ConfigParser()
        config_parser["pytak"] = self.runtime.to_pytak_config()
        config_section = config_parser["pytak"]

        history_worker = HistoryWorker(
            tx_queue,
            config_section,
            event_batches=batches,
            step_delay=self.runtime.step_delay,
            export_path=Path(self.runtime.export_file) if self.runtime.export_file else None,
        )
        tx_worker = await pytak.txworker_factory(tx_queue, config_section)

        await asyncio.gather(history_worker.run(), tx_worker.run())

    def stream_sync(self, history: GameHistory) -> None:
        asyncio.run(self.stream(history))

    # ------------------------------------------------------------------
    def _print_dry_run(self, snapshots: Sequence) -> None:
        table = Table(title="Swarm Dry-Run Feed", show_lines=True)
        table.add_column("Tick", justify="right")
        table.add_column("Drones")
        table.add_column("Interceptors")
        table.add_column("Targets")
        for snapshot in snapshots:
            drones = ", ".join(
                f"{d.drone_id}:{d.status.value}@({d.x},{d.y})" for d in snapshot.drones
            )
            interceptors = ", ".join(
                f"{i.interceptor_id}->{i.assigned_drone or '---'}@({i.x},{i.y})"
                for i in snapshot.interceptors
            )
            targets = ", ".join(
                f"{t.cluster_id}:{'X' if t.is_destroyed else 'O'}@({t.x},{t.y})"
                for t in snapshot.targets
            )
            table.add_row(str(snapshot.tick), drones, interceptors, targets)
        console.print(table)

    def _export_batches(self, batches: Sequence[List[str]]) -> None:
        if not self.runtime.export_file:
            return
        content = []
        for batch in batches:
            content.extend(batch)
        Path(self.runtime.export_file).write_text("\n".join(content), encoding="utf-8")
        console.print(f"[green]Exported CoT stream to {self.runtime.export_file}")
