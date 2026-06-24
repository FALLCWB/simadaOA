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

"""Tests for AGP, plots, and export utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

import warnings

from simada.analysis.agp import generate_agp, generate_comparison_agp
from simada.analysis.plots import _color_for_metric, plot_bg_trace, plot_metrics_comparison
from simada.analysis.style import CELL_GREEN, CELL_RED, CELL_YELLOW
from simada.analysis.utils import extract_archetype


class TestExtractArchetype:
    """Tests for the shared extract_archetype utility."""

    def test_extracts_adherent(self) -> None:
        assert extract_archetype("adult001_adherent_000") == "adherent"

    def test_extracts_nonadherent(self) -> None:
        assert extract_archetype("adolescent003_nonadherent_015") == "nonadherent"

    def test_extracts_moderate(self) -> None:
        assert extract_archetype("child010_moderate_029") == "moderate"

    def test_unknown_returns_unknown(self) -> None:
        assert extract_archetype("some_other_name") == "unknown"

    # Regression tests for BUG #7: previously a naive split-by-underscore
    # caused failures on names where an archetype-looking substring appeared
    # in a non-archetype position (e.g. "moderate-control"), and it could
    # not distinguish between names containing the archetype as a true
    # token vs. a substring.

    def test_nonadherent_not_confused_with_adherent(self) -> None:
        """'nonadherent' must not be misread as 'adherent' (substring trap)."""
        # If the regex alternation order were wrong, this would return
        # 'adherent' (the trailing component) instead of 'nonadherent'.
        assert extract_archetype("adult001_nonadherent_000") == "nonadherent"

    def test_archetype_at_start(self) -> None:
        """Archetype at the very start of the string should still match."""
        assert extract_archetype("adherent_run1") == "adherent"

    def test_archetype_at_end(self) -> None:
        """Archetype at the very end of the string should still match."""
        assert extract_archetype("run1_adherent") == "adherent"

    def test_archetype_only(self) -> None:
        """Bare archetype string should match."""
        assert extract_archetype("adherent") == "adherent"

    def test_substring_not_matched(self) -> None:
        """A substring not bounded by underscores must not match."""
        # "moderatecontrol" contains "moderate" but is not the archetype
        # token. Without word-like boundaries, the old code would have
        # returned "moderate" on "moderate_extra" but never on
        # "moderatecontrol"; the new regex correctly skips it.
        assert extract_archetype("moderatecontrol") == "unknown"

    def test_multiple_matches_last_wins(self) -> None:
        """When multiple archetype tokens appear, the LAST one wins."""
        # Filename convention puts the archetype after the patient id;
        # if a custom run somehow includes two archetype tokens, the
        # rightmost (most specific to the result) should win.
        assert extract_archetype("adherent_pool_moderate_000") == "moderate"


class TestAGP:
    """Tests for AGP generation (smoke tests — verify no crash)."""

    def _make_bg_series(self, n_days: int = 3) -> pd.Series:
        rng = np.random.default_rng(42)
        samples_per_day = 24 * 60 // 3
        n = n_days * samples_per_day
        bg = rng.normal(140, 40, size=n).clip(40, 400)
        return pd.Series(bg)

    @staticmethod
    def _assert_valid_png(path: Path) -> None:
        """Assert that a file is a well-formed PNG (not just non-empty).

        Checks the PNG magic bytes (first 8 bytes) and that the file size
        is large enough to contain real pixel data (> 10 000 bytes). A
        truncated or corrupted matplotlib save would produce either wrong
        magic bytes or a suspiciously small file.
        """
        _PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
        assert path.exists(), f"Output PNG does not exist: {path}"
        assert path.stat().st_size > 10_000, (
            f"PNG file too small ({path.stat().st_size} bytes); "
            "likely corrupted or empty matplotlib figure"
        )
        header = path.read_bytes()[:8]
        assert header == _PNG_MAGIC, (
            f"File does not start with PNG magic bytes; got {header!r}"
        )

    def test_generate_agp_saves_to_tmpdir(self, tmp_path: Path) -> None:
        bg = self._make_bg_series()
        out = tmp_path / "agp_smoke.png"
        generate_agp(bg, output_path=out)
        self._assert_valid_png(out)

    def test_generate_agp_saves_png(self, tmp_path: Path) -> None:
        bg = self._make_bg_series()
        out = tmp_path / "test_agp.png"
        generate_agp(bg, output_path=out)
        self._assert_valid_png(out)

    def test_comparison_agp_saves_to_tmpdir(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        n = 480 * 3
        bg_dict = {
            "adherent": pd.Series(rng.normal(120, 20, n).clip(40, 400)),
            "nonadherent": pd.Series(rng.normal(180, 60, n).clip(40, 400)),
        }
        out = tmp_path / "agp_cmp_smoke.png"
        generate_comparison_agp(bg_dict, output_path=out)
        self._assert_valid_png(out)


class TestPlots:
    """Tests for visualization functions (smoke tests)."""

    def test_bg_trace_runs(self) -> None:
        rng = np.random.default_rng(42)
        bg = pd.Series(rng.normal(140, 40, size=480).clip(40, 400))
        fig = plot_bg_trace(bg, title="Test Trace")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_bg_trace_saves_png(self, tmp_path: Path) -> None:
        bg = pd.Series(np.full(480, 120.0))
        out = tmp_path / "trace.png"
        plot_bg_trace(bg, output_path=out)
        assert out.exists()

    def test_metrics_comparison_runs(self) -> None:
        summary = pd.DataFrame([
            {"patient": "adult001_adherent_000", "tir": 75, "tbr_l1": 3,
             "tbr_l2": 0.5, "tar_l1": 18, "tar_l2": 3.5},
            {"patient": "adult002_nonadherent_001", "tir": 45, "tbr_l1": 8,
             "tbr_l2": 3, "tar_l1": 25, "tar_l2": 19},
        ])
        fig = plot_metrics_comparison(summary)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMetricColorThresholds:
    """Regression tests for BUG #1/#6: clinical color thresholds are strict
    inequalities. Values exactly AT the threshold must be YELLOW, not GREEN.
    """

    @staticmethod
    def _metrics(**overrides: float) -> dict:
        base = {
            "tir": 80, "tbr_l1": 0, "tbr_l2": 0,
            "tar_l1": 0, "tar_l2": 0, "gmi": 6.0, "cv": 30,
            "mean_bg": 130, "std_bg": 30, "lbgi": 1.0, "hbgi": 1.0, "mage": 30,
            "readings": 1000, "severe_hypo_episodes": 0,
            "bg_min": 80, "bg_max": 170, "max_bg": 170, "min_bg": 80,
        }
        base.update(overrides)
        return base

    # TBR L1: target <4%. v=4 is NOT on-target.
    def test_tbr_l1_strictly_below_4_is_green(self) -> None:
        assert _color_for_metric("TBR L1 (54-69)", self._metrics(tbr_l1=3.9)) == CELL_GREEN

    def test_tbr_l1_exactly_4_is_yellow(self) -> None:
        """Target is <4%, so exactly 4% must NOT be green (regression for off-by-one)."""
        assert _color_for_metric("TBR L1 (54-69)", self._metrics(tbr_l1=4.0)) == CELL_YELLOW

    def test_tbr_l1_above_8_is_red(self) -> None:
        assert _color_for_metric("TBR L1 (54-69)", self._metrics(tbr_l1=10.0)) == CELL_RED

    # TBR L2: target <1%.
    def test_tbr_l2_exactly_1_is_yellow(self) -> None:
        assert _color_for_metric("TBR L2 (<54)", self._metrics(tbr_l2=1.0)) == CELL_YELLOW

    def test_tbr_l2_below_1_is_green(self) -> None:
        assert _color_for_metric("TBR L2 (<54)", self._metrics(tbr_l2=0.5)) == CELL_GREEN

    # TAR L1: target <25%.
    def test_tar_l1_exactly_25_is_yellow(self) -> None:
        assert _color_for_metric("TAR L1 (181-250)", self._metrics(tar_l1=25.0)) == CELL_YELLOW

    def test_tar_l1_below_25_is_green(self) -> None:
        assert _color_for_metric("TAR L1 (181-250)", self._metrics(tar_l1=24.9)) == CELL_GREEN

    # TAR L2: target <5%.
    def test_tar_l2_exactly_5_is_yellow(self) -> None:
        assert _color_for_metric("TAR L2 (>250)", self._metrics(tar_l2=5.0)) == CELL_YELLOW

    def test_tar_l2_below_5_is_green(self) -> None:
        assert _color_for_metric("TAR L2 (>250)", self._metrics(tar_l2=4.9)) == CELL_GREEN


class TestAGPShortSeriesWarning:
    """Regression tests for BUG #3: AGP must warn when given <1 day of data."""

    def test_partial_day_emits_userwarning(self) -> None:
        """<1 day of samples should emit a UserWarning."""
        # 100 samples at 3-min intervals = 5 hours, well below one day.
        bg = pd.Series(np.full(100, 120.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_agp(bg)
            # Find a UserWarning mentioning the short-series condition.
            short_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning) and "<1 day" in str(w.message)
            ]
            assert short_warnings, f"expected UserWarning about <1 day, got: {[str(w.message) for w in caught]}"

    def test_full_day_does_not_emit_short_warning(self) -> None:
        """One full day of samples should NOT trigger the short-series warning."""
        samples_per_day = 24 * 60 // 3  # 480
        bg = pd.Series(np.full(samples_per_day, 120.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_agp(bg)
            short_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning) and "<1 day" in str(w.message)
            ]
            assert not short_warnings, (
                f"unexpected short-series warning with one full day: "
                f"{[str(w.message) for w in short_warnings]}"
            )

    def test_empty_series_emits_warning(self) -> None:
        """An empty BG series should also warn (no clinical signal at all)."""
        bg = pd.Series([], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_agp(bg)
            empty_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning) and "empty" in str(w.message).lower()
            ]
            assert empty_warnings, f"expected empty-BG warning, got: {[str(w.message) for w in caught]}"
