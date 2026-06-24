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

"""Shared analysis utilities."""

from __future__ import annotations

import re

from simada.analysis.style import WATERMARK, add_watermark  # noqa: F401

ARCHETYPE_NAMES = frozenset({"adherent", "moderate", "nonadherent"})

# Match an archetype token bounded by underscores or string ends. Order
# matters: "nonadherent" must be listed before "adherent" because regex
# alternation is left-to-right greedy and "adherent" would otherwise
# capture the "adherent" tail of "nonadherent" and report the wrong
# archetype.
_ARCHETYPE_RE = re.compile(r"(?:^|_)(nonadherent|adherent|moderate)(?:_|$)")


def extract_archetype(patient_name: str) -> str:
    """Extract archetype from patient filename like 'adult001_adherent_000'.

    Uses a regex with word-like boundaries (underscores or string ends)
    so we don't get a false hit from a custom name that happens to contain
    an archetype substring (e.g. ``my-adherent-cohort.parquet`` is not a
    valid simada filename). If multiple archetype tokens appear, the
    LAST one wins -- conventional simada filenames place the archetype
    after the patient id (``adult001_adherent_000``), so the rightmost
    match is the most specific.
    """
    matches = list(_ARCHETYPE_RE.finditer(patient_name))
    if not matches:
        return "unknown"
    return matches[-1].group(1)
