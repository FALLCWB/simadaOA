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
"""Map free-text food names from LLM-generated scenarios to TACO entries.

LLM output uses natural Brazilian food names with full accents/casing
(e.g. "Pao de queijo" vs "Pão de queijo", "Manga Tommy" vs "Manga tommy").
This module normalises (case + diacritics) for an exact match first, then
falls back to a fuzzy match against the TACO vocabulary. Items that cannot
be matched above a similarity threshold are reported so the caller can
decide to drop or substitute them.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches

from simada.meals.taco import Food, TACODatabase

logger = logging.getLogger(__name__)

# Connectives/prepositions that carry no food information. A query made only
# of these (e.g. JSON debris like "de", "with") must never token-match a TACO
# entry. Covers Portuguese (BR table) and English (US/JP tables); pass a custom
# set to TACOMapper for other vocabularies.
# NOTE: "sem"/"without" are negation particles, NOT connectives -- they change a
# food's identity ("cafe sem acucar" vs "cafe com acucar"). Stripping them would
# let a sugar-free query token-match a sugar-containing food, so they are NOT
# stopwords (keeping them lets the subset test exclude the wrong variant).
_STOPWORDS_PT = frozenset(
    {"de", "da", "do", "das", "dos", "com", "e", "a", "o", "as",
     "os", "em", "no", "na", "nos", "nas", "ao", "aos", "para", "por"}
)
_STOPWORDS_EN = frozenset(
    {"of", "with", "and", "the", "a", "an", "in", "on", "to"}
)
_STOPWORDS = _STOPWORDS_PT | _STOPWORDS_EN
# Negation particles that flip a food's identity (sugar-free vs sugar); never
# stripped, and used to reject fuzzy matches of the wrong polarity.
_NEGATION = frozenset({"sem", "without", "sans"})


def normalize(name: str) -> str:
    """Lowercase, strip diacritics/punctuation, collapse whitespace.

    Punctuation (commas, parentheses, etc.) is replaced by spaces so names
    like "Okayu (papa de arroz)" tokenize cleanly for matching.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_punct = re.sub(r"[^a-z0-9\s]", " ", no_accents.lower())
    return " ".join(no_punct.split())


def _name_aliases(name: str) -> list[str]:
    """Normalized aliases for a food name with parentheses.

    Returns the normalized segment BEFORE the first "(" and the normalized
    segment(s) INSIDE "()". For "Aji (carapau grelhado)" -> ["aji",
    "carapau grelhado"]; for "Rice ball (onigiri)" -> ["rice ball", "onigiri"].
    Names without parentheses yield no alias (the whole name is already the
    primary key). Lets bare romaji / common names resolve to "Romaji
    (description)" style entries.
    """
    if "(" not in name:
        return []
    aliases: list[str] = []
    lead = normalize(name.split("(", 1)[0])
    if lead:
        aliases.append(lead)
    for inside in re.findall(r"\(([^)]*)\)", name):
        norm_inside = normalize(inside)
        if norm_inside:
            aliases.append(norm_inside)
    return aliases


@dataclass(frozen=True)
class FoodMatch:
    """Result of mapping an LLM food string to TACO."""

    query: str
    food: Food | None
    method: str  # "exact", "token", "fuzzy", or "unmatched"
    score: float  # 1.0 for exact, difflib ratio for token/fuzzy, 0.0 for unmatched


class TACOMapper:
    """Resolve LLM food strings to TACO ``Food`` entries.

    Exact (normalised) match first, then a guarded token-subset match,
    then fuzzy fallback. ``fuzzy_cutoff`` is the minimum difflib similarity
    (0-1) to accept a fuzzy match; ``token_cutoff`` is the minimum difflib
    similarity to accept a token-subset match.
    """

    def __init__(
        self,
        taco: TACODatabase,
        fuzzy_cutoff: float = 0.82,
        token_cutoff: float = 0.6,
        stopwords: frozenset[str] | None = None,
    ) -> None:
        self._taco = taco
        self._cutoff = fuzzy_cutoff
        self._token_cutoff = token_cutoff
        self._stopwords = stopwords if stopwords is not None else _STOPWORDS
        # normalised name -> Food; on collision keep the first entry
        # (deterministic: TACODatabase preserves CSV order) and warn.
        self._norm_index: dict[str, Food] = {}
        for food in taco.all_foods:
            key = normalize(food.nome_pt)
            existing = self._norm_index.get(key)
            if existing is not None:
                logger.warning(
                    "Normalized name collision for %r: keeping first entry "
                    "%r, ignoring %r",
                    key,
                    existing.nome_pt,
                    food.nome_pt,
                )
                continue
            self._norm_index[key] = food
        self._norm_keys = list(self._norm_index.keys())

        # Alias index: many entries are "Romaji (description)" or
        # "English name (romaji)" -- e.g. "Aji (carapau grelhado)",
        # "Rice ball (onigiri)". An LLM generating Japanese scenarios naturally
        # writes the bare romaji ("Aji", "Onigiri"), which the token matcher
        # rejects as a single bare token. We register the part BEFORE the first
        # "(" and the part(s) INSIDE "()" as exact aliases so those natural
        # names resolve. Aliases that collide with a primary key or with each
        # other are dropped (ambiguous -> no alias), so this never overrides a
        # real exact match and never introduces a guessing match.
        alias_counts: dict[str, int] = {}
        alias_to_food: dict[str, Food] = {}
        for food in taco.all_foods:
            for alias in _name_aliases(food.nome_pt):
                if alias in self._norm_index:
                    continue  # never shadow a real primary name
                alias_counts[alias] = alias_counts.get(alias, 0) + 1
                alias_to_food[alias] = food
        self._alias_index: dict[str, Food] = {
            a: f for a, f in alias_to_food.items() if alias_counts[a] == 1
        }

    def match(self, name: str) -> FoodMatch:
        norm = normalize(name)
        exact = self._norm_index.get(norm)
        if exact is not None:
            return FoodMatch(name, exact, "exact", 1.0)
        # Romaji / parenthetical alias (e.g. "Aji" -> "Aji (carapau grelhado)",
        # "Onigiri" -> "Rice ball (onigiri)"). Unambiguous aliases only.
        alias = self._alias_index.get(norm)
        if alias is not None:
            return FoodMatch(name, alias, "alias", 1.0)
        # Token-subset: the LLM often drops a qualifier ("Feijao carioca"
        # for "Feijao carioca cozido"). Guards against false positives
        # ("Agua" -> "Agua de coco", "de" -> "Pao de alho"):
        #   1. the query must contain at least one non-stopword (content) token;
        #   2. the query's CONTENT tokens (stopwords stripped) must cover more
        #      than half of the candidate's content tokens (dropping one
        #      qualifier is fine, matching a fragment of a longer name is not);
        #   3. the overall similarity must reach ``token_cutoff``.
        q_tokens = set(norm.split())
        q_content = q_tokens - self._stopwords
        if q_content:
            candidates: list[tuple[float, int, str]] = []
            for key in self._norm_keys:
                k_tokens = key.split()
                k_set = set(k_tokens)
                k_content = k_set - self._stopwords
                if not q_content.issubset(k_set):
                    continue
                # A single bare content token (e.g. "white", "cooked", "agua",
                # "arroz") is structurally ambiguous: it cannot be told apart
                # from a dangerous fragment ("agua" -> "Agua de coco" injects
                # carbs). Since the generator constrains foods to the exact
                # vocabulary, bare tokens do not occur in valid output, so we
                # reject them rather than risk a spurious caloric match.
                if len(q_content) < 2:
                    continue
                # Multi-token: query content must cover MORE THAN half the
                # candidate's content tokens (dropping one qualifier is fine;
                # matching a small fragment of a longer name is not).
                if len(q_content) * 2 <= len(k_content):
                    continue
                score = SequenceMatcher(None, norm, key).ratio()
                if score >= self._token_cutoff:
                    candidates.append((score, -len(key), key))
            if candidates:
                score, _, best = max(candidates)
                return FoodMatch(name, self._norm_index[best], "token", score)
        close = get_close_matches(norm, self._norm_keys, n=1, cutoff=self._cutoff)
        if close:
            best_key = close[0]
            # Negation guard (also needed on the fuzzy path, not just token-subset):
            # "cafe sem acucar" is ~0.9 similar to "cafe com acucar", so fuzzy would
            # flip a sugar-free query to a sugar-containing food. Reject when the
            # negation polarity differs between query and candidate.
            q_neg = q_tokens & _NEGATION
            k_neg = set(best_key.split()) & _NEGATION
            if q_neg != k_neg:
                return FoodMatch(name, None, "unmatched", 0.0)
            food = self._norm_index[best_key]
            score = SequenceMatcher(None, norm, best_key).ratio()
            return FoodMatch(name, food, "fuzzy", score)
        return FoodMatch(name, None, "unmatched", 0.0)
