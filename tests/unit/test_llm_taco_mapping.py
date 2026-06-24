# simada -- Simulation of AID Adherence
# Copyright (C) 2026 Dr. Filipe Augusto da Luz Lemos, MSc. Ph.D.
# Contact: filipellemos@gmail.com | filipe@falleng.com.br | fadaluzl@syr.edu
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
"""Tests for the LLM->TACO food mapper."""

from pathlib import Path

import pytest

from simada.llm.taco_mapping import TACOMapper, normalize
from simada.meals.taco import TACODatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACO_CSV = PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"
JAPAN_CSV = PROJECT_ROOT / "data" / "taco" / "japan_foods.csv"


@pytest.fixture
def mapper() -> TACOMapper:
    return TACOMapper(TACODatabase(TACO_CSV))


def test_normalize_strips_accents_and_case():
    assert normalize("Pão de queijo") == "pao de queijo"
    assert normalize("Manga Tommy") == "manga tommy"
    assert normalize("  Café  com   leite ") == "cafe com leite"


def test_normalize_strips_punctuation():
    # japan_foods.csv names carry parentheses that must not break matching
    assert normalize("Okayu (papa de arroz)") == "okayu papa de arroz"
    assert normalize("Arroz, branco; cozido.") == "arroz branco cozido"


def test_exact_match_after_normalization(mapper: TACOMapper):
    # LLM emits accented/cased variants that must resolve exactly
    for q in ["Pão de queijo", "Manga Tommy", "Macarrão cozido", "Pão integral"]:
        m = mapper.match(q)
        assert m.food is not None, f"{q} should map"
        assert m.method == "exact", f"{q} should be exact after normalization, got {m.method}"
        assert m.score == 1.0


def test_fuzzy_match_for_near_miss(mapper: TACOMapper):
    # slight typo / variant should still resolve via fuzzy
    m = mapper.match("Arroz branco cozido ")
    assert m.food is not None
    m2 = mapper.match("Feijao carioca")  # missing "cozido"
    assert m2.food is not None
    assert m2.food.nome_pt == "Feijao carioca cozido"
    assert m2.method in ("token", "fuzzy")


def test_unmatched_returns_none(mapper: TACOMapper):
    m = mapper.match("Croissant com doce de leite")
    assert m.food is None
    assert m.method == "unmatched"
    assert m.score == 0.0


def test_single_generic_word_does_not_map_to_water_carbs(mapper: TACOMapper):
    # CRITICAL: "Agua" must NOT resolve to "Agua de coco" (water would
    # acquire 5 g carbs/100 g and corrupt the simulation input).
    m = mapper.match("Agua")
    assert m.food is None, f"'Agua' must not map to any food, got {m.food}"


def test_negation_sem_is_not_stripped(mapper: TACOMapper):
    # "sem" (without) must NOT be a stopword: a sugar-free query must never
    # resolve to a sugar-containing food (4x carbs -> wrong insulin dose).
    m = mapper.match("Cafe sem acucar")
    if m.food is not None:
        assert "sem acucar" in m.food.nome_pt.lower(), m.food.nome_pt
    # the two variants must stay distinguishable
    sugar = mapper.match("Cafe com leite e acucar")
    nosugar = mapper.match("Cafe com leite sem acucar")
    assert sugar.food is not None and nosugar.food is not None
    assert sugar.food.nome_pt != nosugar.food.nome_pt


def test_stopword_tokens_never_map(mapper: TACOMapper):
    # JSON debris / connective words must be rejected, not token-matched
    # ("de" -> "Pao de alho", "com" -> "Iogurte com frutas").
    for junk in ["de", "com", "e", "sem", "da", "do"]:
        m = mapper.match(junk)
        assert m.food is None, f"{junk!r} mapped to {m.food}"
        assert m.method == "unmatched"


def test_generic_category_word_does_not_map(mapper: TACOMapper):
    # "Frutas" is a generic category word; it must not map to ANY food (a wrong
    # match would inject that food's carbs), not merely avoid "Iogurte com frutas".
    m = mapper.match("Frutas")
    assert m.food is None, f"'Frutas' must not map to any food, got {m.food}"


def test_true_positive_token_matches_still_work(mapper: TACOMapper):
    cases = {
        "Arroz branco": "Arroz branco cozido",
        "Feijao carioca": "Feijao carioca cozido",
        "Pão de queijo": "Pao de queijo",
    }
    for query, expected in cases.items():
        m = mapper.match(query)
        assert m.food is not None, f"{query!r} should map"
        assert m.food.nome_pt == expected


def test_parenthesized_names_match_exactly():
    jp = TACOMapper(TACODatabase(JAPAN_CSV))
    m = jp.match("Okayu (papa de arroz)")
    assert m.food is not None
    assert m.method == "exact"
    # variant without parentheses also resolves to the same entry
    m2 = jp.match("Okayu papa de arroz")
    assert m2.food is not None
    assert m2.food.nome_pt == m.food.nome_pt


def test_bare_romaji_lead_maps_via_alias():
    """An LLM generating Japanese scenarios writes the bare romaji lead
    ("Aji", "Onigiri", "Natto", "Ryokucha") instead of the full
    "Romaji (description)" entry. These must resolve via the alias index
    (method == "alias"), not be rejected as single bare tokens.
    """
    jp = TACOMapper(TACODatabase(JAPAN_CSV))
    cases = {
        "Aji": "Aji (carapau grelhado)",
        "Onigiri": "Onigiri (bolinho de arroz)",
        "Natto": "Natto (soja fermentada)",
        "Ryokucha": "Ryokucha (cha verde)",
        "Saba": "Saba (cavalinha cozida)",
    }
    for romaji, expected in cases.items():
        m = jp.match(romaji)
        assert m.food is not None, f"{romaji!r} should map"
        assert m.method == "alias"
        assert m.food.nome_pt == expected


def test_alias_does_not_shadow_exact_or_introduce_ambiguity():
    """The full exact name still wins (method 'exact'), and a generic
    parenthetical shared by many foods (e.g. 'cozido'/'cooked') is NOT
    registered as an alias (ambiguous -> dropped)."""
    jp = TACOMapper(TACODatabase(JAPAN_CSV))
    full = jp.match("Aji (carapau grelhado)")
    assert full.method == "exact"
    # a generic descriptor common to many entries must not resolve to one food
    assert jp.match("cozido").method == "unmatched"


def test_normalized_name_collision_keeps_first(tmp_path, caplog):
    csv = tmp_path / "dup.csv"
    csv.write_text(
        "nome_pt,nome_en,categoria,carbs_per_100g,fibra_per_100g,"
        "indice_glicemico,porcao_tipica_g\n"
        "Açaí,Acai first,fruta,6.2,2.6,40,100\n"
        "Acai,Acai second,fruta,99.0,0.0,99,100\n",
        encoding="utf-8",
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="simada.llm.taco_mapping"):
        m = TACOMapper(TACODatabase(csv))
    assert any("collision" in r.message.lower() for r in caplog.records)
    match = m.match("acai")
    assert match.food is not None
    assert match.food.nome_en == "Acai first"  # first entry wins, deterministic


def test_real_sanity_foods_map_rate(mapper: TACOMapper):
    # foods observed in the qwen2.5:14b sanity run; most are accent variants
    observed = [
        "Café com leite sem açúcar", "Macarrão cozido", "Manga Tommy",
        "Pão de queijo", "Pão integral", "Omelete de queijo",
        "Frango grelhado", "Arroz branco cozido", "Feijao preto cozido",
        "Banana prata", "Cafe preto com acucar",
    ]
    matched = sum(1 for f in observed if mapper.match(f).food is not None)
    # at least the accent-variants (>=80%) must map
    assert matched / len(observed) >= 0.8, f"only {matched}/{len(observed)} mapped"


def test_english_qualifier_tokens_do_not_false_match():
    # H2/E2: stopwords were PT-only, so EN qualifier/connective tokens
    # false-matched on the US table ("cooked"->Corn cooked, "white"->White bread).
    us = TACOMapper(TACODatabase(PROJECT_ROOT / "data" / "taco" / "us_foods.csv"))
    for debris in ("cooked", "white", "bread", "with", "and"):
        assert us.match(debris).food is None, f"{debris!r} should not match"
    # content-word + EN-stopword (e.g. "macaroni and"): after stripping the EN
    # stopword the query is a single content token, which the bare-token rule
    # then rejects -- so this exercises the EN stopword set feeding that rule.
    for debris in ("macaroni and", "burrito with", "chicken and"):
        assert us.match(debris).food is None, f"{debris!r} should not match"
    # a genuine multi-token name still resolves
    assert us.match("White rice cooked").food is not None


def test_single_bare_token_rejected_no_spurious_carbs():
    br = TACOMapper(TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"))
    # "agua" must NOT map to "Agua de coco" (would inject carbs from water)
    for debris in ("agua", "de", "com", "frango", "arroz"):
        assert br.match(debris).food is None


def test_custom_stopwords_param():
    br = TACOMapper(TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"),
                    stopwords=frozenset({"xyz"}))
    # with a trivial stopword set the mapper still works for exact matches
    assert br.match("Arroz branco cozido").method == "exact"


def test_genuine_fuzzy_match_on_typo(mapper: TACOMapper):
    # a typo ("braco" for "branco") cannot match exactly or as a token subset;
    # it must resolve through the difflib fuzzy fallback, not silently unmatch
    m = mapper.match("Arroz braco cozido")
    assert m.method == "fuzzy"
    assert m.food is not None
    assert m.food.nome_pt == "Arroz branco cozido"
    assert m.score >= 0.82
