import pytest
from workflows.cfb.colors import (
    BACKGROUND, CONTRAST_FLOOR, NEUTRAL, SIMILARITY_DELTA_E,
    contrast_ratio, delta_e, ensure_legible, resolve_pair,
)

TENNESSEE = ("#ff8200", "#ffffff")
GEORGIA_TECH = ("#b3a369", "#ffffff")
CLEMSON = ("#f56600", "#522d80")
PENN_STATE = ("#001e44", "#ffffff")
GEORGIA = ("#ba0c2f", "#2c2a29")


class TestDeltaE:
    def test_identical_colors_are_zero(self):
        assert delta_e("#ff8200", "#ff8200") == pytest.approx(0.0, abs=1e-9)

    def test_tennessee_and_clemson_orange_are_similar(self):
        # The motivating case: different hex, obviously "the same orange".
        assert delta_e("#ff8200", "#f56600") < SIMILARITY_DELTA_E

    def test_tennessee_orange_and_georgia_tech_gold_are_distinct(self):
        assert delta_e("#ff8200", "#b3a369") > SIMILARITY_DELTA_E

    def test_is_symmetric(self):
        assert delta_e("#ff8200", "#b3a369") == pytest.approx(delta_e("#b3a369", "#ff8200"))


class TestContrast:
    def test_bright_orange_clears_the_floor_on_the_dark_background(self):
        assert contrast_ratio("#ff8200", BACKGROUND) > CONTRAST_FLOOR

    def test_navy_fails_on_the_dark_background(self):
        assert contrast_ratio("#001e44", BACKGROUND) < CONTRAST_FLOOR

    def test_a_color_against_itself_is_one_to_one(self):
        assert contrast_ratio("#ff8200", "#ff8200") == pytest.approx(1.0)


class TestEnsureLegible:
    def test_leaves_an_already_legible_color_untouched(self):
        assert ensure_legible("#ff8200") == "#ff8200"

    def test_lifts_a_dark_color_to_exactly_clear_the_floor(self):
        out = ensure_legible("#001e44")
        assert out != "#001e44"
        assert contrast_ratio(out, BACKGROUND) >= CONTRAST_FLOOR

    def test_lifts_no_further_than_needed_so_the_hue_survives(self):
        # Georgia crimson should still read as crimson, not pink.
        out = ensure_legible("#ba0c2f")
        assert contrast_ratio(out, BACKGROUND) == pytest.approx(CONTRAST_FLOOR, abs=0.15)
        assert delta_e(out, "#ba0c2f") < 25


class TestResolvePair:
    def test_distinct_colors_both_keep_their_primary(self):
        home, away = resolve_pair(TENNESSEE, GEORGIA_TECH)
        assert home == "#ff8200"
        assert away == "#b3a369"

    def test_similar_primaries_send_the_away_team_to_its_alternate(self):
        # Tennessee AT Clemson: Clemson is home and keeps its orange.
        home, away = resolve_pair(CLEMSON, TENNESSEE)
        assert home == "#f56600"
        assert away == "#ffffff"

    def test_home_team_always_keeps_its_primary(self):
        home, _ = resolve_pair(TENNESSEE, CLEMSON)
        assert home == "#ff8200"

    def test_falls_back_to_neutral_when_the_alternate_is_also_too_close(self):
        near_white = ("#fefefe", "#ffffff")
        home, away = resolve_pair(near_white, ("#ffffff", "#fdfdfd"))
        assert away == NEUTRAL

    def test_a_near_black_alternate_is_lifted_rather_than_discarded(self):
        # Georgia's alternate (#2c2a29) is essentially the background, but
        # lifting rescues it into a legible grey, so it is still preferred
        # over the neutral.
        crimson_twin = ("#c01030", "#000000")
        home, away = resolve_pair(crimson_twin, GEORGIA)
        assert away != NEUTRAL
        assert contrast_ratio(away, BACKGROUND) >= CONTRAST_FLOOR
        assert delta_e(away, home) >= SIMILARITY_DELTA_E

    def test_falls_back_to_neutral_when_both_candidates_clash(self):
        # A school whose primary and alternate are both the same orange as
        # the home team has nothing left to offer.
        home, away = resolve_pair(("#ff8200", "#ffffff"), ("#fe8100", "#f98a10"))
        assert away == NEUTRAL

    def test_dark_primaries_are_lifted_rather_than_rendered_invisible(self):
        home, away = resolve_pair(PENN_STATE, GEORGIA)
        for c in (home, away):
            assert contrast_ratio(c, BACKGROUND) >= CONTRAST_FLOOR

    def test_missing_alternate_falls_through_to_neutral(self):
        home, away = resolve_pair(("#ff8200", None), ("#fe8100", None))
        assert away == NEUTRAL

    def test_missing_primary_falls_back_to_neutral(self):
        home, away = resolve_pair((None, None), GEORGIA_TECH)
        assert home == NEUTRAL

    def test_similarity_is_judged_on_the_rendered_color_not_the_raw_hex(self):
        # Penn State and Notre Dame are different navies that lift to near-identical
        # blues; the collision must still be caught after lifting.
        notre_dame = ("#0c2340", "#c99700")
        home, away = resolve_pair(PENN_STATE, notre_dame)
        assert delta_e(home, away) >= SIMILARITY_DELTA_E


class TestStoreIntegration:
    """The colour lookup has to survive cfb_teams being absent or incomplete."""

    def test_pair_colors_falls_back_to_neutral_for_unknown_schools(self):
        from workflows.cfb.matchup import Store, _pair_colors
        store = Store(profile=None, games=None, model_row={}, loaded_at=0.0, colors={})
        home, away = _pair_colors(store, "Tennessee", "Georgia Tech")
        assert home == NEUTRAL
        assert away == NEUTRAL

    def test_pair_colors_uses_the_store_when_present(self):
        from workflows.cfb.matchup import Store, _pair_colors
        store = Store(profile=None, games=None, model_row={}, loaded_at=0.0,
                      colors={"Tennessee": TENNESSEE, "Georgia Tech": GEORGIA_TECH})
        home, away = _pair_colors(store, "Tennessee", "Georgia Tech")
        assert home == "#ff8200"
        assert away == "#b3a369"

    def test_one_known_and_one_unknown_school_still_resolves(self):
        from workflows.cfb.matchup import Store, _pair_colors
        store = Store(profile=None, games=None, model_row={}, loaded_at=0.0,
                      colors={"Tennessee": TENNESSEE})
        home, away = _pair_colors(store, "Tennessee", "Nowhere State")
        assert home == "#ff8200"
        assert away == NEUTRAL
