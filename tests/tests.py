"""Tests for every recommendation algorithm.

Run from the project root (the folder with blackjack.py, strategies.py, ...):
    python -m unittest discover -s tests -p "tests.py" -v
or:
    python tests/tests.py

Card format: rank+suit, e.g. "KD", "10S", "AH". Helper ``deal`` returns
(player_cards, table_cards, deck) with the dealt cards removed from a
multi-deck shoe, exactly as the game does.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack import parse_cards, remaining_deck
from strategies import (
    basic_strategy,
    expectimax_strategy,
    expectimax_values,
    monte_carlo_strategy,
    probability_strategy,
)
from pipeline import pipeline_strategy, pipeline_values

SIMS = 3_000
SEED = 1


def deal(table, player, decks=2):
    t, p = parse_cards(table), parse_cards(player)
    return p, t, remaining_deck(t, p, decks)


# ---------------------------------------------------------------- 1. Basic
class TestBasicStrategy(unittest.TestCase):
    def rec(self, table, player, **kw):
        p, t, _ = deal(table, player)
        return basic_strategy(p, t, **kw)

    def test_01_soft_18_vs_ten_hits(self):
        self.assertEqual(self.rec("kd", "ah 7s"), "hit")

    def test_02_hard_18_vs_ten_stands(self):
        self.assertEqual(self.rec("kd", "10s 8h"), "stand")

    def test_03_always_split_eights_and_aces(self):
        self.assertEqual(self.rec("6h", "8s 8c"), "split")
        self.assertEqual(self.rec("10h", "8s 8c"), "split")
        self.assertEqual(self.rec("10h", "ah ac"), "split")

    def test_04_never_split_fives_or_tens(self):
        self.assertNotEqual(self.rec("6h", "5s 5c"), "split")
        self.assertEqual(self.rec("6h", "10s 10c"), "stand")
        self.assertEqual(self.rec("6h", "js kc"), "stand")  # 10-value pair

    def test_05_nines_depend_on_dealer(self):
        self.assertEqual(self.rec("7h", "9s 9c"), "stand")
        self.assertEqual(self.rec("8h", "9s 9c"), "split")

    def test_06_threes_split_only_against_weak_dealer(self):
        self.assertEqual(self.rec("5h", "3s 3c"), "split")
        self.assertEqual(self.rec("10h", "3s 3c"), "hit")

    def test_07_no_double_and_no_split_flags_are_respected(self):
        self.assertEqual(self.rec("6h", "5s 5c", can_double=False), "hit")
        self.assertEqual(self.rec("6h", "8s 8c", can_split=False), "stand")

    def test_08_three_card_hand_cannot_double(self):
        self.assertEqual(self.rec("6h", "2s 4c 5d"), "hit")  # hard 11

    def test_09_soft_18_vs_six_doubles_or_stands(self):
        self.assertEqual(self.rec("6h", "ah 7s"), "double")
        self.assertEqual(self.rec("6h", "ah 7s", can_double=False), "stand")

    def test_10_soft_21_stands(self):
        self.assertEqual(self.rec("10h", "ah kc"), "stand")


# ------------------------------------------------------------ 2. Probability
class TestProbabilityStrategy(unittest.TestCase):
    def rec(self, table, player, **kw):
        p, _, d = deal(table, player)
        return probability_strategy(p, d, **kw)

    def test_01_splits_eights(self):
        self.assertEqual(self.rec("10h", "8s 8c"), "split")

    def test_02_splits_aces(self):
        self.assertEqual(self.rec("6h", "ah ac"), "split")

    def test_03_sevens_are_not_split(self):
        self.assertEqual(self.rec("6h", "7s 7c"), "stand")  # 46% bust risk

    def test_04_hard_18_stands(self):
        self.assertEqual(self.rec("kd", "10s 8h"), "stand")

    def test_05_soft_hand_cannot_bust_so_hits(self):
        self.assertEqual(self.rec("kd", "ah 6s"), "hit")  # soft 17

    def test_06_doubles_hard_11(self):
        self.assertEqual(self.rec("10h", "5s 6c"), "double")

    def test_07_hard_12_hits_low_bust_risk(self):
        self.assertEqual(self.rec("10h", "10s 2c"), "hit")  # ~31% < 40%

    def test_08_hard_16_stands_high_bust_risk(self):
        self.assertEqual(self.rec("6h", "10s 6d"), "stand")

    def test_09_no_double_when_not_allowed_or_three_cards(self):
        self.assertEqual(self.rec("10h", "5s 6c", can_double=False), "hit")
        self.assertEqual(self.rec("10h", "2s 4c 5d"), "hit")

    def test_10_depleted_deck_changes_decision(self):
        p = parse_cards("10s 6d")
        self.assertEqual(probability_strategy(p, ["2H", "3C", "4D"], can_double=False), "hit")
        self.assertEqual(probability_strategy(p, ["KH", "QC", "JD"], can_double=False), "stand")


# --------------------------------------------------------------- 3. Monte Carlo
class TestMonteCarlo(unittest.TestCase):
    def rec(self, table, player, **kw):
        p, t, d = deal(table, player)
        return monte_carlo_strategy(p, t, d, simulations=SIMS, seed=SEED, **kw)

    def test_01_hard_18_vs_ten_stands(self):
        self.assertEqual(self.rec("kd", "10s 8h"), "stand")

    def test_02_soft_18_vs_king_hits(self):
        self.assertEqual(self.rec("kd", "ah 7s"), "hit")

    def test_03_splits_eights_vs_six(self):
        self.assertEqual(self.rec("6h", "8s 8c"), "split")

    def test_04_splits_aces_vs_five(self):
        self.assertEqual(self.rec("5h", "ah ac"), "split")

    def test_05_doubles_pair_of_fives(self):
        self.assertEqual(self.rec("6h", "5s 5c"), "double")

    def test_06_tens_stand_instead_of_splitting(self):
        self.assertEqual(self.rec("10h", "10s 10c"), "stand")

    def test_07_splits_nines_vs_five(self):
        self.assertEqual(self.rec("5h", "9s 9c"), "split")

    def test_08_hard_16_vs_six_stands(self):
        self.assertEqual(self.rec("6h", "10s 6d"), "stand")

    def test_09_flags_remove_actions(self):
        self.assertEqual(self.rec("6h", "5s 5c", can_double=False), "hit")
        self.assertEqual(self.rec("6h", "8s 8c", can_split=False), "stand")

    def test_10_edge_cases_and_determinism(self):
        p, t, _ = deal("6h", "10s 6d")
        self.assertEqual(monte_carlo_strategy(p, t, [], seed=SEED), "stand")
        self.assertEqual(self.rec("10h", "ah kc"), "stand")  # soft 21
        self.assertEqual(self.rec("6h", "5s 5c"), self.rec("6h", "5s 5c"))


# ----------------------------------------------------------------- 4. Expectimax
class TestExpectimax(unittest.TestCase):
    def vals(self, table, player, **kw):
        p, t, d = deal(table, player)
        return expectimax_values(p, t, d, **kw)

    def test_01_split_eights_vs_six_is_best(self):
        v = self.vals("6h", "8s 8c")
        self.assertEqual(max(v, key=v.get), "split")
        self.assertAlmostEqual(v["split"], 0.333, delta=0.01)

    def test_02_hard_18_vs_ten_stand_ev(self):
        v = self.vals("kd", "10s 8h")
        self.assertEqual(max(v, key=v.get), "stand")
        self.assertAlmostEqual(v["stand"], -0.233, delta=0.01)

    def test_03_twenty_one_only_offers_stand(self):
        self.assertEqual(list(self.vals("10h", "ah kc")), ["stand"])

    def test_04_three_cards_no_double_no_split(self):
        v = self.vals("6h", "2s 4c 5d")
        self.assertNotIn("double", v)
        self.assertNotIn("split", v)

    def test_05_can_split_false_removes_split(self):
        self.assertNotIn("split", self.vals("6h", "8s 8c", can_split=False))

    def test_06_can_double_false_removes_double(self):
        self.assertNotIn("double", self.vals("6h", "5s 5c", can_double=False))

    def test_07_aces_split_best_vs_six(self):
        v = self.vals("6h", "ah ac")
        self.assertEqual(max(v, key=v.get), "split")
        self.assertAlmostEqual(v["split"], 0.713, delta=0.01)

    def test_08_tens_stand_beats_split(self):
        v = self.vals("10h", "10s 10c")
        self.assertEqual(max(v, key=v.get), "stand")
        self.assertGreater(v["stand"], v["split"])

    def test_09_fives_double_best(self):
        self.assertEqual(expectimax_strategy(*self._args("6h", "5s 5c")), "double")

    def test_10_rigged_deck_of_tens(self):
        p, t = parse_cards("10s qc"), parse_cards("5h")
        v = expectimax_values(p, t, ["KH"] * 20)
        self.assertAlmostEqual(v["hit"], -1.0)    # every card busts
        self.assertAlmostEqual(v["stand"], 1.0)   # dealer 5+10+10 busts

    @staticmethod
    def _args(table, player):
        p, t, d = deal(table, player)
        return p, t, d


# ------------------------------------------------------------------- 5. Pipeline
class TestPipeline(unittest.TestCase):
    def run_pipeline(self, table, player, **kw):
        p, t, d = deal(table, player)
        kw.setdefault("simulations", SIMS)
        kw.setdefault("seed", SEED)
        return pipeline_values(p, t, d, **kw)

    def test_01_clear_stand_skips_stage_two(self):
        s1, short, s2 = self.run_pipeline("kd", "10s 8h")
        self.assertEqual(short, ["stand"])
        self.assertIsNone(s2)

    def test_02_threes_vs_ten_only_hit_survives(self):
        s1, short, s2 = self.run_pipeline("10h", "3s 3c")
        self.assertEqual(short, ["hit"])  # split/double are > 0.05 behind
        self.assertIsNone(s2)

    def test_03_twenty_one_single_option(self):
        _, short, s2 = self.run_pipeline("10h", "ah kc")
        self.assertEqual(short, ["stand"])
        self.assertIsNone(s2)

    def test_04_close_call_runs_monte_carlo(self):
        s1, short, s2 = self.run_pipeline("10h", "5s 6c")  # hit vs double
        self.assertIn("hit", short)
        self.assertIn("double", short)
        self.assertIsNotNone(s2)
        self.assertEqual(set(s2), set(short))

    def test_05_nines_vs_seven_stand_or_split(self):
        s1, short, s2 = self.run_pipeline("7h", "9s 9c")
        self.assertEqual(set(short), {"stand", "split"})
        self.assertEqual(set(s2), {"stand", "split"})
        p, t, d = deal("7h", "9s 9c")
        self.assertIn(pipeline_strategy(p, t, d, simulations=SIMS, seed=SEED),
                      {"stand", "split"})

    def test_06_clear_split_is_decided_by_stage_one(self):
        s1, short, s2 = self.run_pipeline("6h", "8s 8c")
        self.assertEqual(short, ["split"])
        p, t, d = deal("6h", "8s 8c")
        self.assertEqual(pipeline_strategy(p, t, d), "split")

    def test_07_margin_zero_keeps_only_the_best(self):
        _, short, s2 = self.run_pipeline("10h", "5s 6c", margin=0)
        self.assertEqual(len(short), 1)
        self.assertIsNone(s2)

    def test_08_huge_margin_keeps_every_action(self):
        s1, short, s2 = self.run_pipeline("6h", "8s 8c", margin=10)
        self.assertEqual(set(short), set(s1))
        self.assertEqual(set(s2), set(s1))

    def test_09_same_seed_same_answer(self):
        a = self.run_pipeline("7h", "9s 9c")[2]
        b = self.run_pipeline("7h", "9s 9c")[2]
        self.assertEqual(a, b)

    def test_10_never_returns_illegal_action(self):
        p, t, d = deal("6h", "2s 4c 5d")  # three cards
        act = pipeline_strategy(p, t, d, simulations=SIMS, seed=SEED)
        self.assertIn(act, {"hit", "stand"})
        p, t, d = deal("6h", "5s 5c")
        act = pipeline_strategy(p, t, d, can_double=False, can_split=False,
                                simulations=SIMS, seed=SEED)
        self.assertIn(act, {"hit", "stand"})


if __name__ == "__main__":
    unittest.main(verbosity=2)