"""Two-stage pipeline: exact Expectimax filter -> Monte Carlo refinement.

Stage 1: expectimax_values() gives the exact expected value of every legal
         action; only actions within `margin` of the best one are kept.
Stage 2: if more than one action survives, only those are simulated with
         Monte Carlo (hands finished with basic strategy) and the best wins.

Stage 1 and stage 2 scores are never mixed: they use different play
assumptions after the first card, so stage 2 only ranks the shortlist
against itself.
"""
from __future__ import annotations

import random

from blackjack import card_rank, hand_value
from strategies import _payoff, _play_dealer, _rollout, expectimax_values


def monte_carlo_values(
    player_cards: list[str],
    table_cards: list[str],
    deck: list[str],
    actions: list[str],
    simulations: int = 5_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Average payoff per unit bet of each action in ``actions``."""
    if not deck:
        return {"stand": 0.0}
    rng = random.Random(seed)
    score = dict.fromkeys(actions, 0.0)
    player_total = hand_value(player_cards)[0]

    for _ in range(simulations):
        shuffled = deck[:]
        rng.shuffle(shuffled)

        if "stand" in score:
            score["stand"] += _payoff(
                player_total, _play_dealer(table_cards, shuffled[:])
            )

        if "hit" in score:
            pile = shuffled[:]
            hand = player_cards + [pile.pop()]
            total = _rollout(hand, table_cards, pile)
            score["hit"] += _payoff(total, _play_dealer(table_cards, pile))

        if "double" in score:
            pile = shuffled[:]
            hand = player_cards + [pile.pop()]
            score["double"] += 2 * _payoff(
                hand_value(hand)[0], _play_dealer(table_cards, pile)
            )

        if "split" in score and len(shuffled) >= 4:
            pile = shuffled[:]
            totals = []
            for start in player_cards:
                hand = [start, pile.pop()]
                if card_rank(start) == "A":  # one card on split aces
                    totals.append(hand_value(hand)[0])
                else:
                    totals.append(_rollout(hand, table_cards, pile))
            dealer_total = _play_dealer(table_cards, pile)
            score["split"] += sum(_payoff(t, dealer_total) for t in totals)

    return {a: s / simulations for a, s in score.items()}


def pipeline_values(
    player_cards: list[str],
    table_cards: list[str],
    deck: list[str],
    can_double: bool = True,
    can_split: bool = True,
    simulations: int = 5_000,
    seed: int | None = None,
    margin: float = 0.05,
) -> tuple[dict[str, float], list[str], dict[str, float] | None]:
    """Return (stage1 EVs, shortlist, stage2 MC values or None)."""
    stage1 = expectimax_values(player_cards, table_cards, deck, can_double, can_split)
    best = max(stage1.values())
    shortlist = [
        a for a, v in sorted(stage1.items(), key=lambda kv: -kv[1])
        if v >= best - margin
    ]
    if len(shortlist) == 1:
        return stage1, shortlist, None
    stage2 = monte_carlo_values(
        player_cards, table_cards, deck, shortlist, simulations, seed
    )
    return stage1, shortlist, stage2


def pipeline_strategy(*args, **kwargs) -> str:
    stage1, shortlist, stage2 = pipeline_values(*args, **kwargs)
    scores = stage2 or stage1
    return max(scores, key=scores.get)