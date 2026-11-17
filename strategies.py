"""Algorithms that recommend the next blackjack action.

Actions: "hit", "stand", "double" (only on a 2-card hand) and "split" (only on
a 2-card pair). Rules assumed: dealer stands on all 17s, double allowed on any
two cards (also after a split), no re-splitting, split aces receive one card.
Payoffs: win +1, push 0, loss -1 (x2 when doubled); naturals are not modelled.
"""

from __future__ import annotations
from typing import Tuple
from functools import lru_cache
import random

from blackjack import (
    RANKS,
    VALUES,
    bust_probability,
    card_rank,
    card_value,
    hand_value,
)

NEG_INF = float("-inf")


def _dealer_up_value(table_cards: list[str]) -> int:
    """Dealer up card value (ace = 11). ``table_cards[0]`` is the up card."""
    return hand_value(table_cards[:1])[0] if table_cards else 10


def _pair_action(rank_value: int, up: int) -> bool:
    if rank_value in (1, 8):
        return True
    if rank_value == 9:
        return up in (2, 3, 4, 5, 6, 8, 9)
    if rank_value == 7:
        return 2 <= up <= 7
    if rank_value == 6:
        return 2 <= up <= 6
    if rank_value in (2, 3):
        return 2 <= up <= 7
    if rank_value == 4:
        return up in (5, 6)
    return False  # 5s and 10s never split


def basic_strategy(
    player_cards: list[str],
    table_cards: list[str],
    can_double: bool = True,
    can_split: bool = True,
) -> str:
    """Compact multi-deck basic strategy (dealer stands on all 17s)."""
    total, soft = hand_value(player_cards)
    up = _dealer_up_value(table_cards)
    two_cards = len(player_cards) == 2

    if can_split and two_cards and card_value(player_cards[0]) == card_value(player_cards[1]):
        if _pair_action(card_value(player_cards[0]), up):
            return "split"

    def double_or(fallback: str) -> str:
        return "double" if can_double and two_cards else fallback

    if soft:
        if total >= 20:
            return "stand"
        if total == 19:
            return "stand"
        if total == 18:
            if 3 <= up <= 6:
                return double_or("stand")
            return "hit" if up in (9, 10, 11) else "stand"
        if total == 17:
            return double_or("hit") if 3 <= up <= 6 else "hit"
        if total in (15, 16):
            return double_or("hit") if 4 <= up <= 6 else "hit"
        return double_or("hit") if 5 <= up <= 6 else "hit"  # soft 13-14

    if total >= 17:
        return "stand"
    if total >= 13:
        return "stand" if up <= 6 else "hit"
    if total == 12:
        return "stand" if 4 <= up <= 6 else "hit"
    if total == 11:
        return double_or("hit") if up <= 10 else "hit"
    if total == 10:
        return double_or("hit") if up <= 9 else "hit"
    if total == 9:
        return double_or("hit") if 3 <= up <= 6 else "hit"
    return "hit"


def probability_strategy(
    player_cards: list[str],
    deck: list[str],
    can_double: bool = True,
    can_split: bool = True,
    max_bust_risk: float = 0.40,
    min_double_chance: float = 0.40,
) -> str:
    """Decide from next-card probabilities only.

    - split A-A and 8-8;
    - double when one card lands on 19-21 with probability >= min_double_chance;
    - otherwise hit unless the hand is strong or the bust risk is too high.
    """
    total, soft = hand_value(player_cards)
    two_cards = len(player_cards) == 2

    if can_split and two_cards and card_value(player_cards[0]) == card_value(player_cards[1]):
        if card_value(player_cards[0]) in (1, 8):
            return "split"

    if can_double and two_cards and deck and not soft:
        good = sum(19 <= hand_value(player_cards + [c])[0] <= 21 for c in deck)
        if good / len(deck) >= min_double_chance:
            return "double"

    if total >= 18 or (total == 17 and not soft):
        return "stand"
    if soft:
        return "hit"  # a soft hand cannot bust on the next card
    return "stand" if bust_probability(player_cards, deck) > max_bust_risk else "hit"


# --- Monte Carlo ------------------------------------------------------------

def _play_dealer(dealer: list[str], pile: list[str]) -> int:
    dealer = list(dealer)
    while hand_value(dealer)[0] < 17 and pile:
        dealer.append(pile.pop())
    return hand_value(dealer)[0]


def _payoff(player_total: int, dealer_total: int) -> int:
    if player_total > 21:
        return -1
    if dealer_total > 21 or player_total > dealer_total:
        return 1
    return 0 if player_total == dealer_total else -1


def _rollout(hand: list[str], table_cards: list[str], pile: list[str]) -> int:
    """Finish a hand by following basic strategy (hit/stand only)."""
    while hand_value(hand)[0] < 21 and pile:
        if basic_strategy(hand, table_cards, False, False) != "hit":
            break
        hand.append(pile.pop())
    return hand_value(hand)[0]


def monte_carlo_strategy(
    player_cards: list[str],
    table_cards: list[str],
    deck: list[str],
    can_double: bool = True,
    can_split: bool = True,
    simulations: int = 5_000,
    seed: int | None = None,
) -> str:
    """Simulate each legal action, finishing hands with basic strategy."""
    if not deck:
        return "stand"
    two_cards = len(player_cards) == 2
    actions = ["stand", "hit"]
    if can_double and two_cards:
        actions.append("double")
    split_ok = (
        can_split
        and two_cards
        and card_value(player_cards[0]) == card_value(player_cards[1])
    )
    if split_ok:
        actions.append("split")

    rng = random.Random(seed)
    score = dict.fromkeys(actions, 0.0)
    for _ in range(simulations):
        shuffled = deck[:]
        rng.shuffle(shuffled)

        score["stand"] += _payoff(
            hand_value(player_cards)[0], _play_dealer(table_cards, shuffled[:])
        )

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

        if split_ok:
            pile = shuffled[:]
            totals = []
            for start in player_cards:
                hand = [start, pile.pop()]
                if card_rank(start) == "A":
                    totals.append(hand_value(hand)[0])  # one card on split aces
                else:
                    totals.append(_rollout(hand, table_cards, pile))
            dealer_total = _play_dealer(table_cards, pile)
            score["split"] += sum(_payoff(t, dealer_total) for t in totals)
    return max(actions, key=lambda a: score[a])


# --- Exact expectimax -------------------------------------------------------
# A hand is summarised as (hard_total, has_ace): only that matters for play.

Hand = Tuple[int, bool]


def _total(hand: Hand) -> int:
    hard, has_ace = hand
    return hard + 10 if has_ace and hard + 10 <= 21 else hard


def _hand(cards: list[str]) -> Hand:
    return (
        sum(card_value(c) for c in cards),
        any(card_rank(c) == "A" for c in cards),
    )


def _deal(hand: Hand, i: int) -> Hand:
    v = VALUES[RANKS[i]]
    return hand[0] + v, hand[1] or v == 1


def _counts(cards: list[str]) -> tuple[int, ...]:
    return tuple(sum(card_rank(c) == r for c in cards) for r in RANKS)


def _sub(counts: tuple[int, ...], i: int) -> tuple[int, ...]:
    return counts[:i] + (counts[i] - 1,) + counts[i + 1:]


@lru_cache(maxsize=None)
def _dealer_dist(hand: Hand, deck: tuple[int, ...]) -> tuple[float, ...]:
    """Dealer outcome probabilities: indices 0-4 = 17..21, index 5 = bust."""
    total = _total(hand)
    if total > 21:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    n = sum(deck)
    out = [0.0] * 6
    if total >= 17 or n == 0:
        out[total - 17 if total >= 17 else 0] = 1.0
        return tuple(out)
    for i, c in enumerate(deck):
        if c:
            sub = _dealer_dist(_deal(hand, i), _sub(deck, i))
            for k in range(6):
                out[k] += c / n * sub[k]
    return tuple(out)


def _stand_ev(total: int, dealer: Hand, deck: tuple[int, ...]) -> float:
    if total > 21:
        return -1.0
    dist = _dealer_dist(dealer, deck)
    ev = dist[5]
    for k in range(5):
        d = 17 + k
        ev += dist[k] * (1 if total > d else -1 if total < d else 0)
    return ev


@lru_cache(maxsize=None)
def _best(
    player: Hand, dealer: Hand, deck: tuple[int, ...], first: bool
) -> tuple[float, float, float]:
    """Return (stand_ev, hit_ev, double_ev) for a hand; ``first`` = 2 cards."""
    total = _total(player)
    stand = _stand_ev(total, dealer, deck)
    n = sum(deck)
    if total >= 21 or n == 0:
        return stand, NEG_INF, NEG_INF
    hit = 0.0
    double = 0.0
    for i, c in enumerate(deck):
        if not c:
            continue
        new = _deal(player, i)
        rest = _sub(deck, i)
        new_total = _total(new)
        p = c / n
        if new_total > 21:
            hit += p * -1.0
            double += p * -2.0
        else:
            s, h, _ = _best(new, dealer, rest, False)
            hit += p * max(s, h)
            if first:
                double += p * 2 * _stand_ev(new_total, dealer, rest)
    return stand, hit, (double if first else NEG_INF)


def _split_ev(
    pair_card: str, dealer: Hand, deck: tuple[int, ...], can_double: bool
) -> float:
    """EV of splitting: two hands that start from ``pair_card``.

    Approximation: both hands are valued against the same remaining deck.
    """
    start = _hand([pair_card])
    n = sum(deck)
    if n == 0:
        return NEG_INF
    ev = 0.0
    for i, c in enumerate(deck):
        if not c:
            continue
        hand = _deal(start, i)
        rest = _sub(deck, i)
        if card_rank(pair_card) == "A":  # one card only on split aces
            value = _stand_ev(_total(hand), dealer, rest)
        else:
            s, h, d = _best(hand, dealer, rest, True)
            value = max(s, h, d if can_double else NEG_INF)
        ev += c / n * value
    return 2 * ev


def expectimax_values(
    player_cards: list[str],
    table_cards: list[str],
    deck: list[str],
    can_double: bool = True,
    can_split: bool = True,
) -> dict[str, float]:
    """Expected payoff of every legal action (payoff in units of the bet)."""
    player, dealer, shoe = _hand(player_cards), _hand(table_cards), _counts(deck)
    two_cards = len(player_cards) == 2
    stand, hit, double = _best(player, dealer, shoe, two_cards)
    values = {"stand": stand}
    if hit > NEG_INF:
        values["hit"] = hit
    if can_double and two_cards and double > NEG_INF:
        values["double"] = double
    if (
        can_split
        and two_cards
        and card_value(player_cards[0]) == card_value(player_cards[1])
    ):
        split = _split_ev(player_cards[0], dealer, shoe, can_double)
        if split > NEG_INF:
            values["split"] = split
    return values


def expectimax_strategy(
    player_cards: list[str],
    table_cards: list[str],
    deck: list[str],
    can_double: bool = True,
    can_split: bool = True,
) -> str:
    values = expectimax_values(player_cards, table_cards, deck, can_double, can_split)
    return max(values, key=values.get)
