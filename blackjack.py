"""Blackjack cards, hand evaluation, and game state."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("C", "D", "H", "S")
VALUES = {
    rank: 1 if rank == "A" else min(int(rank), 10) if rank.isdigit() else 10
    for rank in RANKS
}
MIN_DECKS, MAX_DECKS = 1, 8


def parse_cards(text: str) -> list[str]:
    """Parse comma-separated cards such as ``"As, 10h"``.

    Duplicates are allowed here because a multi-deck shoe holds several copies
    of each card; ``remaining_deck`` checks them against the shoe size.
    """
    cards = []
    for raw_card in text.replace(",", " ").split():
        card = raw_card.strip().upper()
        if not card:
            continue
        if len(card) not in (2, 3) or card[:-1] not in RANKS or card[-1] not in SUITS:
            raise ValueError(
                f"Invalid card '{raw_card.strip()}'. "
                "Use ranks 2-10, J/Q/K/A and suits C/D/H/S."
            )
        cards.append(card)
    if not cards:
        raise ValueError("Enter at least one card.")
    return cards


def card_rank(card: str) -> str:
    return card[:-1]


def card_value(card: str) -> int:
    return VALUES[card_rank(card)]


def hand_value(cards: list[str]) -> tuple[int, bool]:
    """Return the best hand total and whether an ace is counted as 11."""
    total = sum(card_value(card) for card in cards)
    aces = sum(card_rank(card) == "A" for card in cards)
    soft_aces = 0
    while aces and total + 10 <= 21:
        total += 10
        aces -= 1
        soft_aces += 1
    return total, soft_aces > 0


def is_pair(cards: list[str]) -> bool:
    """Two cards of equal blackjack value (10/J/Q/K count as a pair)."""
    return len(cards) == 2 and card_value(cards[0]) == card_value(cards[1])


def make_deck(num_decks: int = 1) -> list[str]:
    return [rank + suit for rank in RANKS for suit in SUITS] * num_decks


def remaining_deck(
    table_cards: list[str], player_cards: list[str], num_decks: int = 1
) -> list[str]:
    if not MIN_DECKS <= num_decks <= MAX_DECKS:
        raise ValueError(f"Number of decks must be {MIN_DECKS}-{MAX_DECKS}.")
    deck = make_deck(num_decks)
    for card in table_cards + player_cards:
        if card not in deck:
            raise ValueError(
                f"Card {card} appears more than {num_decks} time(s); "
                f"a {num_decks}-deck shoe has only {num_decks} copy(ies) of it."
            )
        deck.remove(card)
    return deck


def card_probabilities(deck: list[str]) -> dict[str, tuple[int, int]]:
    """Return each remaining rank and its count/total-card denominator."""
    counts = Counter(card_rank(card) for card in deck)
    return {rank: (counts[rank], len(deck)) for rank in RANKS if counts[rank]}


def bust_probability(player_cards: list[str], deck: list[str]) -> float:
    """Chance that the next card busts the player (0.0 if the deck is empty)."""
    if not deck:
        return 0.0
    busts = sum(hand_value(player_cards + [card])[0] > 21 for card in deck)
    return busts / len(deck)


@dataclass
class Game:
    table_cards: list[str]
    player_cards: list[str]
    num_decks: int = 1
    deck: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.table_cards = list(self.table_cards)
        self.player_cards = list(self.player_cards)
        self.deck = remaining_deck(self.table_cards, self.player_cards, self.num_decks)

    def add_card(self, card: str | None = None) -> str:
        """Deal ``card`` to the player, or a random remaining card if omitted."""
        if card is None:
            card = random.choice(self.deck)
        elif card not in self.deck:
            raise ValueError(f"Card '{card}' is not in the remaining shoe.")
        self.deck.remove(card)
        self.player_cards.append(card)
        return card