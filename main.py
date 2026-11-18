"""Menu flow for the interactive blackjack probability explorer."""

from __future__ import annotations

from blackjack import (
    MAX_DECKS,
    MIN_DECKS,
    Game,
    card_probabilities,
    card_rank,
    hand_value,
    is_pair,
    parse_cards,
)
from strategies import (
    basic_strategy,
    expectimax_strategy,
    expectimax_values,
    monte_carlo_strategy,
    probability_strategy,
)

from pipeline import pipeline_values

ALGORITHM_NAMES = {
    "1": "Basic strategy",
    "2": "Next-card probabilities",
    "3": "Monte Carlo simulation",
    "4": "Expectimax (exact expected value)",
    "5": "Manual choice",
    "6": "Pipeline (Expectimax -> Monte Carlo)",
}

QUIT_KEY = "0"


def read_num_decks() -> int:
    while True:
        text = input(f"How many decks does the game use [{MIN_DECKS}-{MAX_DECKS}]: ")
        if text.strip().isdigit() and MIN_DECKS <= int(text) <= MAX_DECKS:
            return int(text)
        print(f"Please enter a whole number from {MIN_DECKS} to {MAX_DECKS}.")


def choose_algorithm() -> str:
    print("\nChoose the algorithm:")
    for key, name in ALGORITHM_NAMES.items():
        print(f"  {key}. {name}")
    print(f"  {QUIT_KEY}. Quit")
    while True:
        choice = input(
            f"Algorithm [1-{len(ALGORITHM_NAMES)}, {QUIT_KEY} to quit]: "
        ).strip()
        if choice in ALGORITHM_NAMES or choice == QUIT_KEY:
            return choice
        print("Please choose a valid algorithm.")


def read_starting_cards() -> tuple[list[str], list[str]]:
    """Read '<table card>, <player card 1>, <player card 2>' in one prompt."""
    while True:
        try:
            cards = parse_cards(
                input("Cards [table card], [your 1st card], [your 2nd card]: ")
            )
            if len(cards) != 3:
                raise ValueError("Enter exactly 3 cards, e.g. 10s, ah, 6d.")
            return cards[:1], cards[1:]
        except ValueError as error:
            print(f"Input error: {error}")


def show_probabilities(game: Game) -> None:
    print(f"\nNext-card probabilities ({len(game.deck)} cards remaining):")
    for rank, (count, total) in card_probabilities(game.deck).items():
        print(f"  {rank:>2}: {count}/{total} ({count / total:.2%})")


def legal_actions(game: Game, can_split: bool) -> list[str]:
    actions = ["hit", "stand"]
    if len(game.player_cards) == 2:
        actions.append("double")
        if can_split and is_pair(game.player_cards):
            actions.append("split")
    return actions


def ask_manual_action(actions: list[str]) -> str:
    while True:
        action = input(f"Action ({'/'.join(actions)}): ").strip().lower()
        if action in actions:
            return action
        print("Please type one of the listed actions.")


def recommend(game: Game, choice: str, can_split: bool) -> str:
    actions = legal_actions(game, can_split)
    can_double, split_ok = "double" in actions, "split" in actions
    cards, table, deck = game.player_cards, game.table_cards, game.deck
    if choice == "1":
        action = basic_strategy(cards, table, can_double, split_ok)
    elif choice == "2":
        action = probability_strategy(cards, deck, can_double, split_ok)
    elif choice == "3":
        action = monte_carlo_strategy(cards, table, deck, can_double, split_ok)
    elif choice == "4":
        values = expectimax_values(cards, table, deck, can_double, split_ok)
        shown = ", ".join(f"{k}: {v:+.3f}" for k, v in values.items())
        print(f"Expected value per unit bet -> {shown}")
        action = expectimax_strategy(cards, table, deck, can_double, split_ok)
    elif choice == "5":
        return ask_manual_action(actions)
    elif choice == "6":
        stage1, shortlist, stage2 = pipeline_values(
            cards, table, deck, can_double, split_ok
        )
        print("Stage 1 EV - " + ", ".join(f"{k} {v:.3f}" for k, v in stage1.items()))
        if stage2:
            print("Stage 2 Monte Carlo (" + "/".join(shortlist) + ") - "
                  + ", ".join(f"{k} {v:.3f}" for k, v in stage2.items()))
            action = max(stage2, key=stage2.get)
        else:
            action = shortlist[0]
    
    print(f"Recommended action: {action}")
    return action


def deal_next_card(game: Game) -> str:
    """Let the user type the real card; empty input draws a random one."""
    while True:
        text = input("Card received (e.g. 5h, empty = random): ").strip().upper()
        try:
            return game.add_card(text or None)
        except ValueError as error:
            print(f"Input error: {error}")


def play_hand(game: Game, choice: str, can_split: bool = True) -> None:
    """Play one hand to the end; a split plays two hands from the same shoe."""
    while True:
        total, soft = hand_value(game.player_cards)
        qualifier = " (soft)" if soft else ""
        print(f"\nTable: {', '.join(game.table_cards)}")
        print(f"Player: {', '.join(game.player_cards)} -> {total}{qualifier}")

        if total > 21:
            print("The hand busted.")
            return
        if total == 21:
            print("21 reached. The hand stands.")
            return
        if not game.deck:
            print("No cards remain in the shoe.")
            return

        show_probabilities(game)
        action = recommend(game, choice, can_split)

        if action == "stand":
            print("The hand stands.")
            return
        if action == "hit":
            print(f"Next card: {deal_next_card(game)}")
        elif action == "double":
            print(f"Double down - one card only: {deal_next_card(game)}")
            total, _ = hand_value(game.player_cards)
            print(f"Final hand: {', '.join(game.player_cards)} -> {total}")
            return
        else:
            play_split(game, choice)
            return


def play_split(game: Game, choice: str) -> None:
    """Play two hands, each starting from one card of the pair (no re-split)."""
    pair = list(game.player_cards)
    for number, start_card in enumerate(pair, start=1):
        print(f"\n--- Split hand {number} of 2 ---")
        game.player_cards = [start_card]
        print(f"Starting card: {start_card}. Deal its second card.")
        deal_next_card(game)
        if card_rank(start_card) == "A":
            total, _ = hand_value(game.player_cards)
            print(f"Split aces get one card: {', '.join(game.player_cards)} -> {total}")
            continue
        play_hand(game, choice, can_split=False)


def main() -> None:
    print("Blackjack probability explorer")
    num_decks = read_num_decks()          # asked once per program run
    print("Cards are rank+suit, e.g. 10s, ah, 6d (suits C/D/H/S).")

    try:
        while True:
            choice = choose_algorithm()
            if choice == QUIT_KEY:
                print("Goodbye!")
                return
            print(f"{ALGORITHM_NAMES[choice]} with {num_decks} deck(s).")

            while True:
                table_cards, player_cards = read_starting_cards()
                try:
                    game = Game(table_cards, player_cards, num_decks)
                    break
                except ValueError as error:
                    print(f"Input error: {error}")

            play_hand(game, choice)
            print("\nGame over. Back to the algorithm menu.")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()