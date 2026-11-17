# Blackjack Probability Explorer

An interactive command-line tool that recommends the next blackjack action
(hit, stand, double or split) using several different algorithms, so you can
compare how each one reasons about the same hand.

You enter the dealer's up card and your two cards. The program removes them
from a multi-deck shoe, shows the probability of each next card, and
recommends an action. You then type the card you actually received (or press
Enter for a random one) and it recommends again, until the hand ends.

## Project structure

```
blackjack.py    Cards, hand evaluation, shoe and game state
strategies.py   Basic strategy, next-card probabilities, Monte Carlo, Expectimax
pipeline.py     Two-stage pipeline: Expectimax filter -> Monte Carlo refinement
main.py         Menu and interactive game loop
tests/tests.py  50 unit tests (10 per algorithm)
```

## Running

Requires Python 3.10 or newer. There are no external dependencies.

```
python main.py
```

1. Choose the number of decks (1 to 8).
2. Choose an algorithm (the same one is used for every step of the round).
3. Enter three cards: dealer card, your first card, your second card.
   Cards are rank + suit, using ranks 2-10, J, Q, K, A and suits C, D, H, S.
   Spaces, commas or both can separate them, e.g. `kd ah 7s` or `kd, ah, 7s`.
4. After each recommendation, type the card you received (e.g. `5h`), or press
   Enter to draw a random one. The prompt only accepts cards, not actions.

## The algorithms

| # | Algorithm | How it decides |
|---|-----------|----------------|
| 1 | Basic strategy | Fixed lookup table using your total, soft/hard, the dealer card and pairs. Ignores which cards are left. |
| 2 | Next-card probabilities | Looks one card ahead at the real remaining deck: splits A-A and 8-8, doubles if one card reaches 19-21 with probability >= 40%, otherwise hits unless the bust risk is above 40%. |
| 3 | Monte Carlo | Simulates each legal action many times on random shuffles of the remaining deck and picks the best average payoff. |
| 4 | Expectimax | Computes the exact expected payoff of every action by exploring every possible card, assuming optimal play after the first move. |
| 5 | Manual choice | You type the action yourself. |
| 6 | Pipeline | Stage 1 runs Expectimax and keeps the actions within a margin of the best. Stage 2 runs Monte Carlo only on those actions if more than one remains. |

### Monte Carlo in detail

For each of 5,000 simulations the remaining cards are shuffled once, and every
legal action is tried on that same shuffle:

- stand: the dealer plays out; score +1 / 0 / -1
- hit: take one card, then finish the hand with basic strategy; then the dealer plays
- double: one card, then stand; payoff counted twice
- split: both hands are dealt from the same pile (split aces get one card each)

The action with the highest average payoff is recommended. Because it is
sampling, results are slightly noisy, and cards after the first move are played
by basic strategy rather than searched for the best move.

### Expectimax in detail

Expectimax walks the full decision tree:

- Chance nodes average over every possible next card, weighted by
  `count / cards remaining`.
- Decision nodes take the maximum of standing and hitting.
- The dealer's final total is computed exactly from the remaining deck.
- Results are cached with `lru_cache`, using a compact hand summary
  `(hard_total, has_ace)` and a tuple of 13 rank counts as the deck.

Splitting is valued by averaging over the second card of one hand and
multiplying by two (two bets, each with the same expected value before any card
is drawn). This is an approximation: both hands are valued against the same
remaining deck, so card removal between the two hands is ignored.

### Pipeline in detail

1. Compute every action's exact expected value with Expectimax.
2. Keep actions within `margin` (default 0.05) of the best one.
3. If one action remains, return it. Otherwise, simulate only those actions
   with Monte Carlo and pick the best.

Expectimax and Monte Carlo values are never mixed, because they use different
assumptions about later play. Stage 2 only ranks the shortlist against itself.

## Rules assumed (and where in the code)

- **Dealer stands on all 17s.**
  The dealer draws until reaching 17 or more, then stops, including on soft 17.
  Code: `_play_dealer` in `strategies.py` loops `while hand_value(dealer)[0] < 17`
  (Monte Carlo). `_dealer_dist` in `strategies.py` stops recursing once
  `total >= 17` (Expectimax). `hand_value` in `blackjack.py` returns the best
  total, so A+6 counts as 17 and the dealer stands.

- **Double is allowed on any two cards, including after a split.**
  Doubling means one extra card and then standing, with the bet doubled.
  Code: `legal_actions` in `main.py` adds `double` whenever
  `len(game.player_cards) == 2`. The strategies check the same condition
  (`two_cards` in `basic_strategy`, `probability_strategy` and
  `monte_carlo_strategy`; `first` in `_best` for Expectimax). `play_hand` in
  `main.py` handles `double` by dealing exactly one card and ending the hand.
  After a split, `play_split` calls `play_hand` for each hand, which again has
  two cards, so doubling is offered. `_split_ev` includes it with
  `d if can_double`. Monte Carlo's split simulation finishes split hands with
  `_rollout` (hit/stand only), so it does not simulate doubling after a split.

- **A pair means two cards of equal value (so K-Q counts as a pair).**
  Splitting is decided by value, not by rank.
  Code: `is_pair` in `blackjack.py` returns `card_value(cards[0]) == card_value(cards[1])`
  for a 2-card hand, and `legal_actions` in `main.py` uses it to offer `split`.
  The strategies repeat the same comparison inline in `basic_strategy`,
  `probability_strategy`, `monte_carlo_strategy` and `expectimax_values`.
  Basic strategy never splits 10-value pairs (`_pair_action`).

- **No re-splitting. Split aces receive one card each.**
  After a split, each hand plays alone and cannot be split again, and a split
  ace gets a single card and then stops.
  Code (no re-splitting): `play_split` in `main.py` plays each hand with
  `play_hand(game, choice, can_split=False)`. In `strategies.py`, `_split_ev`
  continues with `_best`, which has no split option, and Monte Carlo finishes
  split hands with `_rollout`, which only hits or stands.
  Code (split aces): `play_split` checks `card_rank(start_card) == "A"`, deals
  one card, prints the total and moves to the next hand. `_split_ev` and the
  split branch of `monte_carlo_strategy` do the same, valuing the hand right
  after that single card.

- **Payoffs: win +1, push 0, loss -1 (times 2 when doubled).**
  Results are measured in units of the original bet.
  Code: `_payoff` in `strategies.py` returns -1 for a player bust, +1 for a win
  or a dealer bust, 0 for a push, and -1 for a loss. `_stand_ev` applies the
  same scoring to the exact dealer distribution. In `_best`, a bust gives
  `-1.0` for hit and `-2.0` for double, and a double that survives is
  `2 * _stand_ev(...)`. Monte Carlo's double uses `2 * _payoff(...)`, and a
  split adds the payoffs of both hands.

- **Blackjack payouts (naturals) and insurance are not modelled.**
  Code: nothing in the project handles them. A starting Ace + 10-value hand is
  treated as an ordinary 21: `play_hand` in `main.py` prints
  "21 reached. The hand stands." when the total is 21, and the strategies only
  return `stand` for it. The docstring at the top of `strategies.py` states
  this assumption.

- **The dealer's hidden card is treated as unknown, drawn from the remaining deck.**
  You only enter the dealer's up card, so the hole card is just another random
  card from the shoe.
  Code: `Game.__post_init__` in `blackjack.py` calls
  `remaining_deck(table_cards, player_cards, num_decks)`, which removes only the
  up card and your cards. The dealer's later cards, including the hole card, are
  drawn from that deck by `_play_dealer` (Monte Carlo) and by the loop in
  `_dealer_dist` (Expectimax).

## Reading the output

- **Next-card probabilities** show each rank's count out of the cards left.
- **Expected value** (algorithms 4 and 6) is the average payoff per unit bet,
  so -0.24 means you lose about 24 cents per unit bet on average.
- The pipeline prints stage 1 values and, if it ran, the stage 2 Monte Carlo values.

## Tests

From the project root:

```
python tests/tests.py
```

or `python -m unittest discover -s tests -p "tests.py" -v`.

There are 50 tests, 10 each for basic strategy, probabilities, Monte Carlo,
Expectimax and the pipeline. They cover special cases such as when to split,
when doubling is not allowed (three cards or `can_double=False`), 21, an empty
or skewed deck, and reproducibility with a fixed random seed.

## Known limitations

- Monte Carlo is statistical: close decisions can flip between runs.
- Monte Carlo does not simulate doubling after a split.
- Split values in Expectimax are approximate (see above).
- The shoe is treated as fixed: no reshuffling or card counting across rounds.
- Only one player is modelled.
