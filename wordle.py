"""Solver for Wordle word-guessing game.

# Rules of the game

In the game Wordle, we try to guess a 5-letter word. After each guess, we get
one of the following feedback values for each letter in the guess:

    HERE (h): The letter appears in the word in this exact position.
    NOWHERE (n): The letter does not appear anywhere in the word.
    ELSEWHERE (e): The letter appears in the word, but not in this position.

The ELSEWHERE feedback is computed after "filtering out" all of the HERE
letters. This is best illustrated by example. Consider the guess

    EE

for the word

    EX.

Wordle will give the feedback

    hn.

Notice that the feedback for the second E is NOWHERE even though there is
an E in a different position in the word.

The Wordle word list can be extracted from the website's source code.


# Our strategy

Let W denote the set of words that are consistent with all feedback we have
receieved in all past rounds. We guess based on a one-step (greedy) exhaustive
search to minimize the size of the next round's feasible word list. Precisely,
we select the guess g that minimizes the quantity

    E_{w in W} |{w' in W: feedback(w', g) = feedback(w, g)}|

where feedback(w, g) returns the sequence of {HERE, NOWHERE, ELSEWHERE}
feedback values for the word/guess pair.

Another possible method would be to use the maximum instead of the expectation.

A minimax-optimal method would search the entire game tree and minimize the
number of rounds until completion. However, this is too expensive. Trying a
Monte Carlo tree search could be interesting.

The guess g may be selected from either the full word list ("easy mode") or the
current possible words W ("hard mode"). Note that "easy" and "hard" refer to
the amount of information we can gain from our guess. From a computational
perspective, "hard" mode is a much smaller search problem.

Our greedy-exhaustive strategy leads to high quality guesses, but it is too
slow for the initial guess where W is the entire word list. Therefore, for the
initial guess we instead use a heuristic strategy based on letter frequencies.
See the source code for details.
"""

from collections import Counter
import sys

# Use tqdm for search progress bar if available.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


ALL_WORDS = list(w.strip() for w in open("dictionary.txt"))

# Used like an enum internally, but also user-facing when the user is encoding
# feedback from the Wordle website on the command line.
NOWHERE = "n"
ELSEWHERE = "e"
HERE = "h"


# Common subroutine for recursive handling of HERE matches.
def cut(s, i):
    return s[:i] + s[i+1:]


# Returns the subset of `words` that are consistent with the guess feedback.
def prune(words, guess, feedback):

    # Recursion is the easiest way to deal with the effect of HERE matches on
    # ELSEWHERE vs. NOWHERE decisions for other letters (see docstring for
    # details).
    for i, (letter, result) in enumerate(zip(guess, feedback)):
        if result == HERE:
            consistent = [w for w in words if w[i] == letter]
            reduced = [cut(w, i) for w in consistent]
            pruned = prune(reduced, cut(guess, i), cut(feedback, i))
            restored = [w[:i] + letter + w[i:] for w in pruned]
            return restored

    # If we get here, there are no HERE matches in the feedback.
    for i, (letter, result) in enumerate(zip(guess, feedback)):
        assert result != HERE
        if result == NOWHERE:
            words = [w for w in words if letter not in w]
        if result == ELSEWHERE:
            words = [w for w in words if letter in w and w[i] != letter]
    return words


# Implements the exhaustive search guessing method described in the docstring.
def guess_greedy(words, guesses):
    N = len(words)
    minval = N * N
    argmin = None
    for guess in tqdm(guesses):
        s = 0
        for w in words:
            feedback = give_feedback(w, guess)
            s += len(prune(words, guess, feedback))
            if s > minval:
                # Early exit - cannot be argmin. Significant performance boost.
                break
        if s < minval:
            minval = s
            argmin = guess
    message = f"expected size after pruning: {minval/N:.1f}."
    return argmin, message


def give_feedback(word, guess):
    out = []
    for i, (wl, gl) in enumerate(zip(word, guess)):
        if gl == wl:
            rest = give_feedback(cut(word, i), cut(guess, i))
            return rest[:i] + [HERE] + rest[i:]
        elif gl in word:
            out.append(ELSEWHERE)
        else:
            out.append(NOWHERE)
    return out


def manual_feedback(guess):
    return input(
        f"enter feedback code:\n"
        f"{HERE} = here, {ELSEWHERE} = elsewhere, {NOWHERE} = nowhere.\n"
    )


def loop(words, guesser, feedbacker, hard=False):
    # The exhaustive search is too slow for generating an initial guess.
    # Instead, we find a word whose unique letters collectively appear most
    # often in the word list. The idea is that any NOWHERE feedback from this
    # guess will help us eliminate a lot of words.
    letter_counts = Counter("".join("".join(set(w)) for w in words))
    def coverage(word):
        return sum(letter_counts[l] for l in set(word))
    g = max(words, key=coverage)
    msg = "heuristic initial guess."

    N = 10
    for i in range(N):
        print(f"guess: {g}\n{msg}")
        f = feedbacker(g)
        words = prune(words, g, f)
        if len(words) == 0:
            raise ValueError("All possibilities eliminated!")
        if len(words) == 1:
            print(f"solution after {i+1} guesses:", words[0])
            return
        print(f"after pruning, {len(words)} possibilities:")
        print(words)
        guesses = words if hard else ALL_WORDS
        g, msg = guesser(words, guesses)

    print(f"giving up after {N} tries. possibilities:")
    print(words)


def main():
    if len(sys.argv) > 1:
        # Test on known word.
        word = sys.argv[1]
        feedbacker = lambda guess: give_feedback(word, guess)
    else:
        # Manual input for interacting with Wordle site.
        feedbacker = manual_feedback
    loop(ALL_WORDS, guess_greedy, feedbacker)


if __name__ == "__main__":
    main()
