"""
search.py - numpy-only helpers for the ML solver:
  * gen_batch : training data from random scrambles (scramble a solved cube, label
                each state with the INVERSE of the last scramble move)
  * beam_search : policy-guided beam search that works with any policy function
"""
import numpy as np
from cube import SOLVED, FACES, PERMS, FACE_OF_MOVE, INVERSE_MOVE, MOVE_NAMES

COLOR_IDX = {c: i for i, c in enumerate(FACES)}
SOLVED_ARR = np.array([COLOR_IDX[c] for c in SOLVED], dtype=np.int8)


def to_array(state_str):
    return np.array([COLOR_IDX[c] for c in state_str], dtype=np.int8)


def gen_batch(n_scrambles, max_depth, rng):
    """Returns X (n*max_depth, 54) int8 states and y (n*max_depth,) move labels.
    Each scramble contributes one training sample per depth 1..max_depth."""
    states = np.tile(SOLVED_ARR, (n_scrambles, 1))
    prev_face = np.full(n_scrambles, -1)
    X, y = [], []
    for _ in range(max_depth):
        mv = rng.integers(0, 18, n_scrambles)
        bad = FACE_OF_MOVE[mv] == prev_face            # never turn the same face twice in a row
        while bad.any():
            mv[bad] = rng.integers(0, 18, int(bad.sum()))
            bad = FACE_OF_MOVE[mv] == prev_face
        states = np.take_along_axis(states, PERMS[mv], axis=1)
        prev_face = FACE_OF_MOVE[mv]
        X.append(states.copy())
        y.append(INVERSE_MOVE[mv])
    return np.concatenate(X), np.concatenate(y)


def beam_search(start_state, policy_logp, beam_width=1024, max_steps=26):
    """
    start_state : 54-char string
    policy_logp : function (states int8 (W,54)) -> log-probabilities (W,18)
    Returns a list of move names, or None if no solution was found.
    """
    start = to_array(start_state)
    if (start == SOLVED_ARR).all():
        return []
    states = start[None, :]
    scores = np.zeros(1)
    paths = [[]]
    last_face = np.array([-1])

    for _ in range(max_steps):
        logp = policy_logp(states)                                   # (W, 18)
        total = scores[:, None] + logp
        total[last_face[:, None] == FACE_OF_MOVE[None, :]] = -np.inf  # skip same-face repeats
        flat = total.ravel()
        k = min(beam_width * 3, flat.size)                            # over-select, dedupe below
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]
        top = top[np.isfinite(flat[top])]
        parent, mv = top // 18, top % 18

        new_states = np.take_along_axis(states[parent], PERMS[mv], axis=1)

        done = (new_states == SOLVED_ARR).all(axis=1)
        if done.any():
            j = int(np.argmax(done))
            return [MOVE_NAMES[m] for m in paths[parent[j]] + [mv[j]]]

        _, first = np.unique(new_states, axis=0, return_index=True)   # drop duplicate states
        keep = np.sort(first)[:beam_width]                            # order is best-score first
        parent, mv, new_states = parent[keep], mv[keep], new_states[keep]

        paths = [paths[p] + [m] for p, m in zip(parent, mv)]
        scores = flat[top][keep]
        last_face = FACE_OF_MOVE[mv]
        states = new_states
    return None
