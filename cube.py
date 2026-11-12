"""
cube.py - Rubik's cube model.

State = 54-character string of face letters in Kociemba order:
    U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9
Each character is the face that sticker's colour belongs to (U,R,F,D,L,B),
i.e. colours are named after the centre sticker of each face.

Move permutations are generated from 3D geometry, so there are no
hand-typed tables to get wrong.
"""
import random
import numpy as np

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)

# Outward normals: x = right, y = up, z = towards the viewer (front)
NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


def _facelet_pos(face, i):
    """3D position of facelet i (0..8, row-major as seen when looking at the
    face from outside, in the standard Kociemba net orientation)."""
    r, c = divmod(i, 3)
    if face == "U": return (c - 1, 1, r - 1)      # back row on top
    if face == "R": return (1, 1 - r, 1 - c)      # front column on left
    if face == "F": return (c - 1, 1 - r, 1)
    if face == "D": return (c - 1, -1, 1 - r)     # front row on top
    if face == "L": return (-1, 1 - r, c - 1)     # back column on left
    if face == "B": return (1 - c, 1 - r, -1)     # right column on left
    raise ValueError(face)


_STICKERS = []   # index -> (position, normal)
_INDEX = {}      # (position, normal) -> index
for _fi, _f in enumerate(FACES):
    for _i in range(9):
        key = (_facelet_pos(_f, _i), NORMALS[_f])
        _INDEX[key] = _fi * 9 + _i
        _STICKERS.append(key)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rot_cw(v, n):
    """Rotate v by 90 degrees clockwise when looking at the face with outward
    normal n from outside: v' = -(n x v) + n (n . v)."""
    c = _cross(n, v)
    d = _dot(n, v)
    return (-c[0] + n[0] * d, -c[1] + n[1] * d, -c[2] + n[2] * d)


def _quarter_perm(face):
    """P such that new_state[b] = old_state[P[b]] for a clockwise quarter turn."""
    n = NORMALS[face]
    P = list(range(54))
    for idx, (pos, nor) in enumerate(_STICKERS):
        if _dot(pos, n) == 1:                      # sticker is in the turning layer
            b = _INDEX[(_rot_cw(pos, n), _rot_cw(nor, n))]
            P[b] = idx
    return P


MOVE_NAMES = [f + s for f in FACES for s in ("", "'", "2")]   # 18 moves
MOVE_INDEX = {m: i for i, m in enumerate(MOVE_NAMES)}


def _build_perms():
    perms = []
    for f in FACES:
        p1 = np.array(_quarter_perm(f))
        p2 = p1[p1]
        p3 = p2[p1]
        perms += [p1, p3, p2]        # f, f', f2
    return np.array(perms, dtype=np.int64)


PERMS = _build_perms()                                   # (18, 54)
FACE_OF_MOVE = np.repeat(np.arange(6), 3)                # move idx -> face idx
INVERSE_MOVE = np.array([MOVE_INDEX[invert] for invert in
                         [m[0] + {"": "'", "'": "", "2": "2"}[m[1:]] for m in MOVE_NAMES]])


def apply_move(state, move):
    p = PERMS[MOVE_INDEX[move]]
    return "".join(state[i] for i in p)


def apply_moves(state, moves):
    """moves: list of move names or a space separated string."""
    if isinstance(moves, str):
        moves = moves.split()
    for m in moves:
        state = apply_move(state, m)
    return state


def invert_move(move):
    return move[0] + {"": "'", "'": "", "2": "2"}[move[1:]]


def invert_moves(moves):
    if isinstance(moves, str):
        moves = moves.split()
    return [invert_move(m) for m in reversed(moves)]


def is_solved(state):
    return state == SOLVED


def random_scramble(n, rng=None):
    rng = rng or random
    out, last = [], None
    while len(out) < n:
        m = rng.choice(MOVE_NAMES)
        if m[0] == last:
            continue
        out.append(m)
        last = m[0]
    return out


# ---------------------------------------------------------------- validation
def _cubies():
    groups = {}
    for idx, (pos, _) in enumerate(_STICKERS):
        groups.setdefault(pos, []).append(idx)
    return [v for v in groups.values() if len(v) >= 2]      # 8 corners + 12 edges


_CUBIES = _cubies()


def validate(state):
    """Cheap sanity checks. Returns (ok, message).
    Twist / flip / parity errors are caught later by the Kociemba solver."""
    if len(state) != 54 or any(c not in FACES for c in state):
        return False, "State must be 54 characters from URFDLB."
    for f in FACES:
        if state.count(f) != 9:
            return False, f"Colour {f} appears {state.count(f)} times (needs 9)."
    centres = [state[i * 9 + 4] for i in range(6)]
    if sorted(centres) != sorted(FACES):
        return False, "The six centre stickers must all be different colours."
    solved_sets = sorted(tuple(sorted(SOLVED[i] for i in c)) for c in _CUBIES)
    state_sets = sorted(tuple(sorted(state[i] for i in c)) for c in _CUBIES)
    if solved_sets != state_sets:
        return False, "Some corner/edge piece has an impossible colour combination."
    return True, "OK"
