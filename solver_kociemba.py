"""solver_kociemba.py - optimal-ish (<= ~20 moves) solutions via Kociemba's two-phase algorithm."""


def solve(state):
    """state: 54-char facelet string (URFDLB order). Returns list of move names."""
    import kociemba                                  # pip install kociemba
    try:
        solution = kociemba.solve(state)
    except Exception as e:                           # invalid cube (twist/flip/parity ...)
        raise ValueError(f"Kociemba rejected this cube state: {e}") from e
    if solution.startswith("Error"):
        raise ValueError(f"Kociemba rejected this cube state: {solution}")
    return solution.split()
