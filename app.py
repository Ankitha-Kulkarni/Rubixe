"""
app.py - scan a cube with your webcam, check the colours, solve it, and follow the moves.

    python app.py                              # webcam scan  -> Kociemba solver
    python app.py --solver ml                  # ML solver (falls back to Kociemba if it fails)
    python app.py --scramble "R U F' D2 L"     # no camera: solve a scrambled virtual cube
    python app.py --state <54 letters> --no-gui
"""
import argparse
import sys

import numpy as np

from cube import FACES, SOLVED, apply_move, apply_moves, invert_move, validate

DEFAULT_PALETTE = {                     # BGR, used when no camera scan is available
    "U": (245, 245, 245), "R": (40, 40, 210), "F": (60, 170, 0),
    "D": (30, 215, 250), "L": (20, 120, 250), "B": (200, 90, 0),
}
# net layout (block row, block col): the classic cross
NET_POS = {"U": (0, 1), "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3), "D": (2, 1)}
CELL, MARGIN = 34, 20
FACE_NAME = {"U": "top", "D": "bottom", "L": "left", "R": "right", "F": "front", "B": "back"}


def describe(move):
    f, mod = move[0], move[1:]
    turn = {"": "90 deg clockwise", "'": "90 deg counter-clockwise", "2": "180 deg"}[mod]
    return f"Turn the {FACE_NAME[f]} face {turn}"


# ------------------------------------------------------------------ drawing
def facelet_rect(idx):
    face, i = FACES[idx // 9], idx % 9
    br, bc = NET_POS[face]
    r, c = divmod(i, 3)
    x = MARGIN + (bc * 3 + c) * CELL
    y = MARGIN + (br * 3 + r) * CELL
    return x, y


def hit_test(px, py):
    for idx in range(54):
        x, y = facelet_rect(idx)
        if x <= px < x + CELL and y <= py < y + CELL:
            return idx
    return None


def draw_net(state, palette, lines=(), highlight_face=None):
    import cv2
    w, h = MARGIN * 2 + 12 * CELL, MARGIN * 2 + 9 * CELL + 28 * max(len(lines), 1) + 10
    img = np.full((h, w, 3), 40, np.uint8)
    for idx, ch in enumerate(state):
        x, y = facelet_rect(idx)
        cv2.rectangle(img, (x, y), (x + CELL - 2, y + CELL - 2), palette[ch], -1)
        cv2.rectangle(img, (x, y), (x + CELL - 2, y + CELL - 2), (0, 0, 0), 1)
    for f, (br, bc) in NET_POS.items():
        x, y = MARGIN + bc * 3 * CELL, MARGIN + br * 3 * CELL
        col = (0, 255, 255) if f == highlight_face else (90, 90, 90)
        cv2.rectangle(img, (x, y), (x + 3 * CELL - 2, y + 3 * CELL - 2), col, 3 if f == highlight_face else 1)
    y0 = MARGIN * 2 + 9 * CELL
    for j, (text, color) in enumerate(lines):
        cv2.putText(img, text, (MARGIN, y0 + 22 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return img


# ------------------------------------------------------------------ editor
def edit_state(state, palette):
    """Click a sticker to cycle its colour (left = next, right = previous). ENTER = confirm."""
    import cv2
    from cube import validate
    state = list(state)
    win = "Check colours"

    def on_mouse(ev, x, y, flags, param):
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            idx = hit_test(x, y)
            if idx is not None and idx % 9 != 4:                # centres are fixed
                step = 1 if ev == cv2.EVENT_LBUTTONDOWN else -1
                state[idx] = FACES[(FACES.index(state[idx]) + step) % 6]

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        ok, msg = validate("".join(state))
        lines = [("Click a sticker to fix its colour (right-click = previous colour)", (255, 255, 255)),
                 ("ENTER = solve   ESC = quit", (255, 255, 255)),
                 (msg if not ok else "Cube looks valid", (0, 255, 0) if ok else (0, 0, 255))]
        cv2.imshow(win, draw_net("".join(state), palette, lines))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if k in (10, 13) and ok:
            cv2.destroyWindow(win)
            return "".join(state)


# ------------------------------------------------------------------ solving
def get_solution(state, solver, model_path, beam):
    if solver == "ml":
        try:
            import solver_ml
            sol = solver_ml.solve(state, model_path, beam)
            if sol is not None and apply_moves(state, sol) == SOLVED:
                return sol, "ML policy + beam search"
            print("ML solver found no solution - falling back to Kociemba.")
        except Exception as e:
            print(f"ML solver unavailable ({e}) - falling back to Kociemba.")
    try:
        import solver_kociemba
        return solver_kociemba.solve(state), "Kociemba two-phase"
    except ImportError:
        sys.exit("Missing dependency: run  pip install kociemba")


def follow_solution(state, moves, palette):
    """SPACE / N = next move, P / BACKSPACE = previous, ESC = quit."""
    import cv2
    win, i, cur = "Solve", 0, state
    while True:
        if i < len(moves):
            nxt = moves[i]
            lines = [(f"Move {i + 1}/{len(moves)}:  {nxt}   ({describe(nxt)})", (0, 255, 255)),
                     ("Hold the cube as scanned: scanned-top up, scanned-front towards you", (200, 200, 200)),
                     ("SPACE next   P previous   ESC quit", (255, 255, 255))]
            hl = nxt[0]
        else:
            lines = [("SOLVED!" if cur == SOLVED else "Done", (0, 255, 0)), ("ESC to quit", (255, 255, 255))]
            hl = None
        cv2.imshow(win, draw_net(cur, palette, lines, hl))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k in (ord(" "), ord("n")) and i < len(moves):
            cur = apply_move(cur, moves[i]); i += 1
        elif k in (ord("p"), 8) and i > 0:
            i -= 1; cur = apply_move(cur, invert_move(moves[i]))
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["kociemba", "ml"], default="kociemba")
    ap.add_argument("--model", default="cube_policy.pt")
    ap.add_argument("--beam", type=int, default=1024)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--state", help="54-letter facelet string (URFDLB order), skips the camera")
    ap.add_argument("--scramble", help='e.g. "R U F\' D2", skips the camera')
    ap.add_argument("--no-gui", action="store_true", help="print the solution instead of opening windows")
    a = ap.parse_args()

    palette = DEFAULT_PALETTE
    if a.state:
        state = a.state.strip().upper()
    elif a.scramble:
        state = apply_moves(SOLVED, a.scramble)
    else:
        import vision
        samples = vision.scan_cube(a.camera)
        if samples is None:
            return
        state, palette = vision.classify(samples)
        print("Scanned state:", state)

    ok, msg = validate(state)
    scanned = not (a.state or a.scramble)
    if not a.no_gui and (scanned or not ok):          # let the user verify / fix stickers
        state = edit_state(state, palette)
        ok, msg = validate(state)
    if not ok:
        sys.exit(f"Invalid cube: {msg}")

    try:
        moves, name = get_solution(state, a.solver, a.model, a.beam)
    except ValueError as e:
        sys.exit(f"Could not solve: {e}\nTip: a single mis-read sticker usually causes this - fix it in the editor.")
    print(f"{name}: {len(moves)} moves\n{' '.join(moves)}")
    assert apply_moves(state, moves) == SOLVED
    if not a.no_gui:
        follow_solution(state, moves, palette)


if __name__ == "__main__":
    main()
