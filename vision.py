"""
vision.py - read the six faces of a cube from a webcam and turn them into a facelet string.

Scanning protocol (cube stays in your hand, TOP face always pointing up unless noted):
    F : show the face you call FRONT to the camera
    R : turn the cube so the face on the RIGHT side of the image comes to the front
    B : turn it again the same way
    L : turn it again the same way
    U : go back to F, then tilt the cube so its TOP face points at the camera
        (the old front face ends up at the BOTTOM of the image)
    D : go back to F, then tilt so the BOTTOM face points at the camera
        (the old front face ends up at the TOP of the image)

Colour recognition adapts to your lighting: the 6 centre stickers are the colour
references, every other sticker is matched to the nearest reference in a lighting-robust LAB feature space, and
a Hungarian assignment forces exactly 9 stickers per colour.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cube import FACES

SCAN_ORDER = ["F", "R", "B", "L", "U", "D"]
PROMPTS = {
    "F": "FRONT: show any face to the camera, keep the TOP face you choose pointing up",
    "R": "RIGHT: turn the cube so the face on the right side of the image comes forward",
    "B": "BACK: turn the cube the same way again",
    "L": "LEFT: turn the cube the same way again",
    "U": "TOP: back to FRONT, then tilt so the TOP faces the camera (old front at bottom)",
    "D": "BOTTOM: back to FRONT, then tilt so the BOTTOM faces the camera (old front at top)",
}
ROI_SIZE = 270          # pixels of the 3x3 scanning grid
PATCH = 10              # half-size of the sampled patch in each sticker


def roi_origin(frame, size=ROI_SIZE):
    h, w = frame.shape[:2]
    return (w - size) // 2, (h - size) // 2


def sample_face(frame, x0, y0, size=ROI_SIZE, patch=PATCH):
    """Median BGR colour of the centre patch of each of the 9 stickers -> (9, 3)."""
    cell = size // 3
    out = []
    for r in range(3):
        for c in range(3):
            cx, cy = x0 + c * cell + cell // 2, y0 + r * cell + cell // 2
            p = frame[cy - patch:cy + patch, cx - patch:cx + patch]
            out.append(np.median(p.reshape(-1, 3), axis=0))
    return np.array(out, dtype=np.float32)


def bgr_to_lab(bgr_array):
    a = (np.asarray(bgr_array, dtype=np.float32) / 255.0).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def color_features(bgr_array):
    """Lighting-robust colour features: LAB chromaticity divided by lightness, plus a
    little lightness. Dim orange no longer looks like bright red."""
    lab = bgr_to_lab(bgr_array)
    L = np.maximum(lab[:, 0:1], 20.0)
    return np.hstack([0.3 * lab[:, 0:1], lab[:, 1:2] / L * 60.0, lab[:, 2:3] / L * 60.0])


def classify(samples):
    """
    samples: dict face letter -> (9, 3) BGR array (the centre of each is index 4).
    Returns (state_string, palette) where palette maps face letter -> BGR of its centre.
    """
    bgr = np.vstack([samples[f] for f in FACES]).astype(np.float32)      # Kociemba order
    lab = color_features(bgr)
    centres = lab[[i * 9 + 4 for i in range(6)]]
    dist = np.linalg.norm(lab[:, None, :] - centres[None, :, :], axis=2)  # (54, 6)
    for fi in range(6):                                                   # centres are known
        dist[fi * 9 + 4, :] = 1e4
        dist[fi * 9 + 4, fi] = 0
    cost = np.repeat(dist, 9, axis=1)                                     # 9 slots per colour
    rows, cols = linear_sum_assignment(cost)
    labels = np.empty(54, dtype=int)
    labels[rows] = cols // 9
    state = "".join(FACES[l] for l in labels)
    palette = {f: tuple(int(v) for v in samples[f][4]) for f in FACES}
    return state, palette


def _draw_mini(img, colors, x, y, cell=12):
    for i, col in enumerate(colors):
        r, c = divmod(i, 3)
        cv2.rectangle(img, (x + c * cell, y + r * cell), (x + (c + 1) * cell - 1, y + (r + 1) * cell - 1),
                      tuple(int(v) for v in col), -1)


def scan_cube(cam_index=0):
    """Interactive scan. SPACE = capture, B = back one face, ESC = quit. Returns samples dict or None."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")
    # Optional: lock exposure / white balance so colours don't drift (ignored if unsupported)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    samples, i = {}, 0
    while i < 6:
        face = SCAN_ORDER[i]
        ok, frame = cap.read()
        if not ok:
            continue
        x0, y0 = roi_origin(frame)
        live = sample_face(frame, x0, y0)
        view = frame.copy()
        cell = ROI_SIZE // 3
        for r in range(3):
            for c in range(3):
                cv2.rectangle(view, (x0 + c * cell, y0 + r * cell),
                              (x0 + (c + 1) * cell, y0 + (r + 1) * cell), (255, 255, 255), 1)
                cx, cy = x0 + c * cell + cell // 2, y0 + r * cell + cell // 2
                cv2.circle(view, (cx, cy), 14, (0, 0, 0), 2)                       # what we read
                cv2.circle(view, (cx, cy), 12, tuple(int(v) for v in live[r * 3 + c]), -1)
        cv2.putText(view, f"Face {i + 1}/6  [{face}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(view, PROMPTS[face], (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(view, "SPACE capture   B back   ESC quit", (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for j, f in enumerate(SCAN_ORDER):                                         # captured thumbnails
            if f in samples:
                _draw_mini(view, samples[f], 10 + j * 44, 75, cell=12)
        cv2.imshow("Scan cube", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release(); cv2.destroyAllWindows()
            return None
        if k in (ord("b"), ord("B")) and i > 0:
            i -= 1
            samples.pop(SCAN_ORDER[i], None)
        elif k == ord(" "):
            acc = []
            for _ in range(8):                                                     # average a few frames
                ok, f = cap.read()
                if ok:
                    acc.append(sample_face(f, x0, y0))
            if acc:
                samples[face] = np.median(acc, axis=0)
                i += 1
    cap.release()
    cv2.destroyWindow("Scan cube")
    return samples
