"""
solver_ml.py - learned solver: a residual MLP policy network + beam search.

Training data is free: scramble a solved cube and label every state with the inverse
of the last scramble move ("what move undoes the scramble?"). At solve time the
network proposes moves and a beam search keeps the most promising partial solutions.

    python solver_ml.py train --steps 8000            # train  (GPU strongly recommended)
    python solver_ml.py test  --depth 8 --n 50        # solve-rate on random scrambles
    python solver_ml.py solve --scramble "R U F' D2"  # solve one scramble

Expect good results for scrambles up to ~10-12 moves; deeper cubes need a bigger
model / more training (see README), or just fall back to Kociemba.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cube import SOLVED, MOVE_NAMES, apply_moves, random_scramble
from search import gen_batch, beam_search, to_array

MODEL_PATH = "cube_policy.pt"


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f1, self.b1 = nn.Linear(w, w), nn.BatchNorm1d(w)
        self.f2, self.b2 = nn.Linear(w, w), nn.BatchNorm1d(w)

    def forward(self, x):
        h = F.relu(self.b1(self.f1(x)))
        h = self.b2(self.f2(h))
        return F.relu(x + h)


class PolicyNet(nn.Module):
    def __init__(self, width=512, blocks=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(54 * 6, width), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.out = nn.Linear(width, 18)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def encode(states_int8, device):
    t = torch.as_tensor(states_int8, dtype=torch.long, device=device)
    return F.one_hot(t, 6).reshape(len(t), -1).float()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(steps=8000, scrambles=128, max_depth=12, lr=1e-3, width=512, blocks=3, path=MODEL_PATH):
    device = get_device()
    print(f"training on {device}")
    model = PolicyNet(width, blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        X, y = gen_batch(scrambles, max_depth, rng)
        xb = encode(X, device)
        yb = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps:
            acc = (logits.argmax(1) == yb).float().mean().item()
            print(f"step {step:6d}/{steps}  loss {loss.item():.3f}  acc {acc:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "width": width, "blocks": blocks}, path)
    print("saved", path)


def load_model(path=MODEL_PATH):
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    model = PolicyNet(ckpt["width"], ckpt["blocks"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, device


def solve(state, model_or_path=MODEL_PATH, beam_width=1024, max_steps=26):
    """Returns a list of move names or None if the beam search failed."""
    if isinstance(model_or_path, str):
        model, device = load_model(model_or_path)
    else:
        model, device = model_or_path

    @torch.no_grad()
    def policy_logp(S):
        return F.log_softmax(model(encode(S, device)), dim=1).cpu().numpy()

    return beam_search(state, policy_logp, beam_width, max_steps)


def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--scrambles", type=int, default=128)
    t.add_argument("--max-depth", type=int, default=12)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--blocks", type=int, default=3)
    t.add_argument("--model", default=MODEL_PATH)
    e = sub.add_parser("test")
    e.add_argument("--depth", type=int, default=8)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--beam", type=int, default=1024)
    e.add_argument("--model", default=MODEL_PATH)
    s = sub.add_parser("solve")
    s.add_argument("--scramble", required=True)
    s.add_argument("--beam", type=int, default=1024)
    s.add_argument("--model", default=MODEL_PATH)
    a = ap.parse_args()

    if a.cmd == "train":
        train(a.steps, a.scrambles, a.max_depth, width=a.width, blocks=a.blocks, path=a.model)
    elif a.cmd == "test":
        model = load_model(a.model)
        ok, total_len, t0 = 0, 0, time.time()
        for _ in range(a.n):
            st = apply_moves(SOLVED, random_scramble(a.depth))
            sol = solve(st, model, a.beam)
            if sol is not None and apply_moves(st, sol) == SOLVED:
                ok += 1
                total_len += len(sol)
        print(f"depth {a.depth}: solved {ok}/{a.n}  avg length {total_len / max(ok, 1):.1f}  "
              f"{(time.time() - t0) / a.n:.2f}s per cube")
    else:
        st = apply_moves(SOLVED, a.scramble)
        sol = solve(st, a.model, a.beam)
        print("solution:", " ".join(sol) if sol is not None else "not found")


if __name__ == "__main__":
    _cli()
