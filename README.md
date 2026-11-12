# Rubik's Cube Solver: webcam scan + Kociemba + learned (ML) solver

## Setup
```
pip install -r requirements.txt        # torch is only needed for the ML solver
```

## Run
```
python app.py                          # scan with webcam -> verify colours -> solve -> follow moves
python app.py --solver ml              # use the neural solver (falls back to Kociemba on failure)
python app.py --scramble "R U F' D2"   # no camera: test with a virtual cube
python app.py --state <54 letters> --no-gui
```

## Train the ML solver (optional)
```
python solver_ml.py train --steps 8000          # saves cube_policy.pt (GPU / Colab recommended)
python solver_ml.py test --depth 8 --n 50       # solve rate on random scrambles
python solver_ml.py solve --scramble "R U F' D2"
```
Tip: train with `--max-depth 8` first and check `test --depth 6`, then raise it (14-20).

## Scanning protocol (6 faces, SPACE to capture, B to go back)
1. **F**: show any face to the camera, the face you want as TOP pointing up.
2. **R / B / L**: turn the cube the same way each time so the face on the right side of the image comes forward.
3. **U**: return to F, tilt so the TOP face points at the camera (old front at the bottom of the image).
4. **D**: return to F, tilt so the BOTTOM face points at the camera (old front at the top of the image).
Then check the colour net, click wrong stickers to fix them, press ENTER.

## Files
| file | role |
|---|---|
| `cube.py` | cube model (54 facelets, 18 moves from 3D geometry), validation |
| `vision.py` | webcam scan, lighting-robust colour classification (centre references + Hungarian 9-per-colour assignment) |
| `solver_kociemba.py` | optimal-ish solver wrapper |
| `search.py` | training-data generator + policy-guided beam search (numpy only) |
| `solver_ml.py` | residual MLP policy network, training, CLI |
| `app.py` | UI: scan, colour editor, step-by-step solution |

## Notes
- Good scanning conditions matter most: even, diffuse light, no glare, matte stickers.
- "Cube rejected by Kociemba" almost always means one sticker was mis-read: fix it in the editor.
- Ideas to extend the ML solver: add a value head (distance-to-solved) and use it as an A* / beam heuristic (DeepCubeA), train deeper, or add MCTS.
