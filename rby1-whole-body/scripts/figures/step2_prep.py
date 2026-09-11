"""Background plate + sharpness table -> plate.png, sharp.npy

The plate is a temporal median of the scene from 44 s on (the camera is static
throughout), used as the empty-room background the ghosts are drawn over. The
sharpness table is the Laplacian variance around the box for every frame of the
transport window, so the compositor can pick the crispest frame per pose.
"""
import cv2, numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = HERE + "/"
VID = os.environ.get("SWEEP_VIDEO", os.path.join(HERE, "20260811_174657.mp4"))

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
K = W / 1280.0
trk = {i: (x, y) for i, x, y in json.load(open(S + "box_track.json"))}
print("res", W, H, "fps", fps, "N", N, "scale", K)

# ---- median background plate, in row chunks to bound memory ----
idx = set(range(int(44 * fps), N, 6))
buf = []
cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx)); i = min(idx)
while True:
    ok, f = cap.read()
    if not ok: break
    if i in idx: buf.append(f)
    i += 1
print("plate frames", len(buf))
plate = np.zeros((H, W, 3), np.uint8)
CH = 90
for r in range(0, H, CH):
    sl = np.array([b[r:r + CH] for b in buf], np.uint8)
    plate[r:r + CH] = np.median(sl, axis=0).astype(np.uint8)
del buf
cv2.imwrite(S + "plate.png", plate)

# ---- per-frame sharpness over the transport window ----
f0, f1 = int(58.2 * fps), int(66.5 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
out = []
R = int(200 * K)
for i in range(f0, f1 + 1):
    ok, fr = cap.read()
    if not ok: break
    cx, cy = trk[i]
    x0, x1 = int(max(0, cx - R)), int(min(W, cx + R))
    y0, y1 = int(max(0, cy - R)), int(min(H, cy + R))
    g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    out.append((i, cv2.Laplacian(g, cv2.CV_64F).var()))
np.save(S + "sharp.npy", np.array(out))
print("sharpness frames", len(out))
