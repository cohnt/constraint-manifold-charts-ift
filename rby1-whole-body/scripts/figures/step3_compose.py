import cv2, numpy as np, json, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
S = HERE + "/"
VID = os.environ.get("SWEEP_VIDEO", os.path.join(HERE, "20260811_174657.mp4"))

cap = cv2.VideoCapture(VID); fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
K = W / 1280.0                                   # everything below is authored at 720p scale
plate = cv2.imread(S + "plate.png")
trk = {i: (x, y) for i, x, y in json.load(open(S + "box_track.json"))}
sharp = dict(np.load(S + "sharp.npy"))
odd = lambda v: int(round(v * K)) | 1
px = lambda v: int(round(v * K))

T0, T1 = 58.2, 66.5
f0, f1 = int(T0 * fps), int(T1 * fps)
NG      = int(sys.argv[1]);   OUT     = sys.argv[2]
ARM_LO  = float(sys.argv[3]); ARM_HI  = float(sys.argv[4])
BOX_LO  = float(sys.argv[5]); BOX_HI  = float(sys.argv[6])
FIN_ARM = float(sys.argv[7]); FIN_BOX = float(sys.argv[8])
DBG     = len(sys.argv) > 9

frames = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
for i in range(f0, f1 + 1):
    ok, fr = cap.read()
    if not ok: break
    frames[i] = fr

idx = sorted(frames)
P = np.array([trk[i] for i in idx])
dcum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
picks = []
for t in np.linspace(0, dcum[-1], NG):
    j = int(np.argmin(np.abs(dcum - t)))
    lo, hi = max(0, j - 3), min(len(idx) - 1, j + 3)
    j = max(range(lo, hi + 1), key=lambda c: sharp.get(idx[c], 0))
    if not picks or idx[j] != picks[-1]: picks.append(idx[j])
print("poses:", [round(i / fps, 2) for i in picks])

def fg_mask(img, cx, cy):
    d = cv2.absdiff(img.astype(np.int16), plate.astype(np.int16)).astype(np.uint8)
    g = cv2.GaussianBlur(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), (odd(5), odd(5)), 0)
    m = ((g > 20).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((px(15), px(15)), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((px(5), px(5)), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 1200 * K * K or x > px(1150): continue
        if abs(cent[k][0] - cx) > px(760) or abs(cent[k][1] - cy) > px(520): continue
        keep[lab == k] = 255
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((px(25), px(25)), np.uint8))

def cardboard(img, cx, cy, fg):
    """the box itself, by colour - the white/silver wrist and gripper stay with the arm"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (odd(5), odd(5)), 0),
                       cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Sa, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = ((Hh >= 7) & (Hh <= 20) & (Sa >= 85) & (Sa <= 185) &
         (V >= 60) & (V <= 180)).astype(np.uint8) * 255
    loc = np.zeros_like(c)
    cv2.ellipse(loc, (int(cx), int(cy)), (px(190), px(180)), 0, 0, 360, 255, -1)
    m = cv2.bitwise_and(cv2.bitwise_and(c, loc), fg)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((px(11), px(11)), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((px(5), px(5)), np.uint8))

# --- paint the base-mounted lidar module out at source ---------------------
# it is rigidly mounted and lands on the box trail; where it is visible we fill
# its footprint with the surrounding base colour before any masking happens
LID = dict(c=(px(781), px(403)), ax=(px(19), px(18)), ang=0)
_tf = frames[max(frames)] if frames else None
_TX0, _TY0, _TX1, _TY1 = px(765), px(388), px(799), px(420)   # sensor only
_tmpl = cv2.cvtColor(_tf[_TY0:_TY1, _TX0:_TX1], cv2.COLOR_BGR2GRAY).astype(np.float32)

def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

lid_m = np.zeros((H, W), np.float32)
cv2.ellipse(lid_m, LID["c"], LID["ax"], LID["ang"], 0, 360, 1.0, -1)
lid_m = cv2.GaussianBlur(lid_m, (odd(11), odd(11)), 0)[..., None]
_ring = np.zeros((H, W), np.uint8)
cv2.ellipse(_ring, LID["c"], (LID["ax"][0] * 3, LID["ax"][1] * 3), LID["ang"], 0, 360, 255, -1)
cv2.ellipse(_ring, LID["c"], (LID["ax"][0] + px(6), LID["ax"][1] + px(6)), LID["ang"], 0, 360, 0, -1)

def hide_lidar(img, force=False):
    g = cv2.cvtColor(img[_TY0:_TY1, _TX0:_TX1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not force and _ncc(g, _tmpl) < 0.40:
        return img                                   # occluded here - nothing to hide
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sel = (_ring > 0) & (hsv[..., 2] < 90)
    col = np.median(img[sel], axis=0) if sel.sum() > 500 else np.array([39., 35., 37.])
    return (img * (1 - lid_m) + col[None, None, :] * lid_m).astype(np.uint8)

plate = hide_lidar(plate, force=True)
frames = {i: hide_lidar(f) for i, f in frames.items()}

def crisp(img):
    b = cv2.GaussianBlur(img, (0, 0), 1.6 * K)
    return np.clip(cv2.addWeighted(img, 1.45, b, -0.45, 0), 0, 255)

soft = lambda m: cv2.GaussianBlur(m, (odd(7), odd(7)), 0).astype(np.float32) / 255.0
layers = []
for n, i in enumerate(picks):
    cx, cy = trk[i]
    fg = fg_mask(frames[i], cx, cy)
    bx = cardboard(frames[i], cx, cy, fg)
    arm = cv2.bitwise_and(fg, cv2.bitwise_not(bx))
    if DBG:
        v = frames[i].copy(); v[bx > 0] = (0, 0, 255)
        cv2.imwrite(S + f"hd_boxmask_{n}.png", v[::2, ::2])
    layers.append((crisp(frames[i].astype(np.float32)), soft(arm), soft(bx)))

last = len(layers) - 1
canvas = plate.astype(np.float32)
for k, (img, marm, mbox) in enumerate(layers):
    frac = k / last
    a_arm = ARM_LO + (ARM_HI - ARM_LO) * frac ** 1.2
    a_box = BOX_LO + (BOX_HI - BOX_LO) * frac ** 1.2
    if k == last: a_arm, a_box = FIN_ARM, FIN_BOX
    A = np.clip(marm * a_arm + mbox * a_box, 0, 1)[..., None]
    canvas = canvas * (1 - A) + img * A

union = np.max([l[1] + l[2] for l in layers], axis=0); union[:, px(1150):] = 0
ys, xs = np.where(union > 0.15)
x0, x1c = max(0, xs.min() - px(46)), min(W, xs.max() + px(24))
y0, y1c = max(0, ys.min() - px(22)), min(H, ys.max() + px(24))
out = np.clip(canvas, 0, 255).astype(np.uint8)
ramp = np.zeros((H, W), np.float32)
ramp[:, px(1195):] = 1.0
ramp[:, px(1155):px(1195)] = np.linspace(0, 1, px(1195) - px(1155))[None, :]
out = np.clip(out * (1 - ramp[..., None]) + plate.astype(np.float32) * ramp[..., None],
              0, 255).astype(np.uint8)[y0:y1c, x0:x1c]
cv2.imwrite(S + OUT, out)
print("wrote", OUT, out.shape[1], "x", out.shape[0])
