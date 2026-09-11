"""Track the cardboard box -> box_track.json

Pyramidal Lucas-Kanade on features seeded inside the box: median flow per step,
forward/backward consistency check, reseeding when features drop off.

Runs at whatever resolution the video is (the seed box is authored at 720p and
scaled by K = width/1280). Memory is bounded by the tracked window, not the
video: frames from WIN_T0 up to the seed are buffered so the backward pass can
walk them in reverse, then the forward pass streams with one frame in hand.
"""
import cv2, numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = HERE + "/"
VID = os.environ.get("SWEEP_VIDEO", os.path.join(HERE, "20260811_174657.mp4"))

WIN_T0, WIN_T1 = 57.0, 69.0      # tracked window; the figure needs 58.2 - 66.5
SEED_T = 64.0                    # hand-placed box, verified against the frame
SEED_BOX_720 = (500, 285, 185, 215)

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
K = W / 1280.0
x, y, w, h = [v * K for v in SEED_BOX_720]
f_lo, f_seed, f_hi = (int(round(t * fps)) for t in (WIN_T0, SEED_T, WIN_T1))
print(f"res {W}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  scale {K}  "
      f"frames {f_lo}..{f_hi}  seed {f_seed}")

lk = dict(winSize=(int(31 * K) | 1, int(31 * K) | 1), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
MIN_DIST = max(3, int(6 * K))

def seed_points(img, cx, cy):
    m = np.zeros_like(img)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    m[max(0, y0):int(y0 + h), max(0, x0):int(x0 + w)] = 255
    return cv2.goodFeaturesToTrack(img, 300, 0.01, MIN_DIST, mask=m)

def step(prev, nxt, p, cx, cy):
    """One LK step. Returns (points, cx, cy) or None if the track is lost."""
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p, None, **lk)
    p0r, _, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, p1, None, **lk)
    good = (np.linalg.norm(p - p0r, axis=2).ravel() < 1.0 * K) & (st.ravel() == 1)
    if good.sum() < 8:
        return None
    d = np.median((p1[good] - p[good]).reshape(-1, 2), axis=0)
    cx, cy = cx + d[0], cy + d[1]
    p = p1[good].reshape(-1, 1, 2)
    if len(p) < 25:                                   # reseed in the new window
        pn = seed_points(nxt, cx, cy)
        if pn is not None:
            p = pn
    return p, cx, cy

# ---- read the window: buffer up to the seed, then stream ----
cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
buf, track = [], {}
for i in range(f_lo, f_hi + 1):
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if i <= f_seed:
        buf.append((i, g))
    else:
        if i == f_seed + 1:                           # switch to streaming
            prev_i, prev_g = buf[-1]
            cx, cy = x + w / 2, y + h / 2
            p = seed_points(prev_g, cx, cy)
        r = step(prev_g, g, p, cx, cy)
        if r is None:
            print(f"forward track lost at frame {i}")
            break
        p, cx, cy = r
        track[i] = (cx, cy)
        prev_g = g
print(f"buffered {len(buf)} frames ({(len(buf) * buf[0][1].nbytes) >> 20} MB), "
      f"streamed {len(track)}")

# ---- backward pass over the buffer ----
cx, cy = x + w / 2, y + h / 2
track[f_seed] = (cx, cy)
p = seed_points(buf[-1][1], cx, cy)
for k in range(len(buf) - 1, 0, -1):
    r = step(buf[k][1], buf[k - 1][1], p, cx, cy)
    if r is None:
        print(f"backward track lost at frame {buf[k - 1][0]}")
        break
    p, cx, cy = r
    track[buf[k - 1][0]] = (cx, cy)

data = [[int(i), float(track[i][0]), float(track[i][1])] for i in sorted(track)]
json.dump(data, open(S + "box_track.json", "w"))
print(f"tracked {len(data)} frames, {data[0][0]} -> {data[-1][0]}")
for k in range(0, len(data), 30):
    i, cx, cy = data[k]
    print(f"t={i / fps:6.2f} x={cx:7.1f} y={cy:7.1f}")
