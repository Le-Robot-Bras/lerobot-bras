"""Compare a dance_live log with librosa offline beats."""
import sys, numpy as np
log = np.load(sys.argv[1]); bt = np.load(sys.argv[2])
t, br, bpm, conf, sec = log["t"], log["beat_raw"], log["bpm"], log["conf"], log["section"]
secs = list(log["sections"]); moves = list(log["moves"])
# tempo convergence: first time after which bpm stays within 3% of 96 (offline 95.7)
ok = np.abs(bpm / 95.7 - 1) < 0.03
bad = np.where(~ok)[0]
tc = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else (t[0] if not len(bad) else None)
print(f"tempo within 3% of 95.7 for good from t = {tc}")
for T in (4, 6, 8, 10, 15, 20, 40, 80, 120, 160, 200):
    i = np.searchsorted(t, T)
    if i < len(t): print(f"  t={T:4d}s bpm {bpm[i]:6.2f} conf {conf[i]:.2f}")
spb = 60 / 95.7
tstart = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
sel = bt[(bt > tstart) & (bt < t[-1])]
b = np.interp(sel, t, br)
ph = b - np.round(b)
err = ph * spb * 1000
print(f"phase error vs librosa ({len(sel)} beats after {tstart}s): mean {np.mean(np.abs(err)):.1f} ms, median {np.median(np.abs(err)):.1f} ms, signed mean {np.mean(err):+.1f} ms, p90 {np.percentile(np.abs(err),90):.1f}")
# sections over time
cur = None
for i in range(0, len(t)):
    if sec[i] != cur:
        cur = sec[i]; print(f"  {int(t[i]//60)}:{t[i]%60:04.1f} {secs[cur]}")
q = log["q"]; v = np.abs(np.diff(q, axis=0)) / 0.02
print("max joint speed rad/s:", dict(zip(log["joints"], v.max(0).round(2))))
from collections import Counter
print("moves:", Counter(moves[m] for m in log["move"]))
for k in ("bass","mid","high","energy"):
    x = log[k]; print(k, "mean %.2f std %.2f min %.2f max %.2f" % (x.mean(), x.std(), x.min(), x.max()))
