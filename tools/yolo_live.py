#!/usr/bin/env python3
"""yolo_live.py - live view of a YOLO training, in the browser (auto-refresh).

    python3 tools/yolo_live.py --run runs/yolo/so101_seg --data datasets/so101_seg \
        --log train.log          # optional: the training stdout, for the batch progress
    # -> http://localhost:8008

Progress bars (whole training + current epoch, with ETA), refreshed every 2 s.
Every new epoch, from what ultralytics writes:
  - curves.png: train/val losses and mask mAP50 / mAP50-95 (results.csv)
  - preds.png:  masks predicted by the CURRENT weights (weights/last.pt) on fixed
                validation images, next to the ground truth: you see the network learn.
Read-only: never touches the training. Needs ultralytics, matplotlib, opencv.
"""
import argparse
import csv
import glob
import http.server
import json
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>YOLO training</title><style>
body{background:#111;color:#ddd;font:14px system-ui;margin:16px} img{max-width:100%;border:1px solid #333}
h1{font-size:18px} .s{color:#8c8} .bar{background:#222;border:1px solid #444;border-radius:4px;height:22px;
position:relative;margin:4px 0 12px;max-width:900px} .fill{background:#2e7d32;height:100%;border-radius:3px;
transition:width .5s} .fill.e{background:#1565c0} .txt{position:absolute;inset:0;text-align:center;line-height:22px}
</style></head><body>
<h1>YOLO-seg training &mdash; <span class="s" id="status">waiting for epoch 1</span></h1>
<div>Training</div><div class="bar"><div class="fill" id="fa" style="width:0"></div><div class="txt" id="ta"></div></div>
<div>Current epoch</div><div class="bar"><div class="fill e" id="fb" style="width:0"></div><div class="txt" id="tb"></div></div>
<img id="curves"><h1>Predictions of the current weights (left: ground truth, right: prediction)</h1>
<img id="preds"><script>
let shown = null;
function hms(s){ if(s==null) return "?"; s=Math.round(s); const h=Math.floor(s/3600), m=Math.floor(s%3600/60);
  return (h? h+"h":"")+String(m).padStart(h?2:1,"0")+"m"+String(s%60).padStart(2,"0")+"s"; }
async function tick(){
  try {
    const st = await (await fetch("status.json?"+Date.now())).json();
    document.getElementById("status").innerHTML = st.status;
    const pa = 100*st.done/st.epochs;
    document.getElementById("fa").style.width = pa+"%";
    document.getElementById("ta").textContent = `epoch ${st.done}/${st.epochs} (${pa.toFixed(0)}%) - `+
      `elapsed ${hms(st.elapsed)} - remaining ~${hms(st.eta)}`;
    const pb = st.batch_total ? 100*st.batch/st.batch_total : 0;
    document.getElementById("fb").style.width = pb+"%";
    document.getElementById("tb").textContent = st.batch_total ?
      `${st.phase} epoch ${st.epoch}: ${st.batch}/${st.batch_total} (${pb.toFixed(0)}%)` : "(pass --log to see it)";
    if (st.plots !== shown) { shown = st.plots;
      for (const id of ["curves","preds"]) document.getElementById(id).src = id+".png?"+st.plots; }
  } catch (e) {}
  setTimeout(tick, 2000);
}
tick();
</script></body></html>"""
TRAIN_RE = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s.*?\s(\d+)/(\d+)\s")
VAL_RE = re.compile(r"^\s*all\s|Class\s+Images.*?\s(\d+)/(\d+)\s")


def log_progress(path):
    """(phase, epoch, batch, n_batches) from the last progress line of the training stdout."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 4000))
            tail = f.read().decode("utf-8", "ignore")
    except OSError:
        return None
    for line in reversed(re.split(r"[\r\n]", tail.replace("\x1b[K", ""))):
        m = TRAIN_RE.match(line)
        if m:
            return "train", int(m[1]), int(m[3]), int(m[4])
        m = VAL_RE.search(line)
        if m and m.lastindex:
            return "val", None, int(m[1]), int(m[2])
    return None


def read_csv(path):
    with open(path) as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rows


def curves(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), facecolor="#111")
    for a in ax:
        a.set_facecolor("#1b1b1b"); a.tick_params(colors="#bbb"); a.grid(color="#333")
        for sp in a.spines.values():
            sp.set_color("#444")
    for key, lab in (("train/seg_loss", "train seg"), ("val/seg_loss", "val seg"),
                     ("train/box_loss", "train box"), ("val/box_loss", "val box")):
        ax[0].plot(ep, [r[key] for r in rows], label=lab)
    ax[0].set_title("losses", color="#ddd"); ax[0].legend()
    for key, lab in (("train/cls_loss", "train cls"), ("val/cls_loss", "val cls")):
        ax[1].plot(ep, [r[key] for r in rows], label=lab)
    ax[1].set_title("classification loss", color="#ddd"); ax[1].legend(); ax[1].set_yscale("log")
    ax[2].plot(ep, [r["metrics/mAP50(M)"] for r in rows], label="mask mAP50")
    ax[2].plot(ep, [r["metrics/mAP50-95(M)"] for r in rows], label="mask mAP50-95")
    ax[2].set_ylim(0.5, 1.0); ax[2].set_title("validation (masks)", color="#ddd"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch", color="#bbb")
    fig.tight_layout(); fig.savefig(out, dpi=80, facecolor=fig.get_facecolor()); plt.close(fig)


def draw_gt(img, label_file):
    out = img.copy()
    H, W = img.shape[:2]
    if Path(label_file).exists():
        for line in open(label_file):
            v = line.split()
            pts = (np.array(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0) if v[0] == "0" else (255, 200, 0), 1)
    return out


def preds(weights, images, out):
    from ultralytics import YOLO
    model = YOLO(str(weights), task="segment")
    tiles = []
    for p in images:
        img = cv2.imread(p)
        gt = draw_gt(img, p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        r = model.predict(img, imgsz=320, conf=0.25, device="cpu", verbose=False)[0]
        pr = img.copy()
        if r.masks is not None:
            over = pr.copy()
            for poly, cls, conf in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                col = (0, 255, 0) if int(cls) == 0 else (255, 200, 0)
                if len(poly) >= 3:
                    cv2.fillPoly(over, [np.round(poly).astype(np.int32)], col)
                    x, y = np.round(poly.min(0)).astype(int)
                    cv2.putText(pr, f"{conf:.2f}", (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            pr = cv2.addWeighted(over, 0.45, pr, 0.55, 0)
        tiles.append(np.hstack([gt, pr]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    cv2.imwrite(str(out), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run dir (results.csv, weights/)")
    ap.add_argument("--data", required=True, help="dataset dir (images/val, labels/val)")
    ap.add_argument("--out", default=None, help="where to write the page (default: <run>/live)")
    ap.add_argument("--every", type=int, default=20, help="period of the check for a new epoch (s)")
    ap.add_argument("--log", help="training stdout (for the progress bar of the current epoch)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out or run / "live")
    out.mkdir(parents=True, exist_ok=True)
    val = sorted(glob.glob(f"{args.data}/images/val/*.png"))
    images = [val[i] for i in np.linspace(0, len(val) - 1, 8).astype(int)]

    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(out), **k)  # noqa: E731
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"live view: http://localhost:{args.port}", flush=True)
    (out / "index.html").write_text(PAGE)
    last, status, plots, t_check = None, "waiting for epoch 1", 0, 0
    while True:
        rows = []
        try:
            rows = read_csv(run / "results.csv") if (run / "results.csv").exists() else []
            if rows and len(rows) != last and time.time() - t_check > args.every:
                t_check = time.time()
                curves(rows, out / "curves.png")
                preds(run / "weights/last.pt", images, out / "preds.png")
                last, plots = len(rows), int(time.time())
                r = rows[-1]
                status = (f"epoch {int(r['epoch'])}/{args.epochs}, mask mAP50 {r['metrics/mAP50(M)']:.3f}, "
                          f"mAP50-95 {r['metrics/mAP50-95(M)']:.3f} &middot; updated {time.strftime('%H:%M:%S')}")
                print(status.replace("&middot;", "-"), flush=True)
        except Exception as exc:  # file being written: retry next time
            print(f"(retry: {exc})", flush=True)
        done = len(rows)
        elapsed = rows[-1]["time"] if rows else None
        prog = log_progress(args.log) if args.log else None
        st = {"status": status, "epochs": args.epochs, "done": done, "elapsed": elapsed, "plots": plots,
              "eta": elapsed / done * (args.epochs - done) if done else None,
              "phase": prog[0] if prog else "", "epoch": (prog[1] if prog and prog[1] else done + 1),
              "batch": prog[2] if prog else 0, "batch_total": prog[3] if prog else 0}
        if done >= args.epochs or (run / "weights/best.pt").exists() and not prog and done:
            st["phase"] = st["phase"] or "finished"
        (out / "status.json.tmp").write_text(json.dumps(st))
        (out / "status.json.tmp").replace(out / "status.json")
        time.sleep(2)

if __name__ == "__main__":
    main()
