#!/usr/bin/env python3
"""
Usage：python plot.py <output_dir>
Demo：python ~/DreamBooth/plot.py ~/DreamBooth/output/20260417_154645
"""

import sys, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if len(sys.argv) < 2:
    print("Usage: python plot.py <output_dir>")
    sys.exit(1)

output_dir = sys.argv[1].rstrip("/")
timestamp  = os.path.basename(output_dir)   # Infer the timestamp directly from the output directory name.
log_file   = os.path.join(output_dir, "train.log")
max_steps  = 600

if not os.path.exists(log_file):
    print(f"[!] Log file not found: {log_file}")
    print("[!] Please verify that train.sh completed normally and wrote train.log.")
    sys.exit(1)

# -- Parse training loss -------------------------------------------------------
steps, losses = [], []
with open(log_file, errors="ignore") as f:
    for line in f:
        # Expected tqdm format: 320/800 [..., loss=0.0842]
        m = re.search(r"(\d+)/\d+.*?loss=([0-9]+\.[0-9]+(?:e[+-]?\d+)?)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

print(f"  Log file       : {log_file}")
print(f"  Parsed entries : {len(steps)} loss records")

if not steps:
    print("[!] No loss records were found. Displaying the final 10 log lines for diagnostics:")
    with open(log_file) as f:
        lines = f.readlines()
    for l in lines[-10:]:
        print(" |", l.rstrip())
    sys.exit(1)

# -- Export CSV ---------------------------------------------------------------
csv_path = os.path.join(output_dir, "loss.csv")
with open(csv_path, "w") as f:
    f.write("step,loss\n")
    for s, l in zip(steps, losses):
        f.write(f"{s},{l:.6f}\n")
print(f"✅ CSV → {csv_path}")

# -- Generate figure ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(steps, losses, color="#90CAF9", linewidth=1, alpha=0.5, label="Loss original")

w  = max(1, len(losses) // 15)
sm = [sum(losses[max(0, i-w):i+1]) / len(losses[max(0, i-w):i+1])
      for i in range(len(losses))]
ax.plot(steps, sm, color="#1565C0", linewidth=2, label=f"Loss smoothed (window={w})")

for ck in [200, 400, 600, 800]:
    pts = [(s, l) for s, l in zip(steps, sm) if abs(s - ck) <= 5]
    if pts:
        sx, sy = pts[0]
        ax.axvline(x=sx, color="#B0BEC5", linewidth=0.8, linestyle="--")
        ax.annotate(f"{sy:.4f}", xy=(sx, sy),
                    xytext=(6, 8), textcoords="offset points",
                    fontsize=8, color="#1565C0")

ax.annotate(f"Final: {sm[-1]:.4f}", xy=(steps[-1], sm[-1]),
            xytext=(-80, 18), textcoords="offset points",
            fontsize=10, color="#D32F2F", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2))

ax.set_title(f"DreamBooth + LoRA  Loss  (SD v1.5 · FP32 · rank=128)\n{timestamp}",
             fontsize=12)
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_xlim(0, max_steps)
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(fontsize=9)
plt.tight_layout()

out_path = os.path.join(output_dir, "loss_curve.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✅ Loss curve → {out_path}")
