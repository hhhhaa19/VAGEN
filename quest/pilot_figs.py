"""Pilot slide figures: training curves, validation success vs. the authors, and drift.

Run inside the container: bash /projects/p53316/run.sh python pilot_figs.py <out_dir>
"""
import csv, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_wm import parse, tag, well_formed  # noqa: E402  (runs its main on import otherwise)

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
EXP = "/projects/p53316/hhhhaa19/VAGEN/exps/vagen-experiments"
RUNS = {
    "paper": ("/projects/p53316/logs/vagen-paper-full_5614549.log", f"{EXP}/sokoban_bi_level_gae_paper_full_qwen25vl3b/validation"),
    "default": ("/projects/p53316/logs/vagen-full_4539759.log", f"{EXP}/sokoban_bi_level_gae_state_reward_qwen25vl3b/validation"),
}
# Read from the authors' public report (hover tooltips), 2026-10-29; see evidence/9_curve_comparison.md.
AUTHORS = {0: .125, 20: .156, 40: .273, 60: .414, 80: .547, 100: .555, 120: .555, 140: .602, 160: .688,
           180: .734, 200: .750, 220: .781, 240: .758, 260: .789, 280: .758}
KEYS = ["critic/score/mean", "actor/pg_loss", "critic/vf_loss", "actor/entropy", "response_length/mean",
        "val-aux/sokoban/traj_success/mean@1"]

# Reference palette, light mode (dataviz references/palette.md), slots 1-3.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})

def metrics(log):
    rows = {}
    for line in open(log, errors="ignore"):
        m = re.search(r"step:(\d+) (.*)", line)
        if not m:
            continue
        step, row = int(m.group(1)), {}
        for key in KEYS:
            v = re.search(re.escape(key) + r":(?:np\.float64\()?(-?[0-9.e+-]+)", m.group(2))
            if v:
                row[key] = float(v.group(1))
        rows.setdefault(step, {}).update(row)
    return dict(sorted(rows.items()))

def drift(val_dir):
    out = {}
    for name in os.listdir(val_dir):
        step = int(name.split(".")[0])
        total = ok = behind = 0
        for line in open(os.path.join(val_dir, name)):
            for turn in json.loads(line)["output"].split("\nassistant\n"):
                per = tag(turn, "perception")
                total += 1
                ok += per is not None and well_formed(parse(per))
                behind += per is not None and "behind" in per.lower()
        out[step] = (ok / total, behind / total)
    return dict(sorted(out.items()))

def smooth(ys, k=9):
    half = k // 2
    return [sum(ys[max(0, i - half):i + half + 1]) / len(ys[max(0, i - half):i + half + 1]) for i in range(len(ys))]

data = {name: metrics(log) for name, (log, _) in RUNS.items()}
drifts = {name: drift(val) for name, (_, val) in RUNS.items()}

with open(f"{OUT}/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", *KEYS, "perception_parseable", "perception_behind"])
    for name, rows in data.items():
        for step, row in rows.items():
            d = drifts[name].get(step, ("", ""))
            w.writerow([name, step, *[row.get(k, "") for k in KEYS], *d])

# 1. Training curves of the paper-config run.
rows = data["paper"]
steps = [s for s in rows if "actor/pg_loss" in rows[s]]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
panels = [("critic/score/mean", "Training reward (episode score, mean)"),
          ("actor/pg_loss", "Policy loss"), ("critic/vf_loss", "Value loss"), ("actor/entropy", "Policy entropy")]
for ax, (key, title) in zip(axes.flat, panels):
    ys = [rows[s][key] for s in steps]
    ax.plot(steps, ys, color=BLUE, alpha=0.25, linewidth=1)
    ax.plot(steps, smooth(ys), color=BLUE)
    ax.set_title(title)
for ax in axes[1]:
    ax.set_xlabel("Training step")
fig.suptitle("VAGEN-Full on Sokoban, paper config + KL (Qwen2.5-VL-3B, 4x H100, 401 steps, 6h16m)",
             x=0.01, ha="left", fontsize=12, color=INK2)
fig.text(0.01, 0.005, "Thin line: per step. Thick line: 9-step moving average.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(f"{OUT}/fig1_training_curves.png", dpi=200)

# 2. Validation success: ours vs the authors.
fig, ax = plt.subplots(figsize=(11, 5.6))
series = [("Authors, legacy code (128 levels, also in training set)", AUTHORS, AQUA),
          ("Ours, main + paper config + KL (256 held-out maps)",
           {s: r["val-aux/sokoban/traj_success/mean@1"] for s, r in data["paper"].items() if "val-aux/sokoban/traj_success/mean@1" in r}, BLUE),
          ("Ours, main default config (256 held-out maps)",
           {s: r["val-aux/sokoban/traj_success/mean@1"] for s, r in data["default"].items() if "val-aux/sokoban/traj_success/mean@1" in r}, ORANGE)]
for label, pts, color in series:
    xs = sorted(pts)
    ax.plot(xs, [pts[x] for x in xs], color=color, marker="o", markersize=4, label=label)
ax.annotate("0.79 (paper, Table 26)", (260, .789), xytext=(285, .83), color=INK, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("0.488 at step 400", (400, .488), xytext=(335, .58), color=INK, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("collapses to 0.219", (300, .219), xytext=(318, .13), color=INK, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylim(0, 0.9)
ax.set_xlabel("Training step")
ax.set_ylabel("Validation success rate")
ax.set_title("Validation success: same starting point, curves split after step ~40")
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_val_success_vs_authors.png", dpi=200)

# 3. World-model text drift vs validation success (two panels, shared x; no dual axis).
fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
for name, color, label in (("paper", BLUE, "Paper config + KL (judge scorer)"), ("default", ORANGE, "Main default (exact scorer)")):
    d = drifts[name]
    xs = sorted(d)
    top.plot(xs, [d[x][0] for x in xs], color=color, marker="o", markersize=4, label=label)
    v = {s: r["val-aux/sokoban/traj_success/mean@1"] for s, r in data[name].items() if "val-aux/sokoban/traj_success/mean@1" in r}
    bottom.plot(sorted(v), [v[x] for x in sorted(v)], color=color, marker="o", markersize=4, label=label)
top.set_ylim(-0.03, 1.05)
top.set_ylabel("Share of turns")
top.set_title("<perception> written in the required vocabulary (parseable)")
top.legend(loc="lower left", frameon=False, fontsize=10)
bottom.set_ylim(0, 0.6)
bottom.set_ylabel("Success rate")
bottom.set_title("Validation success (256 held-out maps)")
bottom.set_xlabel("Training step")
top.text(292, 0.42, "Perception drifts out of the vocabulary:\nmain default from step 200,\npaper config from step 220", fontsize=9.5, color=INK2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_wm_drift.png", dpi=200)
print("wrote", sorted(os.listdir(OUT)))
