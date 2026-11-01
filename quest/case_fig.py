"""Case-study figure for one validation level across training steps (paper-config run)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

D = sys.argv[1]
SEED = 10010
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, BAD = "#008300", "#e34948"
ROWS = [
    ("step0", "Step 0 (untrained)", False, ["Up", "Up", "Up"],
     'perception: "The box is above and left of the player,\n'
     '             and the target is above and left of the player"\n'
     'answer: Up   (turns 2 and 3: the same text, word for word)',
     "The first Up moves the player; the next two hit the wall and change nothing.\nThe perception never updates, so the agent repeats the blocked action."),
    ("step200", "Step 200", True, ["Left", "Left"],
     'perception: "The box is below andleft of the player,\n'
     '             and the target is above and same column of the player"\n'
     'answer: Left, then Left',
     "Solved, in the required format, but the description is wrong (box is in the same row)\nand words start to fuse ('andleft', 'moveleft')."),
    ("step240", "Step 240", False, [],
     'prediction: "The box will be below same row andleft of the player,\n'
     '             and the target will be same row[Chinese \'and\']left of"   <- cut off\n'
     'perception (turn 2): "The box is <color:red>below</color> same row on theleft ..."',
     "Output breaks off before <answer>; no action executed in any turn.\n32% of all turns at this step have no answer."),
    ("step400", "Step 400", True, ["Left", "Left"],
     'perception: "The box is behind and left of the player,\n'
     '             and the target is behind and left of the player"\n'
     'answer: Left, then Left   (perception identical in both turns)',
     "Solved, but 'behind' is outside the vocabulary and the text no longer tracks the board:\nthe world-model text has decoupled from the action."),
]

fig = plt.figure(figsize=(13, 9.2), facecolor=SURFACE)
grid = fig.add_gridspec(len(ROWS), 5, width_ratios=[1, 1, 1, 1, 5.2], hspace=0.55, wspace=0.08,
                        left=0.02, right=0.99, top=0.9, bottom=0.03)
fig.text(0.02, 0.965, f"One held-out level (seed {SEED}) across training: what the agent writes vs. what it does",
         fontsize=14, fontweight="bold", color=INK)
fig.text(0.02, 0.935, "Truth: box in the same row, left of the player; target further left in the same row. "
         "Two pushes left solve it.", fontsize=10.5, color=INK2)
for r, (key, title, solved, actions, quote, note) in enumerate(ROWS):
    frames = sorted(f for f in os.listdir(D) if f.startswith(f"{SEED}_{key}_")) if key != "step240" else []
    if not frames:
        frames = [f"{SEED}_step0_0.png"]
    labels = ["start"] + actions
    for c in range(4):
        ax = fig.add_subplot(grid[r, c])
        ax.axis("off")
        if c < len(frames):
            ax.imshow(Image.open(os.path.join(D, frames[c])), interpolation="nearest")
            ax.set_title(labels[c] if c < len(labels) else "", fontsize=9, color=INK2, pad=3)
    ax = fig.add_subplot(grid[r, 4])
    ax.axis("off")
    ax.text(0, 1.0, title, fontsize=12.5, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.36, 1.0, "solved" if solved else "failed", fontsize=11, fontweight="bold",
            color=GOOD if solved else BAD, va="top", transform=ax.transAxes)
    ax.text(0, 0.72, quote, fontsize=9, family="monospace", color=INK, va="top", wrap=True, transform=ax.transAxes)
    ax.text(0, 0.22, note, fontsize=9.5, color=INK2, va="top", transform=ax.transAxes)
fig.savefig(sys.argv[2], dpi=200, facecolor=SURFACE)
print("wrote", sys.argv[2])
