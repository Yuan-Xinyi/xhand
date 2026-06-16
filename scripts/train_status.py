"""Print the latest SimToolReal training metrics from the rl_games tensorboard logs.

One-line summary for watching a long run. Usage: python scripts/train_status.py [run_dir]
"""

import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

run_glob = sys.argv[1] if len(sys.argv) > 1 else "logs/rl_games/simtoolreal/*/"
dirs = sorted(glob.glob(os.path.join(run_glob, "**", "events*"), recursive=True))
if not dirs:
    print("no TB events yet")
    sys.exit(0)
ea = event_accumulator.EventAccumulator(dirs[-1])
ea.Reload()
tags = set(ea.Tags()["scalars"])


def last(tag):
    if tag not in tags:
        return None
    s = ea.Scalars(tag)
    return s[-1].value if s else None


def fmt(x, p=4):
    return f"{x:.{p}f}" if x is not None else "n/a"


rew = last("rewards/iter")
lifted = last("Episode/lifted_frac")
lifted_ft = last("Episode/lifted_frac_fromtable")
succ = last("Episode/mean_successes")
tol = last("Episode/success_tolerance")
n = len(ea.Scalars("rewards/iter")) if "rewards/iter" in tags else 0
print(
    f"[train] iter~{n} | reward={fmt(rew,1)} "
    f"| lifted={fmt(lifted,3)} lifted_FROMTABLE={fmt(lifted_ft,4)} "
    f"mean_succ={fmt(succ,4)} tol={fmt(tol,3)}"
)
