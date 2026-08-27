#!/usr/bin/env python3
"""Plot structural identifiability results from si_report.md."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT = "si_report.md"
OUT_PDF = "si_figure.pdf"

status_colors = {
    "globally": "#2ca02c",
    "locally": "#ff7f0e",
    "nonidentifiable": "#d62728",
}
status_labels = {
    "globally": "globally identifiable",
    "locally": "locally identifiable",
    "nonidentifiable": "non-identifiable",
}

params = []
statuses = []
with open(REPORT) as f:
    for line in f:
        m = re.match(r"\|\s*(\S+)\s*\|\s*(\w+)\s*\|", line)
        if m:
            name, status = m.group(1), m.group(2)
            # Skip state variables (contain '(')
            if "(" in name:
                continue
            params.append(name)
            statuses.append(status)

fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(params))))
y_pos = range(len(params))
colors = [status_colors.get(s, "gray") for s in statuses]
ax.barh(y_pos, [1] * len(params), color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(params)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Structural identifiability — baseline COVID-19 model\n(no changepoints; rational simplification)", fontsize=11)
ax.set_xlabel("Identifiability status", fontsize=10)

# Add text labels
for i, (p, s) in enumerate(zip(params, statuses)):
    ax.text(0.5, i, status_labels.get(s, s), ha="center", va="center", color="white", fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=status_colors[k], edgecolor='black', label=status_labels[k]) for k in status_colors]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=300)
print(f"Saved SI figure to {OUT_PDF}")
