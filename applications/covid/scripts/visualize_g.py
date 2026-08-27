import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "revision/outputs"

def category(method):
    if method.startswith("MICA_"):
        return "MICA"
    if method == "ZERO":
        return "ZERO"
    if method.startswith("random_"):
        return "random"
    return "competitor"

def color(cat):
    return {"MICA": "#2ca02c", "ZERO": "#d62728", "random": "#7f7f7f", "competitor": "#1f77b4"}[cat]

def plot_task_g():
    os.makedirs(os.path.join(BASE, "TASK_G/figures"), exist_ok=True)
    winners = sorted(glob.glob(os.path.join(BASE, "TASK_G/winners/*/")))
    for wdir in winners:
        label = os.path.basename(wdir.rstrip('/'))
        df = pd.read_csv(os.path.join(wdir, "refit_summary.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["refit_loss"])
        df['cat'] = df['method'].apply(category)
        df = df.sort_values("refit_loss").reset_index(drop=True)

        # 1. Top 25 refit-loss bar chart
        top_n = 25
        sub = df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(8, 8))
        y = np.arange(len(sub))
        ax.barh(y, sub['refit_loss'], color=[color(c) for c in sub['cat']])
        ax.set_yticks(y)
        ax.set_yticklabels(sub['method'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('SEIRD refit loss')
        ax.set_title(f'Task G — Top {top_n} CP sets by SEIRD refit loss ({label})')
        # legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color(c), label=c) for c in ["MICA", "competitor", "random", "ZERO"]]
        ax.legend(handles=legend_elements, loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(BASE, "TASK_G/figures", f"{label}_refit_loss_top25.pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 2. Random baseline vs number of CPs
        rand = df[df['cat'] == 'random'].copy()
        if not rand.empty:
            rand['size'] = rand['method'].str.split('_').str[1].str.replace('cp', '').astype(int)
            sizes = sorted(rand['size'].unique())
            data = [rand[rand['size']==s]['refit_loss'].values for s in sizes]
            fig, ax = plt.subplots(figsize=(7, 4))
            bp = ax.boxplot(data, labels=sizes, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#7f7f7f')
            # overlay MICA and ZERO lines
            mica_loss = df[df['cat']=='MICA']['refit_loss'].min() if (df['cat']=='MICA').any() else None
            zero_loss = df[df['cat']=='ZERO']['refit_loss'].min() if (df['cat']=='ZERO').any() else None
            if mica_loss is not None:
                ax.axhline(mica_loss, color='#2ca02c', linestyle='--', label='MICA winner')
            if zero_loss is not None:
                ax.axhline(zero_loss, color='#d62728', linestyle=':', label='ZERO')
            ax.set_xlabel('number of random CPs')
            ax.set_ylabel('SEIRD refit loss')
            ax.set_title(f'Task G — Random CP baseline ({label})')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(BASE, "TASK_G/figures", f"{label}_random_by_size.pdf"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 3. Summary CSV of top 10
        top10 = df.head(10)[['method', 'n_cps', 'refit_loss', 'cat']]
        top10.to_csv(os.path.join(BASE, "TASK_G/figures", f"{label}_top10.csv"), index=False)

    print("Task G figures saved.")

if __name__ == "__main__":
    plot_task_g()
