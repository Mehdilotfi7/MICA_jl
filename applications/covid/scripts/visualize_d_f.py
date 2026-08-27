import os
import glob as globmod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "revision/outputs"

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------- TASK D ----------------
def plot_task_d():
    winners = sorted(globmod.glob(os.path.join(BASE, "TASK_D/winners/*/")))
    for wdir in winners:
        label = os.path.basename(wdir.rstrip('/'))
        ci = pd.read_csv(os.path.join(wdir, "parameter_bootstrap_ci.csv"))
        prof = pd.read_csv(os.path.join(wdir, "cp_profile_loss.csv"))

        # global params
        global_df = ci[ci['parameter'].str.endswith('_global')].copy()
        if not global_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(global_df))
            ax.errorbar(x, global_df['best_fit'],
                        yerr=[np.abs(global_df['best_fit']-global_df['lower_95']), np.abs(global_df['upper_95']-global_df['best_fit'])],
                        fmt='o', capsize=4, color='steelblue', label='best fit ± 95% CI')
            ax.scatter(x, global_df['median'], color='orange', marker='s', label='bootstrap median')
            ax.set_xticks(x)
            ax.set_xticklabels(global_df['parameter'], rotation=45, ha='right')
            ax.set_ylabel('parameter value')
            ax.set_title(f'Task D — Global parameter bootstrap CIs ({label})')
            ax.legend()
            savefig(fig, os.path.join(BASE, "TASK_D/figures", f"{label}_global_params_bootstrap.pdf"))

        # segment-specific params: one subplot per parameter
        seg = ci[~ci['parameter'].str.endswith('_global')].copy()
        if not seg.empty:
            seg['base'] = seg['parameter'].str.rsplit('_', n=1).str[0]
            seg['seg'] = seg['parameter'].str.rsplit('_', n=1).str[1]
            bases = sorted(seg['base'].unique())
            n_bases = len(bases)
            cols = 4
            rows = int(np.ceil(n_bases/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), sharex=True)
            if n_bases == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for ax, base in zip(axes, bases):
                sub = seg[seg['base']==base].sort_values('seg')
                x = np.arange(len(sub))
                ax.errorbar(x, sub['best_fit'],
                            yerr=[np.abs(sub['best_fit']-sub['lower_95']), np.abs(sub['upper_95']-sub['best_fit'])],
                            fmt='o', capsize=3, color='steelblue')
                ax.scatter(x, sub['median'], color='orange', marker='s')
                ax.set_xticks(x)
                ax.set_xticklabels(sub['seg'], rotation=45, ha='right')
                ax.set_title(base)
                ax.set_ylabel('value')
            for ax in axes[n_bases:]:
                ax.axis('off')
            fig.suptitle(f'Task D — Segment-specific parameter bootstrap CIs ({label})', y=1.02)
            savefig(fig, os.path.join(BASE, "TASK_D/figures", f"{label}_segment_params_bootstrap.pdf"))

        # CP profile
        if not prof.empty:
            cp_indices = sorted(prof['cp_index'].unique())
            fig, axes = plt.subplots(1, len(cp_indices), figsize=(5*len(cp_indices), 4), sharey=False)
            if len(cp_indices) == 1:
                axes = [axes]
            best_loss = prof['loss'].min() - prof['delta_loss'].min()  # original best
            for ax, idx in zip(axes, cp_indices):
                sub = prof[prof['cp_index']==idx].sort_values('candidate_cp')
                orig = sub['original_cp'].iloc[0]
                ax.plot(sub['candidate_cp'], sub['loss'], '-o', markersize=3)
                ax.axvline(orig, color='r', linestyle='--', label=f'best CP {orig}')
                ax.axhline(best_loss + 3.84, color='g', linestyle=':', label='best + 3.84')
                within = sub[sub['delta_loss'] <= 3.84]
                if not within.empty:
                    ax.axvspan(within['candidate_cp'].min(), within['candidate_cp'].max(), color='green', alpha=0.15)
                ax.set_xlabel('candidate CP (day index)')
                ax.set_ylabel('refit loss')
                ax.set_title(f'CP #{idx}')
                ax.legend(fontsize=8)
            fig.suptitle(f'Task D — CP profile likelihood intervals ({label})', y=1.02)
            savefig(fig, os.path.join(BASE, "TASK_D/figures", f"{label}_cp_profile.pdf"))

    print("Task D figures saved.")

# ---------------- TASK F ----------------
def plot_task_f():
    winners = sorted(globmod.glob(os.path.join(BASE, "TASK_F/winners/*/")))
    for wdir in winners:
        label = os.path.basename(wdir.rstrip('/'))
        red = pd.read_csv(os.path.join(wdir, "redundancy_test.csv"))
        loc = pd.read_csv(os.path.join(wdir, "local_relocation.csv"))

        # redundancy
        if not red.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['tomato' if a=='keep' else 'gray' for a in red['action']]
            ax.bar(red['cp'].astype(str), red['relative_increase'], color=colors)
            ax.axhline(0.05, color='k', linestyle='--', label='5% threshold')
            ax.set_xlabel('change point (day index)')
            ax.set_ylabel('relative loss increase when dropped')
            ax.set_title(f'Task F — Redundancy test ({label})')
            ax.legend()
            savefig(fig, os.path.join(BASE, "TASK_F/figures", f"{label}_redundancy.pdf"))

        # relocation
        if not loc.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            y = np.arange(len(loc))
            ax.scatter(loc['cp'], y, color='black', label='original CP', zorder=3)
            ax.scatter(loc['best_local_cp'], y, color='red', marker='>', label='relocated CP', zorder=3)
            for i, row in loc.iterrows():
                ax.plot([row['cp'], row['best_local_cp']], [i, i], 'k-', alpha=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels([f"CP {c}" for c in loc['cp']])
            ax.set_xlabel('day index')
            ax.set_title(f'Task F — Local CP relocation ({label})')
            ax.legend()
            savefig(fig, os.path.join(BASE, "TASK_F/figures", f"{label}_relocation.pdf"))

    print("Task F figures saved.")

if __name__ == "__main__":
    plot_task_d()
    plot_task_f()
