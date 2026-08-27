import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "revision/outputs"
DATA_DIR = "codes/Mica.jl/examples/Covid-model"
CHANNELS = ["infected", "hospitalized", "icu", "death", "vaccinated"]
COLORS = {"MICA": "#2ca02c", "competitor": "#1f77b4", "random": "#7f7f7f", "ZERO": "#d62728"}
N = 400  # number of time points used by MICA's BIC/MDL/AIC penalty
N_GLOBAL = 8
N_SEGMENT_SPECIFIC = 8


def load_observed():
    """Load the same 400-day window used by the MICA Covid example.

    The MICA script pads every series to a common length with leading zeros and
    then takes the first 400 values (2020-01-27 to 2021-03-27).  It also applies
    a 14-day smoother to cases, cumulative deaths and cumulative vaccinations.
    We reproduce that window and use a centered rolling mean so the cumulative
    series do not drop artificially at the right edge.
    """
    cases = pd.read_csv(os.path.join(DATA_DIR, "case_rki_daily.csv")).total.values
    hosp = pd.read_csv(os.path.join(DATA_DIR, "Hospitalization_rki_daily.csv")).total.values
    icu = pd.read_csv(os.path.join(DATA_DIR, "icu_rki_daily.csv")).total.values
    death = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "death_rki_daily.csv")).Todesfaelle_neu.values)
    vacc = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "vaccination_rki_daily_allShots.csv")).Total.values)
    data = [cases, hosp, icu, death, vacc]
    max_len = max(len(x) for x in data)
    data = [np.pad(x, (max_len - len(x), 0), constant_values=0) for x in data]
    data = [x[:400] for x in data]

    def ma14(x):
        # Centered 14-day average; endpoints use as many points as available
        # (no zero-padding beyond the series) so cumulative curves stay monotone.
        return pd.Series(x).rolling(window=14, center=True, min_periods=1).mean().values

    data[0] = ma14(data[0])   # infected
    data[3] = ma14(data[3])   # cumulative death
    data[4] = ma14(data[4])   # cumulative vaccinated
    return np.column_stack(data)  # (400, 5)


def category(method):
    if method.startswith("MICA_"):
        return "MICA"
    if method == "ZERO":
        return "ZERO"
    if method.startswith("random_"):
        return "random"
    return "competitor"


def family_of(label):
    if label == "ZERO":
        return "ZERO"
    if label.startswith("HMMRegime_"):
        return "HMM-Regime"
    for suffix in ["_param", "_pen", "_prior", "_thresh"]:
        if suffix in label:
            return label.rsplit(suffix, 1)[0]
    return label


def winner_criterion(label):
    """Select the model-selection score matching the MICA winner's objective."""
    low = label.lower()
    if "mdl" in low:
        return "mdl"
    if "aic" in low:
        return "aic"
    return "bic"


def add_model_selection_scores(summary):
    summary = summary.copy()
    summary['n_cps'] = summary['n_cps'].astype(int)
    summary['p_total'] = N_GLOBAL + (summary['n_cps'] + 1) * N_SEGMENT_SPECIFIC + summary['n_cps']
    logn = np.log(N)
    summary['bic'] = summary['refit_loss'] + summary['p_total'] * logn
    summary['mdl'] = summary['refit_loss'] + 0.5 * summary['p_total'] * logn
    summary['aic'] = summary['refit_loss'] + 2.0 * summary['p_total']
    return summary


def plot_gof(wdir, label, observed, methods_to_plot, score_col):
    sims_dir = os.path.join(wdir, "simulations")
    cp_dir = os.path.join(wdir, "cp_sets")
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    days = np.arange(1, observed.shape[0] + 1)
    score_label = score_col.upper()

    # Distinctive colours: MICA green, competitor blue, random candidates from
    # tab10 excluding those two colours so every algorithm is identifiable.
    cmap = plt.cm.tab10
    rand_palette = [cmap(i) for i in range(cmap.N) if i not in {0, 2}]
    method_colors = {}
    rand_idx = 0
    for m, _ in methods_to_plot:
        cat = category(m)
        if cat == "MICA":
            method_colors[m] = COLORS["MICA"]
        elif cat == "competitor":
            method_colors[m] = COLORS["competitor"]
        else:
            method_colors[m] = rand_palette[rand_idx % len(rand_palette)]
            rand_idx += 1

    for k, ch in enumerate(CHANNELS):
        ax = axes[k]
        # Observed data: black, always labelled, drawn on top.
        ax.plot(days, observed[:, k], 'k-', linewidth=1.5, alpha=0.8, label='observed', zorder=10)

        for method, info in methods_to_plot:
            sim_path = os.path.join(sims_dir, f"{method}.csv")
            cp_path = os.path.join(cp_dir, f"{method}.csv")
            if not os.path.exists(sim_path):
                continue
            sim = pd.read_csv(sim_path)
            cps = []
            if os.path.exists(cp_path):
                cps = pd.read_csv(cp_path)['cp'].astype(int).tolist()
            color = method_colors[method]
            ax.plot(days, sim[ch].values, color=color, linewidth=1.5, alpha=0.8,
                    label=f"{method} ({score_label}={info[score_col]:.1f})", zorder=3)
            # Show change points only in the top (infected) panel to avoid clutter.
            if k == 0:
                for cp in cps:
                    ax.axvline(cp, color=color, linestyle='--', linewidth=1.0, alpha=0.4)

        ax.set_ylabel(ch)
        ax.legend(loc='upper left', fontsize=7, ncol=1)

    axes[-1].set_xlabel('day')
    fig.suptitle(f'Task G — Goodness of fit ({label}, {score_label}-selected)', y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "TASK_G/figures", f"{label}_gof_{score_col}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cp_timeline(wdir, label, methods_to_show, summary, score_col):
    fig, ax = plt.subplots(figsize=(14, 8))
    order = sorted(methods_to_show, key=lambda m: (category(m), m))
    for y, method in enumerate(order):
        cp_path = os.path.join(wdir, "cp_sets", f"{method}.csv")
        if os.path.exists(cp_path):
            cps = pd.read_csv(cp_path)['cp'].astype(int).tolist()
        elif method.startswith("MICA_"):
            row = summary[summary['method'] == method]
            if row.empty:
                continue
            cps = [int(c) for c in str(row.iloc[0]['cps']).split(';') if c]
        else:
            continue
        color = COLORS.get(category(method), 'gray')
        ax.scatter(cps, [y]*len(cps), marker='|', s=200, color=color)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel('day index')
    ax.set_title(f'Task G — {score_col.upper()}-selected change points ({label})')
    ax.set_xlim(0, 400)
    from matplotlib.patches import Patch
    present_cats = sorted(set(category(m) for m in order))
    legend = [Patch(facecolor=COLORS[c], label=c) for c in present_cats]
    ax.legend(handles=legend, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "TASK_G/figures", f"{label}_cp_timeline_{score_col}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(os.path.join(BASE, "TASK_G/figures"), exist_ok=True)
    observed = load_observed()
    winners = sorted(glob.glob(os.path.join(BASE, "TASK_G/winners/*/")))
    for wdir in winners:
        label = os.path.basename(wdir.rstrip('/'))
        score_col = winner_criterion(label)
        summary = pd.read_csv(os.path.join(wdir, "refit_summary.csv"))
        summary = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["refit_loss"])
        summary = add_model_selection_scores(summary)
        summary['cat'] = summary['method'].apply(category)

        # Attach family information if candidate_methods.csv exists
        cand_path = os.path.join(wdir, "candidate_methods.csv")
        if os.path.exists(cand_path):
            cand = pd.read_csv(cand_path)
            summary = summary.merge(cand[['label', 'family']], left_on='method', right_on='label', how='left')
            summary = summary.drop(columns=['label'])
            # Fallback for labels not listed (e.g. MICA winners)
            summary['family'] = summary['family'].fillna(summary['method'].apply(family_of))
        else:
            summary['family'] = summary['method'].apply(family_of)

        # Exclude ZERO from all plots
        summary = summary[summary['cat'] != 'ZERO']

        # Select the best configuration within each family using the winner's criterion
        best_idx = summary.groupby('family')[score_col].idxmin().dropna()
        selected = summary.loc[best_idx].copy()

        # GOF plot: MICA winner + best competitor family + next best 2 competitors
        mica_row = selected[selected['cat'] == 'MICA']
        comp_rows = selected[selected['cat'] == 'competitor']
        best_comp = comp_rows.loc[comp_rows[score_col].idxmin()] if not comp_rows.empty else None

        # Top 3 competitor families (including the best one) by the winner's criterion
        top_comp_rows = comp_rows.nsmallest(3, score_col)

        methods_to_plot = []
        if not mica_row.empty:
            r = mica_row.iloc[0]
            methods_to_plot.append((r['method'], {score_col: r[score_col]}))
        for _, r in top_comp_rows.iterrows():
            methods_to_plot.append((r['method'], {score_col: r[score_col]}))

        if methods_to_plot:
            plot_gof(wdir, label, observed, methods_to_plot, score_col)

        # CP timeline: selected competitor families + MICA winner
        timeline_methods = selected[selected['cat'] == 'competitor']['method'].tolist()
        if not mica_row.empty:
            timeline_methods.insert(0, mica_row.iloc[0]['method'])
        if timeline_methods:
            plot_cp_timeline(wdir, label, timeline_methods, summary, score_col)

        # Save a small CSV of the selected configurations for this winner
        selected_out = selected[['family', 'method', 'n_cps', 'bic', 'mdl', 'aic', 'cps']].copy()
        selected_out.to_csv(os.path.join(wdir, f"selected_{score_col}.csv"), index=False)

    print("Task G GOF + CP timeline figures saved.")


if __name__ == "__main__":
    main()
