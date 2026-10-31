"""Report and slide figures, drawn only from committed result CSVs (no simulation; a few seconds).

Writes results/figures/<name>.pdf (report) and <name>.png (slides, 200 dpi):
    test_results   E8 main result: arrivals per policy in the 6 test settings (sampled), and MAPPO - IPPO with
                   95% CI (results/e8_test/summary_arrivals.csv)
    failures       E8 exploratory 2: where the trains go, T1 no disruptions, sampled: arrived / lost / stuck / other,
                   share of all trains, seeds pooled (results/e8_explore/failures_{summary,trains}.csv)
    validation_curves  IPPO (E6) vs MAPPO (E7) validation arrivals over 200 iterations, sampled: each seed and the
                   mean; iteration 0 = the starter-kit checkpoint both arms start from; heuristic for reference
                   (results/{e6,e7}_val_sampled, val_sampled_baselines, e3_val_baselines)
    reward_story   E3 (official reward) vs E5 (chosen-STOP cost 25 + waiting cost 0.5, value normalisation) over
                   100 iterations: (a) arrivals, (b) official score, validation, sampled (results/{e3,e5}_val_sampled)
    greedy_vs_sampled  per network: (a) T1 test arrivals with sampled vs top-action (greedy) play; (b) mean P(go)
                   when ready to depart, 20 validation scenarios (results/e8_test/summary_arrivals.csv,
                   results/e8_explore/departures.csv)
and tables as <name>.md (slides, docs) and <name>.tex (report, booktabs), from results/e8_test/summary_metrics.csv:
    fairness_table     T1 no disruptions: arrivals, score, lateness (mean, p90), halts, penalty Gini per policy,
                       MAPPO - IPPO with 95% CI
    metrics_all_settings  the same key metrics in all 6 test settings (appendix)

    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py --only test_results

Style (same in every figure): one colour per policy (MAPPO blue, IPPO orange, heuristic aqua, checkpoint grey as a
reference), validated for colour-blind separation; trained policies are filled dots, baselines hollow rings (a second
cue besides colour, and overlapping dots stay visible); every series is also named in text (legend or direct label),
as two colours are below 3:1 contrast on white; hairline grid, no top/right frame.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)  # harmless warnings while embedding the system font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
E8 = ROOT / "results" / "e8_test"
EXPLORE = ROOT / "results" / "e8_explore"

COLOUR = {"MAPPO": "#2a78d6", "IPPO": "#eb6834", "heuristic": "#1baf7a", "checkpoint": "#898781"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WIDTH = 6.5  # inches: the report's text width (1" margins)
SETTINGS = [("t1_scene1_a5-10_ll2", "none", "T1, no disruptions (primary)"),
            ("t1_scene1_a5-10_ll2", "breakdowns", "T1, breakdowns"),
            ("t2_scene1_a15-20_ll2", "none", "T2 (15–20 trains), no disruptions"),
            ("t2_scene1_a15-20_ll2", "breakdowns", "T2 (15–20 trains), breakdowns"),
            ("t3_scene4_a5-10_ll2", "none", "T3 (unseen region), no disruptions"),
            ("t3_scene4_a5-10_ll2", "breakdowns", "T3 (unseen region), breakdowns")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


BASELINES = {"checkpoint", "heuristic"}


def marker(name, size=7):
    """Plot style for a policy: trained arms filled with a white ring, baselines hollow."""
    if name in BASELINES:
        return dict(marker="o", ms=size - 1, mfc="white", mec=COLOUR[name], mew=1.8, ls="", zorder=3)
    return dict(marker="o", ms=size, color=COLOUR[name], mec="white", mew=1.2, ls="", zorder=4)


def test_results():
    d = pd.read_csv(E8 / "summary_arrivals.csv")
    d = d[d["mode"] == "sampled"].set_index(["set", "disruption"])
    rows = [d.loc[(s, dis)] for s, dis, _ in SETTINGS]
    y = list(range(len(rows)))[::-1]  # first setting on top

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.7, 1]})
    series = [("checkpoint", "checkpoint_sampled_arrived"), ("IPPO", "ippo_arrived"), ("MAPPO", "mappo_arrived"),
              ("heuristic", "heuristic_arrived")]
    for yi, r in zip(y, rows):
        values = [r[c] * 100 for _, c in series]
        a.plot([min(values), max(values)], [yi, yi], color=GRID, lw=1.5, zorder=1)
        for (name, col), v in zip(series, values):
            a.plot(v, yi, **marker(name))
    a.set_xlim(0, 100)
    a.set_xlabel("trains arrived (%)")
    a.set_yticks(y, [label for *_, label in SETTINGS])
    a.get_yticklabels()[0].set_fontweight("bold")
    a.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], **marker(n)) for n, _ in series]
    fig.legend(handles, ["checkpoint (starter kit)", "IPPO", "MAPPO", "heuristic (deadlock avoidance)"], ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.4, fontsize=8.5)
    a.set_title("(a) trains arrived, per policy", loc="left", fontsize=9, color=INK_2)

    for yi, r in zip(y, rows):
        mean, lo, hi = (r[c] * 100 for c in ("mappo_minus_ippo", "mappo_minus_ippo_lo", "mappo_minus_ippo_hi"))
        b.plot([lo, hi], [yi, yi], color=COLOUR["MAPPO"], lw=2, zorder=2)
        b.plot(mean, yi, **marker("MAPPO"))
        b.text(10.6, yi, f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK_2)
    b.axvline(0, color=AXIS, lw=0.8, zorder=1)
    b.set_xlim(-1, 10)
    b.set_xticks([0, 2, 4, 6, 8, 10])
    b.set_xlabel("MAPPO − IPPO (points)")
    b.grid(axis="y", visible=False)
    b.set_title("(b) MAPPO − IPPO, 95% CI", loc="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(wspace=0.08)
    return fig


def failures():
    """Stacked bars: each policy's trains split by where they end. Failure parts start at 0 so they compare directly."""
    summary = pd.read_csv(EXPLORE / "failures_summary.csv")
    summary = summary[(summary.still_steps == 50) & (summary.late_steps == 20)].set_index("policy")
    f = pd.read_csv(EXPLORE / "failures_trains.csv")
    f["policy"] = f.label.str.split().str[0]  # "IPPO s0" -> "IPPO"; checkpoint and heuristic stay as they are
    lost = (f.bucket == "still_driving") & (f.driven >= f.shortest_route)
    order = ["checkpoint", "IPPO", "MAPPO", "heuristic"]
    parts = [("lost (drives in loops)", "#6250d6", "white"), ("stuck (stands still)", "#eda100", INK),
             ("other (out of time, never left)", MUTED, "white"), ("arrived", GRID, INK_2)]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    for yi, name in zip(range(len(order))[::-1], order):
        n = summary.loc[name, "trains"]
        shares = [(lost & (f.policy == name)).sum() / n, (f[f.policy == name].bucket == "stuck").sum() / n]
        shares.append(summary.loc[name, "not_arrived"] - sum(shares))
        shares.append(1 - summary.loc[name, "not_arrived"])
        left = 0
        for share, (_, colour, text) in zip(shares, parts):
            width = share * 100
            ax.barh(yi, width, left=left, height=0.62, color=colour, edgecolor="white", linewidth=1.5)
            if width >= 6:  # print the share only where it fits
                ax.text(left + 1.2, yi, f"{width:.0f}%", va="center", fontsize=8, color=text)
            left += width
    ax.set_yticks(range(len(order))[::-1], order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of all trains (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in parts]
    ax.legend(handles, [p for p, _, _ in parts], ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.25),
              handlelength=1.0, handletextpad=0.4, columnspacing=1.4, fontsize=8.5)
    ax.set_title("T1, no disruptions, sampled actions; IPPO and MAPPO: 3 seeds pooled", loc="left", fontsize=9,
                 color=INK_2)
    return fig


def val_baseline(folder, policy=None, metric="completion_rate") -> float:
    """A baseline's mean on the validation set: per scenario over evaluation seeds, then over scenarios."""
    e = pd.read_csv(ROOT / "results" / folder / "episodes.csv", dtype={"scenario": str})
    e = e[e.policy == policy] if policy else e
    return e.groupby("scenario")[metric].mean().mean()


def curves(ax, folder, colour, label, start, metric="completion_rate", scale=100):
    """One run's validation curve: a thin line per training seed and a thick mean, from `start` at iteration 0
    (the checkpoint every run is warm-started from). Returns the final mean."""
    s = pd.read_csv(ROOT / "results" / folder / "summary_by_snapshot.csv")
    for _, seed in s.groupby("seed"):
        ax.plot([0, *seed.iteration], [start * scale, *seed[metric] * scale], color=colour, lw=0.8, alpha=0.35)
    mean = s.groupby("iteration")[metric].mean()
    ax.plot([0, *mean.index], [start * scale, *mean * scale], color=colour, lw=2, label=label)
    return mean.iloc[-1]


def validation_curves():
    start = val_baseline("val_sampled_baselines")             # checkpoint, sampled, seeds 0-2 = both arms at it 0
    heuristic = val_baseline("e3_val_baselines", "heuristic")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    for name, folder in (("IPPO", "e6_val_sampled"), ("MAPPO", "e7_val_sampled")):
        final = curves(ax, folder, COLOUR[name], f"{name} (mean of 3 seeds)", start)
        ax.text(203, final * 100, f"{name} {final:.0%}", va="center", fontsize=8.5, color=INK)
    ax.axhline(heuristic * 100, color=COLOUR["heuristic"], lw=1)
    ax.text(203, heuristic * 100, f"heuristic {heuristic:.0%}", va="center", fontsize=8.5, color=INK)
    ax.plot(0, start * 100, clip_on=False, **marker("checkpoint"))
    ax.annotate(f"start: checkpoint {start:.0%}", (0, start * 100), xytext=(8, -12), textcoords="offset points",
                fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 200)
    ax.set_ylim(30, 90)
    ax.set_xticks(range(0, 201, 25))
    ax.set_xlabel("training iteration (32 games each)")
    ax.set_ylabel("trains arrived (%)")
    ax.legend(loc="lower right", fontsize=8.5, handlelength=1.5)
    ax.set_title("validation set (50 scenarios), sampled actions; thin lines: individual training seeds", loc="left",
                 fontsize=9, color=INK_2)
    return fig


def reward_story():
    """Small multiples (not a dual axis): arrivals and the official score of E3 and E5 over the same iterations."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharex=True)
    runs = [("E3: official reward", "e3_val_sampled", MUTED), ("E5: chosen-STOP + waiting cost", "e5_val_sampled",
                                                               COLOUR["IPPO"])]
    for ax, metric, scale, fmt, ylim, title in ((a, "completion_rate", 100, "{:.0%}", (30, 75), "(a) trains arrived (%)"),
                                                (b, "score", 1, "{:.2f}", (0.4, 0.9), "(b) official score")):
        start = val_baseline("val_sampled_baselines", metric=metric)
        finals = [curves(ax, folder, colour, label, start, metric, scale) for label, folder, colour in runs]
        gap = 0.06 * (ylim[1] - ylim[0])  # end labels closer than this are pushed apart
        low, high = sorted(finals)
        shift = max(0, gap - (high - low) * scale) / 2
        for final in finals:
            y = final * scale + (shift if final == high else -shift)
            ax.text(102, y, fmt.format(final), va="center", fontsize=8.5, color=INK)
        ax.plot(0, start * scale, clip_on=False, **marker("checkpoint"))
        ax.set_xlim(0, 100)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(0, 101, 25))
        ax.set_xlabel("training iteration")
        ax.set_title(title, loc="left", fontsize=9, color=INK_2)
    a.annotate("checkpoint (start)", (0, val_baseline("val_sampled_baselines") * 100), xytext=(6, -30),
               textcoords="offset points", fontsize=8, color=INK_2)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8.5,
               handlelength=1.5, columnspacing=2)
    fig.subplots_adjust(wspace=0.3)
    return fig


TINT = {"IPPO": "#f7c3ad", "MAPPO": "#9ec5f4", "checkpoint": "#e1e0d9"}  # light steps of each policy's hue


def greedy_vs_sampled():
    """Dumbbells: sampled (circle) vs greedy (diamond) per network; shape carries the mode, colour the policy."""
    a_rows = pd.read_csv(E8 / "summary_arrivals.csv").set_index(["set", "disruption", "mode"])
    t1 = {mode: a_rows.loc[("t1_scene1_a5-10_ll2", "none", mode)] for mode in ("sampled", "greedy")}
    arrived = {"checkpoint": (t1["sampled"].checkpoint_sampled_arrived, t1["sampled"].checkpoint_greedy_arrived)}
    for arm, col in (("IPPO", "ippo_seeds"), ("MAPPO", "mappo_seeds")):
        for seed, (s, g) in enumerate(zip(t1["sampled"][col].split("/"), t1["greedy"][col].split("/"))):
            arrived[f"{arm} s{seed}"] = (float(s), float(g))
    p_go = pd.read_csv(EXPLORE / "departures.csv").set_index("network").mean_p_go
    names = list(arrived)                                   # checkpoint, IPPO s0-2, MAPPO s0-2
    y = [7.8, 6.4, 5.4, 4.4, 3.0, 2.0, 1.0]                 # gaps between the three groups
    labels = ["checkpoint", "IPPO seed 0", "IPPO seed 1", "IPPO seed 2", "MAPPO seed 0", "MAPPO seed 1",
              "MAPPO seed 2"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    for yi, name in zip(y, names):
        policy = name.split()[0]
        sampled, greedy = (v * 100 for v in arrived[name])
        a.plot([sampled, greedy], [yi, yi], color=GRID, lw=2, zorder=1)
        a.plot(greedy, yi, marker="D", ms=6, mfc=TINT[policy], mec=COLOUR[policy], mew=1.2, ls="", zorder=3)
        a.plot(sampled, yi, marker="o", ms=7, color=COLOUR[policy], mec="white", mew=1.2, ls="", zorder=4)
        b.plot([0, p_go[name]], [yi, yi], color=GRID, lw=2, zorder=1)
        b.plot(p_go[name], yi, marker="o", ms=7, color=COLOUR[policy], mec="white", mew=1.2, ls="", zorder=4)
    a.set_xlim(0, 80)
    a.set_xticks([0, 20, 40, 60])
    a.set_xlabel("trains arrived, test set T1 (%)")
    a.set_yticks(y, labels)
    a.grid(axis="y", visible=False)
    a.set_title("(a) sampled vs greedy play", loc="left", fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ms=7, color=MUTED, mec="white", ls=""),
               plt.Line2D([], [], marker="D", ms=6, mfc=TINT["checkpoint"], mec=MUTED, mew=1.2, ls="")]
    fig.legend(handles, ["sampled (as in training)", "greedy (always the top action)"], ncol=2, loc="upper center",
               bbox_to_anchor=(0.45, -0.02), fontsize=8.5, handletextpad=0.3, columnspacing=2)

    b.axvline(0.5, color=AXIS, lw=0.8, zorder=1)
    b.text(0.52, 8.25, "above 0.5: greedy\ntrains leave at once", fontsize=7.5, color=INK_2, va="bottom")
    b.set_xlim(0, 1)
    b.set_ylim(0.4, 9.1)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    b.set_xlabel("mean P(go) when ready to depart")
    b.grid(axis="y", visible=False)
    b.set_title("(b) departure probability", loc="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(wspace=0.1)
    return fig


# metric -> (row label, scale, decimals); lateness is in steps, among arrived trains only
METRICS = {"completion_rate": ("Trains arrived (%)", 100, 1), "score": ("Official score", 1, 3),
           "mean_lateness": ("Mean lateness (steps)", 1, 1), "p90_lateness": ("p90 lateness (steps)", 1, 1),
           "halts_while_moving": ("Halts while moving, per episode", 1, 1),
           "gini_penalty": ("Gini of train penalties", 1, 3)}
TABLE_NOTE = ("Per scenario: mean over 3 training x 3 evaluation seeds (heuristic: deterministic); then mean over "
              "scenarios. Difference: paired over scenarios, 95% bootstrap CI. Lateness only over trains that "
              "arrived, so the arms are averaged over slightly different trains. Gini: 0 = penalties shared equally, "
              "1 = one train takes all.")


def metric_rows(m: pd.DataFrame, setting):
    """Formatted cells (label, IPPO, MAPPO, MAPPO - IPPO [CI], heuristic) for one test setting."""
    rows = []
    for metric, (label, scale, dec) in METRICS.items():
        r = m.loc[(*setting, metric)]
        f = lambda v: f"{v * scale:.{dec}f}"  # noqa: E731
        g = lambda v: f"{v * scale:+.{dec}f}"  # noqa: E731
        rows.append([label, f(r.ippo), f(r.mappo), f"{g(r.mappo_minus_ippo)} [{g(r.lo)}, {g(r.hi)}]", f(r.heuristic)])
    return rows


def write_table(name, header, rows, caption):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text(f"{caption}\n\n" + "\n".join(md) + f"\n\n{TABLE_NOTE}\n")
    tex = lambda cell: re.sub(r"-(?=\d)", "$-$", cell.replace("%", r"\%").replace("−", "$-$"))  # noqa: E731
    lines = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text(f"% {caption}\n% {TABLE_NOTE}\n" + "\n".join(lines) + "\n")


def load_metrics():
    return pd.read_csv(E8 / "summary_metrics.csv").set_index(["set", "disruption", "metric"])


def fairness_table():
    rows = metric_rows(load_metrics(), ("t1_scene1_a5-10_ll2", "none"))
    write_table("fairness_table", ["T1, no disruptions", "IPPO", "MAPPO", "MAPPO − IPPO [95% CI]", "Heuristic"], rows,
                "Test set T1 (100 scenarios, 5–10 trains), no disruptions, sampled actions.")


def metrics_all_settings():
    m, rows = load_metrics(), []
    keep = ["Trains arrived (%)", "Official score", "p90 lateness (steps)", "Gini of train penalties"]
    for s, dis, label in SETTINGS:
        rows += [[f"{label}: {r[0]}", *r[1:]] for r in metric_rows(m, (s, dis)) if r[0] in keep]
    write_table("metrics_all_settings", ["Setting: metric", "IPPO", "MAPPO", "MAPPO − IPPO [95% CI]", "Heuristic"],
                rows, "All six test settings, sampled actions (appendix).")


FIGURES = {"test_results": test_results, "failures": failures, "validation_curves": validation_curves,
           "reward_story": reward_story, "greedy_vs_sampled": greedy_vs_sampled}
TABLES = {"fairness_table": fairness_table, "metrics_all_settings": metrics_all_settings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=[*FIGURES, *TABLES], help="make one figure or table")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        if args.only and name != args.only:
            continue
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        print(f"wrote {(OUT / name).relative_to(ROOT)}.pdf/.png")
    for name, make in TABLES.items():
        if args.only and name != args.only:
            continue
        make()
        print(f"wrote {(OUT / name).relative_to(ROOT)}.md/.tex")


if __name__ == "__main__":
    sys.exit(main())
