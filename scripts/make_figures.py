"""Report and slide figures, drawn only from committed result CSVs (no simulation; a few seconds).

Writes results/figures/<name>.pdf (report) and <name>.png (slides, 200 dpi):
    test_results   E8 main result: arrivals per policy in the 6 test settings (sampled), and MAPPO - IPPO with
                   95% CI (results/e8_test/summary_arrivals.csv)
    failures       E8 exploratory 2: where the trains go, T1 no disruptions, sampled: arrived / lost / stuck / other,
                   share of all trains, seeds pooled (results/e8_explore/failures_{summary,trains}.csv)
    validation_curves  IPPO vs MAPPO validation arrivals over 400 iterations, sampled: each seed and the mean;
                   iteration 0 = the starter-kit checkpoint both arms start from; 10-200 = E6/E7, 210-400 = E9
                   (bit-identical networks up to 200); heuristic for reference (results/{e6,e7}_val_sampled,
                   e9_{ippo,mappo}_val_sampled, val_sampled_baselines, e3_val_baselines)
    budget_effect  E9 (secondary): test arrivals of each arm at iteration 200 (E8) and 400, and MAPPO - IPPO at
                   both with 95% CI, per test setting (results/e9_test/summary_arrivals.csv)
    dense_collapse E10 (secondary) + E10 exploratory: (a, b) validation arrivals of each dense-trained run (15-20
                   trains), per seed; (c, d) how long trains wait to depart, D1 no disruptions, evaluation seed 0, at
                   iterations 200 and 400 (results/e10_{ippo,mappo}_val_sampled, e10_val_baselines,
                   results/e10_explore/collapse_trains.csv)
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
E9 = ROOT / "results" / "e9_test"
EXPLORE = ROOT / "results" / "e8_explore"
E10_EXPLORE = ROOT / "results" / "e10_explore"

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


def curves(ax, folders, colour, label, start, metric="completion_rate", scale=100):
    """One run's validation curve: a thin line per training seed and a thick mean, from `start` at iteration 0
    (the checkpoint every run is warm-started from). `folders`: one folder, or several holding consecutive
    iterations of the same runs. Returns the mean per iteration."""
    folders = [folders] if isinstance(folders, str) else folders
    s = pd.concat([pd.read_csv(ROOT / "results" / f / "summary_by_snapshot.csv") for f in folders])
    assert not s.duplicated(["seed", "iteration"]).any(), folders
    s = s.sort_values(["seed", "iteration"])
    for _, seed in s.groupby("seed"):
        ax.plot([0, *seed.iteration], [start * scale, *seed[metric] * scale], color=colour, lw=0.8, alpha=0.35)
    mean = s.groupby("iteration")[metric].mean()
    ax.plot([0, *mean.index], [start * scale, *mean * scale], color=colour, lw=2, label=label)
    return mean


def spread(values, gap):
    """Label positions for two values: pushed apart symmetrically until at least `gap` apart."""
    low, high = sorted(values)
    shift = max(0, gap - (high - low)) / 2
    return [v + (shift if v == high else -shift) for v in values]


def validation_curves():
    start = val_baseline("val_sampled_baselines")             # checkpoint, sampled, seeds 0-2 = both arms at it 0
    heuristic = val_baseline("e3_val_baselines", "heuristic")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    arms = (("IPPO", ("e6_val_sampled", "e9_ippo_val_sampled")), ("MAPPO", ("e7_val_sampled", "e9_mappo_val_sampled")))
    means = {name: curves(ax, folders, COLOUR[name], f"{name} (mean of 3 seeds)", start) for name, folders in arms}
    for x, text in ((200, "E8 tested\niteration 200"), (400, "E9 tested\niteration 400")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.text(x - 4, 80, text, ha="right", va="center", fontsize=7.5, color=INK_2)
    ends = spread([means[name].loc[400] * 100 for name, _ in arms], gap=3.6)
    for (name, _), y in zip(arms, ends):
        ax.text(406, y, f"{name} {means[name].loc[200]:.0%} $\\rightarrow$ {means[name].loc[400]:.0%}", va="center",
                fontsize=8.5, color=INK)
    ax.axhline(heuristic * 100, color=COLOUR["heuristic"], lw=1)
    ax.text(406, heuristic * 100, f"heuristic {heuristic:.0%}", va="center", fontsize=8.5, color=INK)
    ax.plot(0, start * 100, clip_on=False, **marker("checkpoint"))
    ax.annotate(f"start: checkpoint {start:.0%}", (0, start * 100), xytext=(8, -12), textcoords="offset points",
                fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 400)
    ax.set_ylim(30, 90)
    ax.set_xticks(range(0, 401, 50))
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
        finals = [curves(ax, folder, colour, label, start, metric, scale).iloc[-1] for label, folder, colour in runs]
        gap = 0.06 * (ylim[1] - ylim[0])  # end labels closer than this are pushed apart
        for final, y in zip(finals, spread([f * scale for f in finals], gap)):
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


def budget_effect():
    """E9: per setting, each arm's arrivals at 200 (light dot) and 400 (filled dot), and MAPPO - IPPO at both."""
    d = pd.read_csv(E9 / "summary_arrivals.csv")
    d = d[d["mode"] == "sampled"].set_index(["set", "disruption"])
    rows = [d.loc[(s, dis)] for s, dis, _ in SETTINGS]
    y = list(range(len(rows)))[::-1]
    off = 0.17  # IPPO / iteration 200 just above the setting's line, MAPPO / iteration 400 just below

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.7, 1]})
    for yi, r in zip(y, rows):
        for name, dy in (("IPPO", off), ("MAPPO", -off)):
            before, after = r[f"{name.lower()}_200"] * 100, r[f"{name.lower()}_400"] * 100
            a.plot([before, after], [yi + dy] * 2, color=COLOUR[name], lw=1.5, zorder=2)
            a.plot(before, yi + dy, marker="o", ms=5.5, mfc=TINT[name], mec=COLOUR[name], mew=1.2, ls="", zorder=3)
            a.plot(after, yi + dy, **marker(name))
    a.set_xlim(50, 80)
    a.set_xticks([50, 60, 70, 80])
    a.set_xlabel("trains arrived (%)")
    a.set_yticks(y, [label for *_, label in SETTINGS])
    a.get_yticklabels()[0].set_fontweight("bold")
    a.grid(axis="y", visible=False)
    a.set_title("(a) trains arrived, iteration 200 $\\rightarrow$ 400", loc="left", fontsize=9, color=INK_2)

    for yi, r in zip(y, rows):
        for it, dy, style in ((200, off, dict(marker="o", ms=5.5, mfc=TINT["checkpoint"], mec=MUTED, mew=1.2,
                                                  ls="", zorder=3)),
                              (400, -off, marker("MAPPO"))):
            mean, lo, hi = (r[f"mappo_minus_ippo_{it}{c}"] * 100 for c in ("", "_lo", "_hi"))
            b.plot([lo, hi], [yi + dy] * 2, color=MUTED if it == 200 else COLOUR["MAPPO"], lw=2, zorder=2)
            b.plot(mean, yi + dy, **style)
        change, lo, hi = (r[f"gap_change{c}"] * 100 for c in ("", "_lo", "_hi"))
        b.text(10.6, yi, f"{change:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK_2)
    b.text(10.6, len(rows) - 0.45, "gap change", va="center", fontsize=8, color=INK_2)
    b.axvline(0, color=AXIS, lw=0.8, zorder=1)
    b.set_xlim(-3, 10)
    b.set_xticks([0, 2, 4, 6, 8, 10])
    b.set_ylim(-0.6, len(rows) - 0.25)
    b.set_xlabel("MAPPO − IPPO (points)")
    b.grid(axis="y", visible=False)
    b.set_title("(b) MAPPO − IPPO, 95% CI", loc="left", fontsize=9, color=INK_2)

    handles = [plt.Line2D([], [], **marker("IPPO")), plt.Line2D([], [], **marker("MAPPO")),
               plt.Line2D([], [], marker="o", ms=5.5, mfc=TINT["checkpoint"], mec=MUTED, mew=1.2, ls=""),
               plt.Line2D([], [], marker="o", ms=7, color=MUTED, mec="white", mew=1.2, ls="")]
    fig.legend(handles, ["IPPO", "MAPPO", "iteration 200 (E8)", "iteration 400 (E9)"], ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.6, fontsize=8.5)
    fig.subplots_adjust(wspace=0.08)
    return fig


def spaced(values, gap):
    """Label positions for any number of values: in order, at least `gap` apart, centred on the originals."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = []
    for i in order:
        placed.append(max(values[i], placed[-1] + gap) if placed else values[i])
    shift = (sum(values) - sum(placed)) / len(values)
    out = [0.0] * len(values)
    for i, y in zip(order, placed):
        out[i] = y + shift
    return out


WAIT_BINS = [("0–5", 5, "#e0dcf7", INK), ("6–100", 100, "#b3a9ee", INK), ("101–300", 300, "#7d6ddf", "white"),
             ("> 300 or never", None, "#3f2f9e", "white")]  # steps waited after the earliest departure


def dense_collapse():
    """E10: per run, validation arrivals (top) and how long its trains wait to depart at iterations 200 and 400
    (bottom), one column per arm. No arm mean: it would average collapsed and healthy runs."""
    start = val_baseline("e10_val_baselines", "actor-sampled:third_party/ecml2026-starterkit/reinforcement_learning/"
                                              "checkpoint.pt")
    heuristic = val_baseline("e10_val_baselines", "heuristic")
    trains = pd.read_csv(E10_EXPLORE / "collapse_trains.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55,
                                                                       "wspace": 0.38})
    for col, name in enumerate(("IPPO", "MAPPO")):
        ax = axes[0, col]
        v = pd.read_csv(ROOT / "results" / f"e10_{name.lower()}_val_sampled" / "summary_by_snapshot.csv")
        ends = {}
        for seed, g in v.sort_values("iteration").groupby("seed"):
            ax.plot([0, *g.iteration], [start * 100, *g.completion_rate * 100], color=COLOUR[name], lw=1.3)
            ends[seed] = g.completion_rate.iloc[-1] * 100
        for seed, y in zip(ends, spaced(list(ends.values()), gap=4.2)):
            ax.text(406, y, f"seed {seed} {ends[seed]:.0f}%", va="center", fontsize=8, color=INK)
        ax.axhline(heuristic * 100, color=COLOUR["heuristic"], lw=1)
        ax.axvline(200, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.plot(0, start * 100, clip_on=False, **marker("checkpoint"))
        ax.set_xlim(0, 400)
        ax.set_ylim(25, 70)
        ax.set_xticks(range(0, 401, 100))
        ax.set_xlabel("training iteration")
        ax.set_title(f"({'ab'[col]}) {name}: validation arrivals per seed", loc="left", fontsize=9, color=INK_2)
        if col == 0:
            ax.set_ylabel("trains arrived (%)")
            ax.text(212, heuristic * 100 + 1.5, f"heuristic {heuristic:.0%}", fontsize=7.5, color=INK_2)
            ax.annotate(f"start: checkpoint {start:.0%}", (0, start * 100), xytext=(8, -9),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
        else:
            ax.tick_params(labelleft=False)

        ax = axes[1, col]
        rows = [(seed, it) for seed in range(3) for it in (200, 400)]
        y = [8 - 3 * seed - (it == 400) for seed, it in rows]  # each seed's two bars together, a gap between seeds
        for yi, (seed, it) in zip(y, rows):
            w = trains[(trains.label == f"{name} s{seed}") & (trains.iteration == it)].departure_wait
            left = 0
            for i, (_, upper, colour, text) in enumerate(WAIT_BINS):
                lower = -1 if i == 0 else WAIT_BINS[i - 1][1]
                inside = (w > lower) & (w <= upper) if upper is not None else (w > lower) | w.isna()
                width = inside.mean() * 100
                ax.barh(yi, width, left=left, height=0.8, color=colour, edgecolor="white", linewidth=1.2)
                if width >= 9:
                    ax.text(left + 1.5, yi, f"{width:.0f}%", va="center", fontsize=7.5, color=text)
                left += width
        ax.set_yticks(y, [f"seed {seed}, it {it}" for seed, it in rows])
        if col == 1:
            ax.tick_params(labelleft=False)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of trains (%)")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"({'cd'[col]}) {name}: steps waited before departing", loc="left", fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in WAIT_BINS]
    fig.legend(handles, [f"{label} steps" if i < 3 else label for i, (label, *_) in enumerate(WAIT_BINS)], ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), handlelength=1.0, handletextpad=0.4, columnspacing=1.6,
               fontsize=8.5, title="waited after its earliest departure (D1, no disruptions, evaluation seed 0)",
               title_fontsize=8)
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
           "budget_effect": budget_effect, "dense_collapse": dense_collapse, "reward_story": reward_story, "greedy_vs_sampled": greedy_vs_sampled}
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
