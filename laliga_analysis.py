"""
Home advantage, crowds and finishing in La Liga 2023-24
======================================================
Every match of the season (380), with expected goals, attendance, venue and
referee. Reads data/scores.csv and writes four figures plus the numbers quoted
in the README.

    python3 laliga_analysis.py

  figures/home_advantage_source.png   where the home edge comes from: chances,
                                      not finishing
  figures/crowd_confound.png          attendance looks decisive until you
                                      control for who is playing
  figures/referee_noise.png           the referee spread against what random
                                      assignment produces
  figures/xg_overperformance.png      which sides out-scored their chances
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CRIMSON, SLATE, GREY, LGREY = "#9A6B1F", "#33302B", "#B9BEC4", "#ECEEF0"  # Spanish gold on warm charcoal
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": SLATE, "axes.labelcolor": SLATE, "text.color": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE, "axes.titlepad": 12,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": GREY,
})
os.makedirs("figures", exist_ok=True)


def strip(ax, keep_left=True):
    for s in ["top", "right"] + ([] if keep_left else ["left"]):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


d = pd.read_csv("data/scores.csv", low_memory=False)
d["result"] = np.sign(d["home_score"] - d["away_score"])
d["home_points"] = (d.result == 1) * 3 + (d.result == 0) * 1
d["xg_edge"] = d["xG"] - d["xG.1"]
d["att_k"] = d["Attendance"] / 1000

# Season points, built from these same matches, as a stand-in for club quality.
points = {}
for _, m in d.iterrows():
    hp, ap = (3, 0) if m.result == 1 else ((0, 3) if m.result == -1 else (1, 1))
    points[m.Home] = points.get(m.Home, 0) + hp
    points[m.Away] = points.get(m.Away, 0) + ap
table = pd.Series(points).sort_values(ascending=False)
d["quality_gap"] = d.Home.map(table) - d.Away.map(table)

print(f"La Liga 2023-24: {len(d)} matches, {d.Home.nunique()} clubs\n")

# ---------------------------------------------------------------------------
# 1. Home advantage: more chances, identical finishing
# ---------------------------------------------------------------------------
home_conv = d.home_score.sum() / d["xG"].sum()
away_conv = d.away_score.sum() / d["xG.1"].sum()
_, p_xg = stats.ttest_rel(d["xG"], d["xG.1"])

# The conversion gap is small enough that it needs a confidence interval before
# it can be called a difference at all. Bootstrap over matches.
_rng = np.random.default_rng(11)
_boot = np.array([
    (lambda s: s.home_score.sum() / s["xG"].sum() - s.away_score.sum() / s["xG.1"].sum())(
        d.iloc[_rng.integers(0, len(d), len(d))])
    for _ in range(4000)
])
conv_lo, conv_hi = np.percentile(_boot, [2.5, 97.5])

print("home advantage")
print(f"  results     home {(d.result == 1).mean():.1%} / draw {(d.result == 0).mean():.1%} "
      f"/ away {(d.result == -1).mean():.1%}")
print(f"  points/match  home {d.home_points.mean():.2f}  away "
      f"{((d.result == -1) * 3 + (d.result == 0)).mean():.2f}")
print(f"  xG            home {d['xG'].mean():.3f}  away {d['xG.1'].mean():.3f}  "
      f"(edge {d.xg_edge.mean():+.3f}, p={p_xg:.1e})")
print(f"  goals         home {d.home_score.mean():.3f}  away {d.away_score.mean():.3f}")
print(f"  goals per xG  home {home_conv:.3f}  away {away_conv:.3f}")
print(f"                difference {home_conv - away_conv:+.3f}, 95% CI "
      f"[{conv_lo:+.3f}, {conv_hi:+.3f}] -> "
      f"{'a real gap' if conv_lo > 0 or conv_hi < 0 else 'no detectable difference'}\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
x = np.arange(2)
a1.bar(x - 0.19, [d["xG"].mean(), d["xG.1"].mean()], 0.38, color=CRIMSON, zorder=3,
       label="expected goals")
a1.bar(x + 0.19, [d.home_score.mean(), d.away_score.mean()], 0.38, color=SLATE, zorder=3,
       label="goals scored")
a1.set_xticks(x)
a1.set_xticklabels(["Playing at home", "Playing away"])
a1.set_ylabel("Per match")
a1.set_ylim(0, 1.85)
a1.set_title("Chances created and goals scored", loc="left", pad=26)
a1.yaxis.grid(True, color=LGREY, zorder=0)
strip(a1)
a1.legend(frameon=False, loc="upper right", fontsize=10.5)
for xi, v in zip(x - 0.19, [d["xG"].mean(), d["xG.1"].mean()]):
    a1.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=11.5, fontweight="bold",
            color=CRIMSON)
for xi, v in zip(x + 0.19, [d.home_score.mean(), d.away_score.mean()]):
    a1.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=11.5, fontweight="bold",
            color=SLATE)

a2.bar(x, [home_conv, away_conv], 0.46, color=[CRIMSON, GREY], zorder=3)
a2.axhline(1.0, color=SLATE, lw=1.3, ls=":", zorder=4)
# No floating annotation: the reference line is explained in the subtitle instead,
# which leaves the plot area clear whatever the bar heights turn out to be.
a2.text(0, 1.045, "1.0 = scoring exactly as many goals as the chances were worth",
        transform=a2.transAxes, fontsize=10, color=SLATE, va="bottom")
a2.set_xticks(x)
a2.set_xticklabels(["Playing at home", "Playing away"])
a2.set_ylabel("Goals scored per expected goal")
a2.set_ylim(0, 1.35)
a2.set_title("Finishing relative to chance quality", loc="left", pad=26)
a2.yaxis.grid(True, color=LGREY, zorder=0)
strip(a2)
for xi, v, c in zip(x, [home_conv, away_conv], [CRIMSON, SLATE]):
    a2.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold", color=c)
fig.tight_layout()
fig.savefig("figures/home_advantage_source.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. The crowd: a confound, not a cause
# ---------------------------------------------------------------------------
d["att_q"] = pd.qcut(d.Attendance, 4, labels=["Smallest\nquarter", "2nd", "3rd",
                                              "Largest\nquarter"])
byq = d.groupby("att_q", observed=True).agg(home_ppm=("home_points", "mean"))

alone = sm.OLS(d.home_points, sm.add_constant(d[["att_k"]])).fit()
ctrl = sm.OLS(d.home_points, sm.add_constant(d[["att_k", "quality_gap"]])).fit()
alone_xg = sm.OLS(d.xg_edge, sm.add_constant(d[["att_k"]])).fit()
ctrl_xg = sm.OLS(d.xg_edge, sm.add_constant(d[["att_k", "quality_gap"]])).fit()
avg_att = d.groupby("Home").Attendance.mean()
r_conf = table.reindex(avg_att.index).corr(avg_att)

# Does a club do better when its OWN crowd is larger than its own average?
d["att_dev"] = d.Attendance - d.groupby("Home").Attendance.transform("mean")
within = sm.OLS(d.home_points, sm.add_constant(d[["att_dev"]])).fit()

print("the crowd")
print(f"  corr(club season points, its average home crowd) = {r_conf:.3f}")
print(f"  home points ~ attendance          coef {alone.params.att_k:+.4f} "
      f"per 1,000  p={alone.pvalues.att_k:.2g}")
print(f"  + club quality gap                coef {ctrl.params.att_k:+.4f} "
      f"per 1,000  p={ctrl.pvalues.att_k:.2g}")
print(f"  home xG edge ~ attendance         coef {alone_xg.params.att_k:+.4f} "
      f"p={alone_xg.pvalues.att_k:.2g}")
print(f"  + club quality gap                coef {ctrl_xg.params.att_k:+.4f} "
      f"p={ctrl_xg.pvalues.att_k:.2g}")
print(f"  within a club, vs its own average coef {within.params.att_dev * 1000:+.4f} "
      f"per 1,000  p={within.pvalues.att_dev:.3f}\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw={"width_ratios": [1.1, 1]})
cols = [GREY, GREY, GREY, CRIMSON]
a1.bar(range(4), byq.home_ppm.values, 0.6, color=cols, zorder=3)
a1.set_xticks(range(4))
a1.set_xticklabels(byq.index, fontsize=10.5, linespacing=1.4)
a1.set_ylabel("Home points per match")
a1.set_ylim(0, 2.55)
a1.set_xlabel("Matches grouped by attendance", fontsize=11)
a1.set_title("Home points by crowd size", loc="left")
a1.yaxis.grid(True, color=LGREY, zorder=0)
strip(a1)
for i, v in enumerate(byq.home_ppm.values):
    a1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=12.5, fontweight="bold",
            color=CRIMSON if i == 3 else SLATE)

labels = ["Attendance\nalone", "Attendance, holding the two\nclubs' quality constant"]
vals = [alone.params.att_k, ctrl.params.att_k]
a2.bar(range(2), vals, 0.5, color=[CRIMSON, GREY], zorder=3)
a2.axhline(0, color=SLATE, lw=1.1)
a2.set_xticks(range(2))
a2.set_xticklabels(labels, fontsize=10.5, linespacing=1.4)
a2.set_ylabel("Extra home points per 1,000 fans")
a2.set_ylim(min(vals) - 0.006, max(vals) * 1.35)
a2.set_title("The same effect, with club quality controlled", loc="left")
a2.yaxis.grid(True, color=LGREY, zorder=0)
strip(a2)
for i, (v, pv) in enumerate(zip(vals, [alone.pvalues.att_k, ctrl.pvalues.att_k])):
    sig = f"p < 0.001" if pv < 0.001 else f"p = {pv:.2f}"
    a2.text(i, v + 0.0012, f"{v:+.4f}\n{sig}", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=CRIMSON if i == 0 else SLATE, linespacing=1.5)
fig.tight_layout()
fig.savefig("figures/crowd_confound.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Referees: a spread that random assignment reproduces
# ---------------------------------------------------------------------------
ref = (d.groupby("Referee").agg(n=("home_points", "size"), ppm=("home_points", "mean"))
       .query("n >= 15").sort_values("ppm"))
observed = ref.ppm.std()
rng = np.random.default_rng(42)
null = np.array([
    d.assign(shuffled=rng.permutation(d.home_points.values))
     .groupby("Referee").agg(n=("shuffled", "size"), m=("shuffled", "mean"))
     .query("n >= 15").m.std()
    for _ in range(4000)
])
p_ref = (null >= observed).mean()
print("referees")
print(f"  {len(ref)} referees with 15+ matches, home points per match "
      f"{ref.ppm.min():.2f} to {ref.ppm.max():.2f}")
print(f"  observed spread {observed:.3f} | shuffled average {null.mean():.3f} "
      f"| p = {p_ref:.3f}\n")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})
a1.barh(range(len(ref)), ref.ppm.values, 0.68, color=GREY, zorder=3)
a1.set_yticks(range(len(ref)))
a1.set_yticklabels(ref.index, fontsize=9.5)
a1.axvline(d.home_points.mean(), color=CRIMSON, lw=1.5, ls=":", zorder=4)
a1.text(d.home_points.mean() + 0.03, -0.9, f"league average {d.home_points.mean():.2f}",
        fontsize=10, color=CRIMSON, fontweight="bold")
a1.set_xlim(0, 2.5)
a1.set_ylim(len(ref) - 0.4, -1.3)
a1.set_xlabel("Home points per match", fontsize=11)
a1.set_title("Home points by referee", loc="left", fontsize=13)
a1.xaxis.grid(True, color=LGREY, zorder=0)
strip(a1, keep_left=False)

a2.hist(null, bins=34, color=GREY, zorder=3)
a2.axvline(observed, color=CRIMSON, lw=2.2, zorder=5)
a2.annotate(f"observed\n{observed:.3f}", xy=(observed, a2.get_ylim()[1] * 0.72),
            xytext=(observed + 0.055, a2.get_ylim()[1] * 0.86), fontsize=11,
            fontweight="bold", color=CRIMSON, linespacing=1.4,
            arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4))
a2.set_xlabel("Spread across referees when results are shuffled at random", fontsize=10.5)
a2.set_ylabel("Simulations")
a2.set_title("The same spread from shuffled results", loc="left", fontsize=13)
a2.yaxis.grid(True, color=LGREY, zorder=0)
strip(a2)
a2.text(0.97, 0.42, f"{p_ref:.0%} of random shuffles\nproduce a spread this wide",
        transform=a2.transAxes, ha="right", fontsize=10.5, color=SLATE, linespacing=1.5)
fig.tight_layout()
fig.savefig("figures/referee_noise.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Finishing against expectation, by club
# ---------------------------------------------------------------------------
home = d[["Home", "home_score", "xG"]].rename(
    columns={"Home": "club", "home_score": "goals", "xG": "xg"})
away = d[["Away", "away_score", "xG.1"]].rename(
    columns={"Away": "club", "away_score": "goals", "xG.1": "xg"})
clubs = (pd.concat([home, away]).groupby("club")
         .agg(goals=("goals", "sum"), xg=("xg", "sum")))
clubs["diff"] = clubs.goals - clubs.xg
clubs = clubs.sort_values("diff")
print("finishing against expectation (goals minus xG, whole season)")
print(clubs.round(1).to_string())

# Same discipline as the referee test: is this spread larger than chance alone?
# Null = every side finishes exactly to its chance quality, goals ~ Poisson(xG).
sides = pd.concat([home, away])
rng_fin = np.random.default_rng(7)
null_spread = np.array([
    sides.assign(goals=rng_fin.poisson(sides.xg.values))
         .groupby("club").apply(lambda t: t.goals.sum() - t.xg.sum(), include_groups=False).std()
    for _ in range(4000)
])
obs_spread = clubs["diff"].std()
p_fin = (null_spread >= obs_spread).mean()
band = np.percentile(null_spread, 95)
print(f"\n  spread across clubs {obs_spread:.2f} goals | pure finishing luck produces "
      f"{null_spread.mean():.2f} on average, 95th percentile {band:.2f}")
print(f"  p = {p_fin:.3f} -> "
      f"{'beyond luck' if p_fin < 0.05 else 'NOT distinguishable from finishing luck'}\n")

fig, ax = plt.subplots(figsize=(9.8, 6.4))
colors = [CRIMSON if v > 0 else SLATE for v in clubs["diff"]]
ax.barh(range(len(clubs)), clubs["diff"].values, 0.68, color=colors, zorder=3)
ax.axvline(0, color=SLATE, lw=1.2)
ax.set_yticks(range(len(clubs)))
ax.set_yticklabels(clubs.index, fontsize=10.5)
ax.set_xlim(clubs["diff"].min() * 1.32, clubs["diff"].max() * 1.32)
ax.set_xlabel("Goals scored minus expected goals, whole season", fontsize=11)
ax.set_title("Finishing against expectation, by club", loc="left")
ax.xaxis.grid(True, color=LGREY, zorder=0)
strip(ax, keep_left=False)
for i, v in enumerate(clubs["diff"].values):
    off = 0.6 if v > 0 else -0.6
    ax.text(v + off, i, f"{v:+.1f}", va="center", ha="left" if v > 0 else "right",
            fontsize=11, fontweight="bold", color=colors[i])
# Show how far chance alone routinely carries a side, so the bars are read against
# something rather than in isolation.
lo95, hi95 = np.percentile(null_spread, 2.5), band
ax.axvspan(-1.96 * null_spread.mean(), 1.96 * null_spread.mean(), color=GREY, alpha=0.16,
           zorder=1)
ax.text(0.985, 0.30,
        "Shaded band: where 95% of clubs land in a simulated\n"
        "league where every side finishes exactly to its chances.\n"
        f"Only two clubs clear it, and {p_fin:.0%} of simulated seasons\n"
        "are at least this spread out.",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.8,
        color=SLATE, linespacing=1.6)
fig.tight_layout()
fig.savefig("figures/xg_overperformance.png", dpi=200)
plt.close(fig)

print("wrote figures/home_advantage_source.png, crowd_confound.png,")
print("      referee_noise.png, xg_overperformance.png")
