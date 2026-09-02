"""
What the `first_goal` column actually contains
=============================================
The match export ships a column named `first_goal`, and the obvious reading is
"the team that opened the scoring". It is not that, and any analysis built on it
inherits a circular result. This script reproduces the check.

    python3 data_quality_audit.py

Findings it prints, all from data/scores.csv:

  1. `first_goal` never names the home team - not once in 380 matches.
  2. It is blank in 131 matches, and blank almost exactly when the away team
     failed to score (99.5% agreement, 2 exceptions).
  3. So it records WHETHER THE AWAY TEAM SCORED, not who scored first.
  4. Recovering the blanks by shutout - if one side scored nothing, the other
     scored first - is a sound deduction on its own terms. But it can only ever
     reach matches where the away side was kept scoreless, so "home scored
     first" exists in the data exclusively for home shutouts, and the resulting
     98% home win rate restates that selection rather than measuring anything.
     The failure is selection, not arithmetic.
"""

import numpy as np
import pandas as pd

d = pd.read_csv("data/scores.csv", low_memory=False)
d["home"] = d["Home"].astype(str).str.strip()
d["away"] = d["Away"].astype(str).str.strip()
d["fg"] = d["first_goal"].astype(str).str.strip()
d["result"] = np.sign(d["home_score"] - d["away_score"])
blank = d["first_goal"].isna()

print(f"matches: {len(d)}\n")

print("1. who does `first_goal` ever name?")
print(f"   the home team : {int((d.fg == d.home).sum())}")
print(f"   the away team : {int((d.fg == d.away).sum())}")
print(f"   blank         : {int(blank.sum())}\n")

print("2. is it blank exactly when the away team was kept scoreless?")
agreement = (blank == (d["away_score"] == 0)).mean()
print(pd.crosstab(blank, d["away_score"] == 0,
                  rownames=["first_goal blank"], colnames=["away scored 0"]).to_string())
print(f"   agreement: {agreement:.1%}")
exceptions = d[blank != (d["away_score"] == 0)]
print(f"   exceptions: {len(exceptions)}")
if len(exceptions):
    print(exceptions[["Wk", "Home", "home_score", "away_score", "Away"]]
          .to_string(index=False, header=True))

print("\n3. what happens if a blank is read as \"the home team scored first\"?")
played = d[~((d.home_score == 0) & (d.away_score == 0))]   # 0-0 draws carry no first goal
imputed = played[played["first_goal"].isna()]
w = int((imputed.result == 1).sum())
dr = int((imputed.result == 0).sum())
l = int((imputed.result == -1).sum())
print(f"   matches selected: {len(imputed)}")
print(f"   home record: {w} W, {dr} D, {l} L  ->  {w / len(imputed):.1%} home win rate")
print(f"   of these, {int((imputed.away_score == 0).sum())} are matches the away team "
      "failed to score in.")
print("   Recovering a shutout's first scorer is a sound deduction, but it only\n"
      "   reaches matches the away side failed to score in. A team that concedes\n"
      "   nothing at home does not lose, so this restates the selection rather\n"
      "   than measuring a first-goal effect.")

print("\n4. the real first-goal rate is not recoverable here")
print(f"   matches the export attributes to the home side: "
      f"{(d.fg == d.home).sum()} of {len(d)}")
print(f"   so every \"home scored first\" in the cleaned file - "
      f"{len(imputed) / len(played):.1%} of matches - comes from the blank-fill,")
print("   against a true first-goal-at-home rate of roughly 55% across leagues.")
drawn = played[played.result == 0]
print(f"   drawn matches labelled \"away scored first\": "
      f"{int((drawn.fg == drawn.away).sum())} of {len(drawn)}")
print("   Goal timings are absent from this export, so the first-goal question\n"
      "   cannot be answered with it. The analysis in the README asks what this\n"
      "   data does support instead.")
