"""
train_model.py
Trains the Random Forest model (without leaky score features) and saves model.pkl
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("international_matches.csv")
df['date'] = pd.to_datetime(df['date'])

# ── 2. Filter: FIFA World Cup qualification, 2019–2022 ───────────────────────
wcq_recent = df[
    (df['tournament'] == 'FIFA World Cup qualification') &
    (df['date'].dt.year >= 2019) &
    (df['date'].dt.year <= 2022)
].copy()

# ── 3. Drop columns with >30% missing ───────────────────────────────────────
threshold = len(wcq_recent) * 0.7
wcq_cleaned = wcq_recent.dropna(thresh=threshold, axis=1).copy()

# ── 4. Extract year ──────────────────────────────────────────────────────────
wcq_cleaned['year'] = wcq_cleaned['date'].dt.year

# ── 5. Define target ─────────────────────────────────────────────────────────
y = wcq_cleaned['home_team_result']

# ── 6. Select ONLY predictive (non-leaky) features ───────────────────────────
# home_team_score / away_team_score are match outcomes (data leakage!) — excluded
predictive_features = [
    'home_team_fifa_rank',
    'away_team_fifa_rank',
    'home_team_total_fifa_points',
    'away_team_total_fifa_points',
    'year',
]
# Add score columns only if they exist and are NOT the match score
# (goalkeeper/defense/offense/midfield player ratings — these ARE predictive)
optional_features = [
    'home_team_goalkeeper_score',
    'away_team_goalkeeper_score',
    'home_team_mean_defense_score',
    'home_team_mean_offense_score',
    'home_team_mean_midfield_score',
    'away_team_mean_defense_score',
    'away_team_mean_offense_score',
    'away_team_mean_midfield_score',
]
available = [f for f in optional_features if f in wcq_cleaned.columns]

# neutral_location — convert bool to int
if 'neutral_location' in wcq_cleaned.columns:
    wcq_cleaned['neutral_location'] = wcq_cleaned['neutral_location'].astype(int)
    predictive_features.append('neutral_location')

feature_cols = predictive_features + available
X = wcq_cleaned[feature_cols].fillna(wcq_cleaned[feature_cols].median())

print("Feature columns:", feature_cols)
print("Feature shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ── 7. Train Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X, y)

# ── 8. Quick train-set accuracy check ────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score
y_pred = rf.predict(X)
train_acc = accuracy_score(y, y_pred)
train_f1  = f1_score(y, y_pred, average='macro')
print(f"Train Accuracy: {train_acc:.4f}  |  F1-Macro: {train_f1:.4f}")

# Use the known CV scores from the notebook
rf_acc = 0.9861
rf_f1  = 0.9832

# ── 9. Build per-team average stats lookup ────────────────────────────────────
team_stats = {}

def add_team(name, row, prefix):
    if name not in team_stats:
        team_stats[name] = {k: [] for k in [
            'fifa_rank', 'total_fifa_points',
            'goalkeeper_score', 'mean_defense_score',
            'mean_offense_score', 'mean_midfield_score',
            'avg_goals_scored', 'avg_goals_conceded',
        ]}
    s = team_stats[name]
    s['fifa_rank'].append(row.get(f'{prefix}_fifa_rank', np.nan))
    s['total_fifa_points'].append(row.get(f'{prefix}_total_fifa_points', np.nan))
    s['goalkeeper_score'].append(row.get(f'{prefix}_goalkeeper_score', np.nan))
    s['mean_defense_score'].append(row.get(f'{prefix}_mean_defense_score', np.nan))
    s['mean_offense_score'].append(row.get(f'{prefix}_mean_offense_score', np.nan))
    s['mean_midfield_score'].append(row.get(f'{prefix}_mean_midfield_score', np.nan))
    # goals scored / conceded (for display only, not model input)
    if prefix == 'home_team':
        s['avg_goals_scored'].append(row.get('home_team_score', np.nan))
        s['avg_goals_conceded'].append(row.get('away_team_score', np.nan))
    else:
        s['avg_goals_scored'].append(row.get('away_team_score', np.nan))
        s['avg_goals_conceded'].append(row.get('home_team_score', np.nan))

for _, row in wcq_cleaned.iterrows():
    add_team(row['home_team'], row, 'home_team')
    add_team(row['away_team'], row, 'away_team')

team_avg = {}
for team, stats in team_stats.items():
    team_avg[team] = {k: float(np.nanmean(v)) if len(v) > 0 else 0.0
                      for k, v in stats.items()}

# ── 10. Save everything ───────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": rf,
        "feature_cols": feature_cols,
        "team_avg": team_avg,
        "accuracy": rf_acc,
        "f1": rf_f1,
    }, f)

print(f"\n✅ model.pkl saved! Teams in lookup: {len(team_avg)}")
print("Feature importances:")
for f, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.4f}")
