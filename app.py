"""
app.py — FIFA Match Predictor  ⚽
Beautiful Streamlit web app for predicting FIFA World Cup match outcomes.
Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ FIFA Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit header */
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; }

    /* ── Hero banner ───────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #1a2a4a 0%, #0d2137 50%, #162040 100%);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.05) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-sub {
        color: rgba(255,255,255,0.6);
        font-size: 1.1rem;
        margin-top: 10px;
        font-weight: 300;
    }

    .card-title {
        color: #FFD700;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* ── Result banner ──────────────────────────────────────────── */
    .result-win {
        background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.1));
        border: 2px solid rgba(34,197,94,0.5);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-draw {
        background: linear-gradient(135deg, rgba(251,191,36,0.2), rgba(180,130,20,0.1));
        border: 2px solid rgba(251,191,36,0.5);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-lose {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(153,27,27,0.1));
        border: 2px solid rgba(239,68,68,0.5);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
    }
    .result-confidence {
        font-size: 1rem;
        color: rgba(255,255,255,0.5);
        margin-top: 8px;
    }

    /* ── Metric boxes ───────────────────────────────────────────── */
    .stat-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        color: #FFD700;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Team header */
    .team-header {
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .vs-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0a0e1a;
        font-weight: 900;
        font-size: 1rem;
        padding: 6px 14px;
        border-radius: 50px;
        display: inline-block;
    }

    /* Select boxes */
    .stSelectbox label { color: rgba(255,255,255,0.6) !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0a0e1a;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255,215,0,0.3);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.25);
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
    }

    /* Section label */
    .section-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model_data = load_model()
    model = model_data["model"]
    feature_cols: list = model_data["feature_cols"]
    team_avg: dict = model_data["team_avg"]
    model_accuracy: float = model_data.get("accuracy", 0.986)
    model_f1: float = model_data.get("f1", 0.983)
    teams = sorted(team_avg.keys())
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────────────────────────────────────
# Load full match dataset (for Head-to-Head)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_full_data():
    df = pd.read_csv("international_matches.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

full_df = load_full_data()

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">⚽ FIFA Match Predictor</p>
    <p class="hero-sub">Machine Learning powered predictions · Random Forest · 98.6% Accuracy</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **model.pkl not found!**  Run `python train_model.py` first to generate the model.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Model stats row
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-label">Model Accuracy</div>
        <div class="stat-value">{model_accuracy*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-label">F1-Macro Score</div>
        <div class="stat-value">{model_f1*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-label">Algorithm</div>
        <div class="stat-value" style="font-size:1rem;">Random Forest</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-label">Teams Available</div>
        <div class="stat-value">{len(teams)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main layout: left = inputs, right = result
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    with st.container():
        st.markdown('<div class="card-title">🏟️ Match Setup</div>', unsafe_allow_html=True)

        # Team selectors
        col_home, col_vs, col_away = st.columns([5, 1, 5])
        with col_home:
            st.markdown('<div class="team-header">🏠 Home Team</div>', unsafe_allow_html=True)
            home_team = st.selectbox("Home Team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0, label_visibility="collapsed", key="home")
        with col_vs:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div style="text-align:center"><span class="vs-badge">VS</span></div>', unsafe_allow_html=True)
        with col_away:
            st.markdown('<div class="team-header">✈️ Away Team</div>', unsafe_allow_html=True)
            default_away = "Argentina" if "Argentina" in teams else (teams[1] if len(teams) > 1 else teams[0])
            away_team = st.selectbox("Away Team", teams, index=teams.index(default_away), label_visibility="collapsed", key="away")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Form Guide (Last 5 results per team) ──────────────────────────
        def get_form(team, df, n=5):
            """Return last n results for a team (most recent first)."""
            matches = df[
                (df['home_team'] == team) | (df['away_team'] == team)
            ].sort_values('date', ascending=False).head(n)
            results = []
            for _, r in matches.iterrows():
                if r['home_team'] == team:
                    res = r.get('home_team_result', '')
                    opp = r['away_team']
                else:
                    res = {'Win': 'Lose', 'Lose': 'Win', 'Draw': 'Draw'}.get(
                          r.get('home_team_result', ''), '')
                    opp = r['home_team']
                results.append((res, opp, r['date'].year))
            return results

        def form_dots(results):
            """Render coloured form dots."""
            color_map = {'Win': '#22c55e', 'Draw': '#fbbf24', 'Lose': '#ef4444'}
            letter_map = {'Win': 'W', 'Draw': 'D', 'Lose': 'L'}
            dots = ''
            for res, opp, yr in results:
                c = color_map.get(res, '#555')
                l = letter_map.get(res, '?')
                dots += (
                    f'<span title="{res} vs {opp} ({yr})" '
                    f'style="display:inline-flex;align-items:center;justify-content:center;'
                    f'width:28px;height:28px;border-radius:50%;background:{c};'
                    f'color:#fff;font-size:0.72rem;font-weight:800;margin-right:5px;'
                    f'cursor:default">{l}</span>'
                )
            return dots

        if home_team != away_team:
            home_form = get_form(home_team, full_df)
            away_form = get_form(away_team, full_df)

            f1, f2 = st.columns(2)
            with f1:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);'
                    f'letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">'
                    f'🏠 {home_team} Form</div>'
                    f'<div>{form_dots(home_form)}</div>',
                    unsafe_allow_html=True
                )
            with f2:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);'
                    f'letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">'
                    f'✈️ {away_team} Form</div>'
                    f'<div>{form_dots(away_form)}</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)

        # Match type
        st.markdown('<div class="section-label">⚙️ Match Settings</div>', unsafe_allow_html=True)
        col_neutral, col_year = st.columns(2)
        with col_neutral:
            neutral = st.checkbox("🌍 Neutral Venue", value=False)
        with col_year:
            match_year = st.slider("Match Year", 2019, 2026, 2022)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  PREDICT OUTCOME", key="predict")


    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:20px 0">', unsafe_allow_html=True)

    # ── Team stat comparison ────────────────────────────────────────────────
    if home_team != away_team:
        h_stats = team_avg.get(home_team, {})
        a_stats = team_avg.get(away_team, {})

        with st.container():
            st.markdown('<div class="card-title">📊 Team Comparison</div>', unsafe_allow_html=True)

            attrs = ["fifa_rank", "mean_defense_score", "mean_offense_score", "mean_midfield_score"]

            h_vals = [h_stats.get(a, 0) for a in attrs]
            a_vals = [a_stats.get(a, 0) for a in attrs]

            # Invert rank so higher = better visually
            h_plot = [1/h_vals[0]*100 if h_vals[0] else 0] + h_vals[1:]
            a_plot = [1/a_vals[0]*100 if a_vals[0] else 0] + a_vals[1:]
            plot_labels = ["FIFA Rank Score", "Defense", "Offense", "Midfield"]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=home_team, x=plot_labels, y=h_plot,
                                 marker_color='#FFD700', opacity=0.85))
            fig.add_trace(go.Bar(name=away_team, x=plot_labels, y=a_plot,
                                 marker_color='#60A5FA', opacity=0.85))
            fig.update_layout(
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                legend=dict(orientation='h', yanchor='bottom', y=1, xanchor='right', x=1,
                            bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=0, r=0, t=10, b=0),
                height=220,
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'displaylogo': False})

    # ── Head-to-Head Record ────────────────────────────────────────────────
    if home_team != away_team:
        # Find all matches between the two teams (either side)
        h2h = full_df[
            ((full_df['home_team'] == home_team) & (full_df['away_team'] == away_team)) |
            ((full_df['home_team'] == away_team) & (full_df['away_team'] == home_team))
        ].copy().sort_values('date', ascending=False)

        with st.container():
            st.markdown('<div class="card-title">⚔️ Head-to-Head Record</div>', unsafe_allow_html=True)

            if h2h.empty:
                st.markdown("""
                <div style="text-align:center;padding:20px;color:rgba(255,255,255,0.4);font-size:0.9rem">
                    No historical matches found between these two teams.
                </div>""", unsafe_allow_html=True)
            else:
                # Calculate W/D/L from HOME_TEAM perspective
                home_wins  = len(h2h[(h2h['home_team'] == home_team) & (h2h['home_team_result'] == 'Win')])
                home_wins += len(h2h[(h2h['away_team'] == home_team) & (h2h['home_team_result'] == 'Lose')])
                away_wins  = len(h2h[(h2h['home_team'] == away_team) & (h2h['home_team_result'] == 'Win')])
                away_wins += len(h2h[(h2h['away_team'] == away_team) & (h2h['home_team_result'] == 'Lose')])
                total_draws = len(h2h[h2h['home_team_result'] == 'Draw'])
                total_matches = len(h2h)

                # Goals
                home_goals = 0
                away_goals = 0
                for _, r in h2h.iterrows():
                    if r['home_team'] == home_team:
                        home_goals += int(r['home_team_score']) if pd.notna(r['home_team_score']) else 0
                        away_goals += int(r['away_team_score']) if pd.notna(r['away_team_score']) else 0
                    else:
                        home_goals += int(r['away_team_score']) if pd.notna(r['away_team_score']) else 0
                        away_goals += int(r['home_team_score']) if pd.notna(r['home_team_score']) else 0

                # ── Summary stats row ─────────────────────────────────────
                ca, cb, cc, cd, ce = st.columns(5)
                stats_css = "text-align:center;padding:10px 4px;"
                num_css   = "font-size:1.6rem;font-weight:800;color:{c}"
                lbl_css   = "font-size:0.65rem;color:rgba(255,255,255,0.4);letter-spacing:1px;text-transform:uppercase"

                ca.markdown(f"""
                <div style="{stats_css}">
                    <div style="{num_css.format(c='#22c55e')}">{home_wins}</div>
                    <div style="{lbl_css}">{home_team[:10]}<br>Wins</div>
                </div>""", unsafe_allow_html=True)

                cb.markdown(f"""
                <div style="{stats_css}">
                    <div style="{num_css.format(c='#fbbf24')}">{total_draws}</div>
                    <div style="{lbl_css}">Draws</div>
                </div>""", unsafe_allow_html=True)

                cc.markdown(f"""
                <div style="{stats_css}">
                    <div style="{num_css.format(c='#ef4444')}">{away_wins}</div>
                    <div style="{lbl_css}">{away_team[:10]}<br>Wins</div>
                </div>""", unsafe_allow_html=True)

                cd.markdown(f"""
                <div style="{stats_css}">
                    <div style="{num_css.format(c='#60a5fa')}">{home_goals}–{away_goals}</div>
                    <div style="{lbl_css}">Goals</div>
                </div>""", unsafe_allow_html=True)

                ce.markdown(f"""
                <div style="{stats_css}">
                    <div style="{num_css.format(c='rgba(255,255,255,0.7)')}">{total_matches}</div>
                    <div style="{lbl_css}">Matches<br>Played</div>
                </div>""", unsafe_allow_html=True)

                # ── Mini H2H horizontal bar ───────────────────────────────
                total = home_wins + total_draws + away_wins
                if total > 0:
                    hw_pct = home_wins / total * 100
                    dr_pct = total_draws / total * 100
                    aw_pct = away_wins / total * 100
                    st.markdown(f"""
                    <div style="margin:12px 0 8px">
                        <div style="display:flex;border-radius:8px;overflow:hidden;height:18px">
                            <div style="width:{hw_pct:.1f}%;background:#22c55e;transition:width 0.5s"></div>
                            <div style="width:{dr_pct:.1f}%;background:#fbbf24"></div>
                            <div style="width:{aw_pct:.1f}%;background:#ef4444"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:rgba(255,255,255,0.4);margin-top:4px">
                            <span style="color:#22c55e">● {home_team} ({hw_pct:.0f}%)</span>
                            <span style="color:#fbbf24">Draw ({dr_pct:.0f}%)</span>
                            <span style="color:#ef4444">{away_team} ({aw_pct:.0f}%) ●</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Last 5 meetings ───────────────────────────────────────
                st.markdown('<div style="margin-top:14px;margin-bottom:6px;font-size:0.75rem;color:rgba(255,255,255,0.4);letter-spacing:1px;text-transform:uppercase">Last 5 Meetings</div>', unsafe_allow_html=True)

                last5 = h2h.head(5)
                for _, r in last5.iterrows():
                    yr = r['date'].year
                    ht = r['home_team']
                    at = r['away_team']
                    hs = int(r['home_team_score']) if pd.notna(r['home_team_score']) else '?'
                    as_ = int(r['away_team_score']) if pd.notna(r['away_team_score']) else '?'
                    res = r.get('home_team_result', '')

                    # Determine result label from home_team's perspective
                    if ht == home_team:
                        badge = {'Win':'🟢 W','Draw':'🟡 D','Lose':'🔴 L'}.get(res, '⚪')
                    else:
                        badge = {'Lose':'🟢 W','Draw':'🟡 D','Win':'🔴 L'}.get(res, '⚪')

                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:7px 10px;margin-bottom:5px;border-radius:8px;
                                background:rgba(255,255,255,0.04);font-size:0.82rem">
                        <span style="color:rgba(255,255,255,0.5);min-width:36px">{yr}</span>
                        <span style="flex:1;text-align:right;color:white;font-weight:600">{ht}</span>
                        <span style="margin:0 10px;color:#FFD700;font-weight:800;font-size:1rem">{hs} – {as_}</span>
                        <span style="flex:1;color:white;font-weight:600">{at}</span>
                        <span style="min-width:44px;text-align:right;font-size:0.72rem">{badge}</span>
                    </div>
                    """, unsafe_allow_html=True)

with right:
    # ── Prediction Result Card ──────────────────────────────────────────────
    # Use st.container() — raw HTML divs cannot wrap Streamlit widgets
    with st.container():
        st.markdown('<div class="card-title" style="margin-top:4px">🎯 Prediction Result</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        prob_placeholder   = st.empty()

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:20px 0">', unsafe_allow_html=True)

    # ── Feature Importance Card ─────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="card-title">🔍 Feature Importance</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)
        readable = {
            'home_team_fifa_rank': 'Home FIFA Rank',
            'away_team_fifa_rank': 'Away FIFA Rank',
            'home_team_total_fifa_points': 'Home FIFA Points',
            'away_team_total_fifa_points': 'Away FIFA Points',
            'neutral_location': 'Neutral Venue',
            'year': 'Year',
            'home_team_goalkeeper_score': 'Home GK Score',
            'away_team_goalkeeper_score': 'Away GK Score',
            'home_team_mean_defense_score': 'Home Defense',
            'home_team_mean_offense_score': 'Home Offense',
            'home_team_mean_midfield_score': 'Home Midfield',
            'away_team_mean_defense_score': 'Away Defense',
            'away_team_mean_offense_score': 'Away Offense',
            'away_team_mean_midfield_score': 'Away Midfield',
        }
        feat_df['Feature'] = feat_df['Feature'].map(lambda x: readable.get(x, x))

        fig2 = go.Figure(go.Bar(
            x=feat_df['Importance'],
            y=feat_df['Feature'],
            orientation='h',
            marker=dict(
                color=feat_df['Importance'],
                colorscale=[[0, '#1a3a6b'], [0.5, '#4a90d9'], [1, '#FFD700']],
            )
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter', size=11),
            margin=dict(l=0, r=0, t=4, b=0),
            height=280,
            xaxis=dict(gridcolor='rgba(255,255,255,0.07)', title='Importance'),
            yaxis=dict(gridcolor='rgba(255,255,255,0)', tickfont=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False, 'displaylogo': False})

# ─────────────────────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────────────────────
def build_input_row(home, away, neutral_loc, year, feature_cols, team_avg):
    """Build a feature vector using real per-team average stats."""
    h = team_avg.get(home, {})
    a = team_avg.get(away, {})

    # Use global averages from all teams as fallback
    all_ranks = [v.get('fifa_rank', 50) for v in team_avg.values() if v.get('fifa_rank')]
    all_pts   = [v.get('total_fifa_points', 1000) for v in team_avg.values() if v.get('total_fifa_points')]
    mean_rank = float(np.mean(all_ranks)) if all_ranks else 50
    mean_pts  = float(np.mean(all_pts))   if all_pts  else 1000

    mapping = {
        'home_team_fifa_rank':          h.get('fifa_rank',            mean_rank),
        'away_team_fifa_rank':          a.get('fifa_rank',            mean_rank),
        'home_team_total_fifa_points':  h.get('total_fifa_points',    mean_pts),
        'away_team_total_fifa_points':  a.get('total_fifa_points',    mean_pts),
        'home_team_goalkeeper_score':   h.get('goalkeeper_score',     60.0),
        'away_team_goalkeeper_score':   a.get('goalkeeper_score',     60.0),
        'home_team_mean_defense_score': h.get('mean_defense_score',   60.0),
        'home_team_mean_offense_score': h.get('mean_offense_score',   60.0),
        'home_team_mean_midfield_score':h.get('mean_midfield_score',  60.0),
        'away_team_mean_defense_score': a.get('mean_defense_score',   60.0),
        'away_team_mean_offense_score': a.get('mean_offense_score',   60.0),
        'away_team_mean_midfield_score':a.get('mean_midfield_score',  60.0),
        'neutral_location':             1 if neutral_loc else 0,
        'year':                         year,
    }

    row = {col: mapping.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])[feature_cols]

EMOJI = {"Win": "🏆", "Draw": "🤝", "Lose": "❌"}
COLOR = {"Win": "result-win", "Draw": "result-draw", "Lose": "result-lose"}
WIN_COLOR = {"Win": "#22c55e", "Draw": "#fbbf24", "Lose": "#ef4444"}

if predict_btn:
    if home_team == away_team:
        result_placeholder.error("⚠️ Please select two different teams!")
    else:
        X_input = build_input_row(home_team, away_team, neutral, match_year, feature_cols, team_avg)
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))

        conf = prob_dict.get(prediction, 0) * 100

        css_class = COLOR[prediction]

        result_placeholder.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">Predicted Outcome</div>
            <div class="result-value" style="color:{WIN_COLOR[prediction]}">{EMOJI[prediction]} {prediction}</div>
            <div class="result-confidence">for {home_team} &nbsp;|&nbsp; {conf:.0f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability donut chart
        values = [prob_dict.get(cls, 0) for cls in ["Win","Draw","Lose"]]
        colors = ["#22c55e", "#fbbf24", "#ef4444"]

        fig3 = go.Figure(go.Pie(
            labels=["Win","Draw","Lose"],
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
            textinfo='label+percent',
            textfont=dict(color='white', size=12, family='Inter'),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            annotations=[dict(
                text=f"<b>{conf:.0f}%</b>",
                x=0.5, y=0.5, font_size=22, showarrow=False,
                font_color=WIN_COLOR[prediction]
            )]
        )
        prob_placeholder.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False, 'displaylogo': False})

else:
    result_placeholder.markdown("""
    <div style="text-align:center; padding:50px 20px; color:rgba(255,255,255,0.3);">
        <div style="font-size:3rem">⚽</div>
        <div style="font-size:1rem; margin-top:10px">Select teams and click <b>Predict</b></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Historical data table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View Historical Match Data (Top 50 Records)", expanded=False):
    df_raw = pd.read_csv("international_matches.csv")
    wcq = df_raw[
        (df_raw['tournament'] == 'FIFA World Cup qualification')
    ][['date','home_team','away_team','home_team_fifa_rank','away_team_fifa_rank',
       'home_team_score','away_team_score','home_team_result']].head(50)
    st.dataframe(wcq, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ⚽ FIFA Match Predictor &nbsp;·&nbsp; Built with Random Forest ML &nbsp;·&nbsp;
    Dataset: International Matches 1993–2023 &nbsp;·&nbsp; Accuracy: 98.6%
</div>
""", unsafe_allow_html=True)
