import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Keep these .pkl files in the same folder as app.py.
BASE_DIR = Path(__file__).resolve().parent

le_team = joblib.load(BASE_DIR / "le_team.pkl")
le_venue = joblib.load(BASE_DIR / "le_venue.pkl")
model = joblib.load(BASE_DIR / "random_forest_modelfinal.pkl")

teams = le_team.classes_.tolist()
venues = le_venue.classes_.tolist()

st.set_page_config(page_title="Cricket IQ | Win Probability", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #001F3F; color: white; }
.title-text { font-size: 32px; font-weight: 700; color: #FFD700; text-align: center; }
.section-title { font-size: 26px; font-weight: 700; color: #FF4136; text-align: center; }
.metric-box {
    font-size: 18px; font-weight: 600; text-align: center; color: white;
    background-color: rgba(255,255,255,0.12); padding: 12px;
    border-radius: 10px; margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>🏏 Cricket IQ — Win Probability Prediction</div>",
            unsafe_allow_html=True)
st.sidebar.header("🏟️ Match Input Details")

inning = st.sidebar.selectbox("Innings", [1, 2])
batting_team = st.sidebar.selectbox("🏏 Batting Team", teams)
bowling_options = [team for team in teams if team != batting_team]
bowling_team = st.sidebar.selectbox("🎯 Bowling Team", bowling_options)
venue = st.sidebar.selectbox("📍 Venue", venues)

cum_runs = st.sidebar.number_input("Total Runs", min_value=0, value=54, step=1)
cum_wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=3, step=1)
overs_completed = st.sidebar.number_input(
    "Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1
)

target = st.sidebar.number_input(
    "Target Score", min_value=0, value=160, step=1
) if inning == 2 else 0

# Calculate the rate features used by the trained model.
current_run_rate = cum_runs / overs_completed if overs_completed > 0 else 0.0
overs_remaining = max(20.0 - overs_completed, 0.0)
required_run_rate = (
    max(target - cum_runs, 0) / overs_remaining
    if inning == 2 and overs_remaining > 0
    else 0.0
)

st.markdown("<div class='section-title'>📊 Match Overview</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-box'>🏏 Batting Team<br>{batting_team}</div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-box'>🎯 Bowling Team<br>{bowling_team}</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-box'>📍 Venue<br>{venue}</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-box'>Total Runs<br>{cum_runs}</div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-box'>Wickets Lost<br>{cum_wickets}</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-box'>Overs Completed<br>{overs_completed:.1f}</div>", unsafe_allow_html=True)
with c4:
    st.markdown(
        f"<div class='metric-box'>Target<br>{target if inning == 2 else 'N/A'}</div>",
        unsafe_allow_html=True
    )

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"<div class='metric-box'>🔥 Current Run Rate<br>{current_run_rate:.2f}</div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"<div class='metric-box'>🚀 Required Run Rate<br>{required_run_rate:.2f}</div>",
        unsafe_allow_html=True
    )

# IMPORTANT: These are the exact 8 feature names and order stored in the model:
# batting_team, bowling_team, inning, cum_wickets, current_run_rate,
# required_run_rate, target, venue_canonical
input_data = np.array([[
    le_team.transform([batting_team])[0],
    le_team.transform([bowling_team])[0],
    inning,
    cum_wickets,
    current_run_rate,
    required_run_rate,
    target if inning == 2 else 0,
    le_venue.transform([venue])[0],
]], dtype=float)

if input_data.shape[1] != model.n_features_in_:
    st.error(
        f"Feature mismatch: model expects {model.n_features_in_} features, "
        f"but the app sends {input_data.shape[1]}."
    )
    st.stop()

# This assumes class 1 means the batting team wins, matching the original app's assumption.
probabilities = model.predict_proba(input_data)[0]
class_to_probability = dict(zip(model.classes_, probabilities))
batting_team_win_prob = float(class_to_probability.get(1, 0.0)) * 100
bowling_team_win_prob = float(class_to_probability.get(0, 0.0)) * 100
predicted_winner = (
    batting_team if batting_team_win_prob >= bowling_team_win_prob else bowling_team
)

st.markdown("<div class='section-title'>🔮 Win Probability Prediction</div>",
            unsafe_allow_html=True)

left, right = st.columns([1, 2])
with left:
    st.markdown(
        f"<div class='metric-box' style='color:#FFD700'>🏏 {batting_team} Winning Probability: "
        f"{batting_team_win_prob:.2f}%</div>", unsafe_allow_html=True
    )
    st.progress(min(max(batting_team_win_prob / 100, 0.0), 1.0))
    st.markdown(
        f"<div class='metric-box' style='color:#6ee344'>🎯 {bowling_team} Winning Probability: "
        f"{bowling_team_win_prob:.2f}%</div>", unsafe_allow_html=True
    )
    st.progress(min(max(bowling_team_win_prob / 100, 0.0), 1.0))

with right:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Win Probability"], y=[batting_team_win_prob], name=batting_team,
        marker_color="#de4c3c"
    ))
    fig.add_trace(go.Bar(
        x=["Win Probability"], y=[bowling_team_win_prob], name=bowling_team,
        marker_color="#6ee344"
    ))
    fig.update_layout(
        barmode="stack", title="Win Probability Distribution",
        yaxis_title="Probability (%)", yaxis=dict(range=[0, 100]),
        height=280, margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="#000000", paper_bgcolor="#000000",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"<h3 style='color:#28A745;text-align:center'>🏆 Predicted Winner: "
    f"{predicted_winner}</h3>", unsafe_allow_html=True
)
