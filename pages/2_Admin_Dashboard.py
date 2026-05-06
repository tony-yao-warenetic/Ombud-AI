"""Admin dashboard for the Ombuds analytics module.

Reads only from the anonymized `analytics_sessions` Firestore collection.
On load, lazily classifies any sessions that have been idle and unanalyzed
(one `gpt-4o-mini` call per session). Gated by a shared password from
`st.secrets["admin_password"]` or `ADMIN_PASSWORD` env var.
"""

import os
from datetime import timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analytics import TOPICS, classify_pending_sessions, compute_dashboard
from firebase_client import init_firebase

load_dotenv()

st.set_page_config(page_title="Ombuds Analytics", page_icon="📊", layout="wide")

if "admin_authed" not in st.session_state:
    st.session_state.admin_authed = False

if not st.session_state.admin_authed:
    st.title("📊 Ombuds Analytics — Admin")
    st.caption("This dashboard shows anonymized aggregates only. No raw conversation text is accessible.")
    pwd = st.text_input("Admin password", type="password")
    if pwd:
        try:
            expected = st.secrets.get("admin_password")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            expected = None
        if not expected:
            expected = os.environ.get("ADMIN_PASSWORD")
        if expected and pwd == expected:
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


db = init_firebase()

st.title("📊 Ombuds Analytics")
st.caption("Anonymized aggregates. No raw chat text is stored or shown here.")


with st.spinner("Classifying recent sessions…"):
    classified_now = classify_pending_sessions(db)
if classified_now:
    st.caption(f"Classified {classified_now} new session(s) on this load.")
    # Bust the cached compute_dashboard so newly-classified sessions appear immediately.
    st.cache_data.clear()


window_label = st.selectbox("Window", ["Last 7 days", "Last 30 days"], index=0)
window_days = 7 if window_label == "Last 7 days" else 30


@st.cache_data(ttl=300, show_spinner="Loading analytics…")
def _load(window_days: int):
    return compute_dashboard(init_firebase(), window_days=window_days)


stats = _load(window_days)

if stats.get("unanalyzed_count"):
    st.caption(
        f"Pending classification: {stats['unanalyzed_count']} session(s) still active "
        "(idle < 10 min). They'll be classified on the next dashboard load after they go idle."
    )

# ---------- Volume metrics row ----------
c1, c2, c3 = st.columns(3)
c1.metric("Total sessions", stats["total_sessions"])
c2.metric("Avg turns / session", f"{stats['avg_turns_per_session']:.1f}")
c3.metric("AI-resolved rate", f"{stats['resolution_rate'] * 100:.0f}%")

st.divider()

# ---------- Alerts ----------
st.subheader("Proactive alerts")
if stats["alerts"]:
    for a in stats["alerts"]:
        st.warning(a)
else:
    st.info("No alerts triggered for this window.")

st.divider()

# ---------- Top topics + sentiment/urgency ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Top topics")
    topic_counts = stats["topic_counts"]
    if topic_counts:
        top5 = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        df_top = pd.DataFrame(top5, columns=["Topic", "Sessions"]).set_index("Topic")
        st.bar_chart(df_top)
    else:
        st.caption("No classified sessions in window.")

with right:
    st.subheader("Sentiment")
    sc = stats["sentiment_counts"]
    if sc:
        df_s = pd.DataFrame([sc]).T.rename(columns={0: "Sessions"})
        st.bar_chart(df_s)
    else:
        st.caption("—")

    st.subheader("Urgency")
    uc = stats["urgency_counts"]
    if uc:
        df_u = pd.DataFrame([uc]).T.rename(columns={0: "Sessions"})
        st.bar_chart(df_u)
    else:
        st.caption("—")

st.divider()

# ---------- Time-series ----------
st.subheader("Sessions per day, by topic")
ts = stats["time_series"]
if ts:
    rows = []
    for day, topic_map in ts.items():
        for topic in TOPICS:
            rows.append({"day": day, "topic": topic, "count": topic_map.get(topic, 0)})
    df_ts = pd.DataFrame(rows)
    pivot = df_ts.pivot(index="day", columns="topic", values="count").fillna(0).sort_index()
    nonzero_cols = [c for c in pivot.columns if pivot[c].sum() > 0]
    pivot = pivot[nonzero_cols] if nonzero_cols else pivot
    st.line_chart(pivot)
else:
    st.caption("No data in window.")

st.divider()

# ---------- Policy friction ----------
st.subheader("Policy friction")
st.caption("How often each policy doc was retrieved during a session.")
sc = stats["source_counts"]
total_policy = stats["policy_retrieval_count"]
if sc and total_policy:
    rows = [
        {
            "Policy doc": k,
            "Sessions touched": v,
            "% of policy-touching sessions": f"{(v / total_policy) * 100:.0f}%",
        }
        for k, v in sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
    ]
    st.table(pd.DataFrame(rows))
else:
    st.caption("No policy retrievals in window.")

st.divider()

# ---------- Recent themes ----------
st.subheader("Recent abstract themes")
st.caption("Anonymized paraphrases generated by the classifier — never raw user text.")
themes = stats["recent_themes"]
if themes:
    rows = [
        {
            "When": t["ts"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M") if t["ts"] else "—",
            "Topic": t["topic"],
            "Sentiment": t["sentiment"],
            "Urgency": t["urgency"],
            "Theme": t["theme"],
        }
        for t in themes
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("No classified sessions in window.")

st.divider()

if st.button("Sign out"):
    st.session_state.admin_authed = False
    st.rerun()
