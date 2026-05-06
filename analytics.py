"""Privacy-first analytics — per-session classification.

The chat path makes NO LLM calls on the analytics path. It just upserts a
session-level rollup doc on each turn (cheap Firestore write). Classification
runs lazily when the admin dashboard loads: idle, unanalyzed sessions get a
single `gpt-4o-mini` call each, and the resulting labels are written back to
the same `analytics_sessions` doc.

The privacy contract: raw user/assistant text never leaves `ombuds_sessions`.
The classifier reads transcripts at classification time, but only labels —
closed-set categories plus an LLM-paraphrased abstract `theme` — are stored.

Public surface:
  Chat path (daemon thread, every turn):
    - update_session_metadata(db, session_id, tool_calls_seen)

  Dashboard path (on load):
    - classify_pending_sessions(db, idle_minutes=30, limit=50, max_chars=6000) -> int
    - compute_dashboard(db, window_days=7) -> dict
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from firebase_admin import firestore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

TOPICS = [
    "Policy Clarification",
    "Harassment/Discrimination",
    "Workplace Conflict",
    "Benefits/Compensation",
    "Leadership Concerns",
    "Leave & Time Off",
    "Performance & Discipline",
    "Safety",
    "Other",
]

SENTIMENTS = ["neutral", "frustrated", "distressed"]
URGENCIES = ["low", "medium", "high"]

_CLASSIFIER_SYSTEM = (
    "You classify an anonymous workplace conversation. The user may have asked "
    "several related questions across the session. Return ONLY a JSON object "
    "with these exact keys:\n"
    f"- topic: one of {TOPICS}\n"
    f"- sentiment: one of {SENTIMENTS} (the user's overall affect)\n"
    f"- urgency: one of {URGENCIES} (urgency of the underlying concern)\n"
    "- theme: a paraphrase, 16 words or fewer, that names the underlying concern. "
    "Do NOT include names, dates, locations, employer details, quoted text, "
    "or any identifying details. Generic phrasing only."
)

_classifier = None


def _get_classifier() -> ChatOpenAI:
    global _classifier
    if _classifier is None:
        _classifier = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _classifier


def classify_session(transcript_text: str) -> dict | None:
    """Classify a whole session transcript. Returns dict or None on failure."""
    if not transcript_text or not transcript_text.strip():
        return None
    try:
        llm = _get_classifier()
        resp = llm.invoke([
            SystemMessage(content=_CLASSIFIER_SYSTEM),
            HumanMessage(
                content=(
                    'Conversation (user messages only, oldest first):\n'
                    f'"""{transcript_text}"""'
                )
            ),
        ])
        parsed = json.loads(resp.content)
    except Exception as e:
        print(f"[analytics] classify_session error: {e}", file=sys.stderr)
        return None

    required = {"topic", "sentiment", "urgency", "theme"}
    if not isinstance(parsed, dict) or not required.issubset(parsed.keys()):
        return None

    topic = parsed["topic"] if parsed["topic"] in TOPICS else "Other"
    sentiment = parsed["sentiment"] if parsed["sentiment"] in SENTIMENTS else "neutral"
    urgency = parsed["urgency"] if parsed["urgency"] in URGENCIES else "low"
    theme = parsed["theme"]
    if not isinstance(theme, str) or len(theme.split()) > 24:
        return None

    return {
        "topic": topic,
        "sentiment": sentiment,
        "urgency": urgency,
        "theme": theme.strip(),
    }


def update_session_metadata(db, session_id: str, tool_calls_seen: list) -> None:
    """Daemon-thread entry, called from the chat path on every turn.

    Cheap: one Firestore read + one write. No LLM call. Resets `analyzed_at`
    to None so a session that grows after being classified gets re-analyzed
    on the next dashboard load.
    """
    try:
        tool_names = [tc.get("tool") for tc in tool_calls_seen if tc.get("tool")]
        new_sources = sorted({s for tc in tool_calls_seen for s in tc.get("sources", []) if s})

        ref = db.collection("analytics_sessions").document(session_id)
        snap = ref.get()
        existing = snap.to_dict() if snap.exists else {}

        merged_tools = sorted(set(list(existing.get("tools_used", [])) + tool_names))
        merged_sources = sorted(set(list(existing.get("sources", [])) + new_sources))

        human_handoff = bool(existing.get("human_handoff", False)) or "connect_to_human_ombuds" in tool_names
        contact_shared = bool(existing.get("contact_shared", False)) or "get_ombuds_contact" in tool_names
        turn_count = int(existing.get("turn_count", 0)) + 1

        update = {
            "last_ts": firestore.SERVER_TIMESTAMP,
            "turn_count": turn_count,
            "tools_used": merged_tools,
            "sources": merged_sources,
            "human_handoff": human_handoff,
            "contact_shared": contact_shared,
            "resolved_by_ai": (not human_handoff) and (not contact_shared) and turn_count >= 2,
            "analyzed_at": None,
        }
        if not snap.exists:
            update["first_ts"] = firestore.SERVER_TIMESTAMP

        ref.set(update, merge=True)
    except Exception as e:
        print(f"[analytics] update_session_metadata failed for {session_id}: {e}", file=sys.stderr)


def _ts_to_dt(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def _fetch_user_transcript(db, session_id: str, max_chars: int) -> str | None:
    """Read user-only messages from `ombuds_sessions` and join into a transcript."""
    try:
        doc = db.collection("ombuds_sessions").document(session_id).get()
    except Exception as e:
        print(f"[analytics] failed reading ombuds_sessions/{session_id}: {e}", file=sys.stderr)
        return None
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    user_lines: list[str] = []
    for m in data.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                user_lines.append(content.strip())
    if not user_lines:
        return None
    transcript = "\n---\n".join(user_lines)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return transcript


def classify_pending_sessions(
    db, idle_minutes: int = 10, limit: int = 50, max_chars: int = 3000
) -> int:
    """Classify sessions idle >= idle_minutes that have no analyzed_at.

    Returns the number of sessions actually classified.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=idle_minutes)

    try:
        candidates = list(
            db.collection("analytics_sessions")
              .where("last_ts", "<=", cutoff)
              .order_by("last_ts", direction=firestore.Query.DESCENDING)
              .limit(limit * 4)
              .stream()
        )
    except Exception as e:
        print(f"[analytics] classify_pending_sessions query failed: {e}", file=sys.stderr)
        return 0

    classified = 0
    for snap in candidates:
        if classified >= limit:
            break
        data = snap.to_dict() or {}
        if data.get("analyzed_at") is not None:
            continue
        sid = snap.id
        transcript = _fetch_user_transcript(db, sid, max_chars)
        if not transcript:
            continue
        cls = classify_session(transcript)
        if cls is None:
            continue
        try:
            db.collection("analytics_sessions").document(sid).set({
                "topic": cls["topic"],
                "sentiment": cls["sentiment"],
                "urgency": cls["urgency"],
                "theme": cls["theme"],
                "analyzed_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            classified += 1
        except Exception as e:
            print(f"[analytics] write of classification failed for {sid}: {e}", file=sys.stderr)

    return classified


def compute_spikes(sessions: list[dict]) -> list[dict]:
    """Detect topic spikes from per-session counts.

    Sessions are heavier than turns, so the floor drops from 5 (turns) to 3
    (sessions) for the 48h current window. 1.4x baseline ratio is unchanged.
    """
    now = datetime.now(timezone.utc)
    cutoff_48h = now - timedelta(hours=48)
    cutoff_9d = now - timedelta(days=9)

    current: dict[str, int] = {}
    prior: dict[str, int] = {}
    for s in sessions:
        ts = s.get("last_ts")
        topic = s.get("topic")
        if ts is None or topic is None or ts < cutoff_9d:
            continue
        if ts >= cutoff_48h:
            current[topic] = current.get(topic, 0) + 1
        else:
            prior[topic] = prior.get(topic, 0) + 1

    spikes: list[dict] = []
    for topic, cur in current.items():
        if cur < 3:
            continue
        baseline = prior.get(topic, 0) / 3.5
        if baseline == 0:
            if cur >= 6:
                spikes.append({"topic": topic, "current": cur, "baseline": 0.0, "pct": None})
            continue
        if cur >= baseline * 1.4:
            pct = round((cur / baseline - 1) * 100)
            spikes.append({"topic": topic, "current": cur, "baseline": baseline, "pct": pct})
    return spikes


def generate_alerts(stats: dict) -> list[str]:
    alerts: list[str] = []

    for spike in stats.get("spikes", []):
        if spike["baseline"] == 0:
            alerts.append(
                f"{spike['topic']}: {spike['current']} sessions in 48h with no prior baseline — emerging topic."
            )
        else:
            alerts.append(
                f"{spike['topic']} jumped {spike['pct']}% over 48h vs 7-day baseline."
            )

    total = stats.get("total_classified_sessions", 0)
    if total > 0:
        for topic, count in stats.get("topic_counts", {}).items():
            if count >= 3 and (count / total) >= 0.15:
                pct = round((count / total) * 100)
                alerts.append(
                    f"{pct}% of this week's sessions relate to {topic} — consider an All-Hands memo."
                )

    policy_total = stats.get("policy_retrieval_count", 0)
    if policy_total > 0:
        for src, count in stats.get("source_counts", {}).items():
            if (count / policy_total) >= 0.20 and count >= 3:
                pct = round((count / policy_total) * 100)
                alerts.append(
                    f"Policy doc {src} cited in {pct}% of policy-touching sessions — may be unclear."
                )

    return alerts


def compute_dashboard(db, window_days: int = 7) -> dict:
    now = datetime.now(timezone.utc)
    window_cutoff = now - timedelta(days=window_days)
    spike_cutoff = now - timedelta(days=9)
    earliest = min(window_cutoff, spike_cutoff)

    raw = list(db.collection("analytics_sessions").where("last_ts", ">=", earliest).stream())
    sessions: list[dict] = []
    for d in raw:
        data = d.to_dict() or {}
        data["session_id"] = d.id
        data["last_ts"] = _ts_to_dt(data.get("last_ts"))
        data["first_ts"] = _ts_to_dt(data.get("first_ts"))
        data["analyzed_at"] = _ts_to_dt(data.get("analyzed_at"))
        sessions.append(data)

    in_window = [s for s in sessions if s.get("last_ts") and s["last_ts"] >= window_cutoff]
    classified = [s for s in in_window if s.get("topic") and s.get("analyzed_at")]
    unanalyzed = [s for s in in_window if not (s.get("topic") and s.get("analyzed_at"))]

    topic_counts: dict[str, int] = {}
    sentiment_counts: dict[str, int] = {}
    urgency_counts: dict[str, int] = {}
    time_series: dict[str, dict[str, int]] = {}
    recent_themes: list[dict] = []

    for s in classified:
        topic = s.get("topic") or "Other"
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        sentiment_counts[s.get("sentiment", "neutral")] = sentiment_counts.get(s.get("sentiment", "neutral"), 0) + 1
        urgency_counts[s.get("urgency", "low")] = urgency_counts.get(s.get("urgency", "low"), 0) + 1

        ts = s.get("last_ts")
        if ts:
            day = ts.date().isoformat()
            time_series.setdefault(day, {}).setdefault(topic, 0)
            time_series[day][topic] += 1

        if s.get("theme"):
            recent_themes.append({
                "ts": ts,
                "topic": topic,
                "sentiment": s.get("sentiment"),
                "urgency": s.get("urgency"),
                "theme": s.get("theme"),
            })

    source_counts: dict[str, int] = {}
    policy_retrieval_count = 0
    for s in in_window:
        srcs = s.get("sources") or []
        if srcs:
            policy_retrieval_count += 1
            for src in srcs:
                source_counts[src] = source_counts.get(src, 0) + 1

    recent_themes.sort(key=lambda r: r["ts"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    recent_themes = recent_themes[:25]

    total_sessions = len(in_window)
    ai_resolved = sum(1 for s in in_window if s.get("resolved_by_ai"))
    resolution_rate = (ai_resolved / total_sessions) if total_sessions else 0.0
    avg_turns = (sum(s.get("turn_count", 0) for s in in_window) / total_sessions) if total_sessions else 0.0

    spikes = compute_spikes(sessions)

    stats = {
        "total_sessions": total_sessions,
        "total_classified_sessions": len(classified),
        "unanalyzed_count": len(unanalyzed),
        "avg_turns_per_session": avg_turns,
        "resolution_rate": resolution_rate,
        "topic_counts": topic_counts,
        "sentiment_counts": sentiment_counts,
        "urgency_counts": urgency_counts,
        "source_counts": source_counts,
        "policy_retrieval_count": policy_retrieval_count,
        "time_series": time_series,
        "spikes": spikes,
        "recent_themes": recent_themes,
    }
    stats["alerts"] = generate_alerts(stats)
    return stats
