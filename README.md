# Ombud-AI

An AI ombuds assistant: a confidential, neutral first stop for employees with
workplace questions or concerns. It answers questions about workplace policy
from a set of policy PDFs, and when someone wants to talk to a person it
either gives them the human ombuds' contact details or, with their consent,
writes a handoff report for them.

It is a Streamlit chat app backed by a LangChain agent on OpenAI, with
conversations stored in Firestore. A password-protected admin dashboard shows
anonymized trends across sessions.

## What it does

- **Answers policy questions** using retrieval over the bundled policies:
  salary, maternity leave, fitness for duty, sick time pay, and harassment.
- **Stays neutral.** The assistant never gives legal advice, never labels
  conduct as harassment or discrimination, avoids collecting personal
  information, and directs anyone who mentions physical harm to law
  enforcement.
- **Hands off to a human.** It can share the ombuds office's contact
  information, or, once the user agrees, generate a handoff report covering
  their preferred name, what happened, and what they need.
- **Resumable sessions.** Each conversation has an ID kept in the URL
  (`?session_id=...`), so a user can bookmark it or enter the ID later to pick
  up where they left off.
- **Privacy-first analytics.** The admin dashboard shows topic, sentiment and
  urgency trends, spikes and alerts. It reads only anonymized labels and never
  the raw conversation text.

## Getting started

Requires Python 3.10+, an OpenAI API key, and a Firebase project with
Firestore enabled.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=sk-...
ADMIN_PASSWORD=choose-a-password   # for the admin dashboard
```

For Firestore, download a service account key from the Firebase console and
save it as `firebase-key.json` in the repo root.

`.env` and `firebase-key.json` are gitignored. Never commit them.

### Run

```bash
streamlit run ombud_ai.py
```

The chat opens at http://localhost:8501. The admin dashboard is listed as
**Admin Dashboard** in the sidebar.

## Deploying to Streamlit Cloud

Streamlit Cloud has no local files, so put the credentials in the app's
secrets (`.streamlit/secrets.toml` locally):

```toml
OPENAI_API_KEY = "sk-..."
admin_password = "choose-a-password"

[firebase]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
# ...the rest of the fields from firebase-key.json
```

`init_firebase()` uses `st.secrets["firebase"]` when it is present and falls
back to `firebase-key.json` otherwise.

## Project layout

| Path | Purpose |
|------|---------|
| `ombud_ai.py` | The chat app: session handling, the RAG index, the agent and its tools, and the guardrail prompts |
| `pages/2_Admin_Dashboard.py` | Password-gated analytics dashboard |
| `analytics.py` | Session rollups, classification (`gpt-4o-mini`), and dashboard stats |
| `firebase_client.py` | Firebase initialization (local key file or Streamlit secrets) |
| `*.pdf` | The policy documents the assistant answers from |

### Firestore collections

- `ombuds_sessions/{session_id}`: the full message history of a conversation.
- `analytics_sessions/{session_id}`: anonymized per-session metadata and
  labels (topic, sentiment, urgency, a short abstract theme). Sessions are
  classified when the dashboard loads, once they have been idle for 30
  minutes.

## How it works

1. On startup, the policy PDFs are split into 1,000-character chunks and
   embedded with `text-embedding-3-large` into an in-memory vector store. It
   is cached per process and rebuilt on a cold start.
2. Each user message goes to a LangChain agent (`gpt-4o-mini`) with three
   tools:
   - `retrieve_context`: searches the policy documents.
   - `get_ombuds_contact`: returns the human ombuds' contact details.
   - `connect_to_human_ombuds`: writes a handoff report. It is used only after
     the user agrees to one.
3. Guardrails are applied twice: in the agent's system prompt, and as
   instructions appended to the user's latest message on every turn. If you
   change the assistant's tone or rules, update both.
4. After each reply, the conversation is saved to Firestore and the analytics
   rollup is updated in a background thread.

## Adding a policy

Put the PDF in the repo root and add its filename to the `file_paths` list in
`setup_ombud_system()` in `ombud_ai.py`. Documents are not discovered
automatically.
