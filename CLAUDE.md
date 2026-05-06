# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install deps: `pip install -r requirements.txt` (use the bundled `venv/`)
- Run the app: `streamlit run ombud_ai.py`
- No test suite, linter, or build step is configured.

## Required environment

- `.env` (gitignored) supplies `OPENAI_API_KEY` — read via `python-dotenv` in `ombud_ai.py`. The sidebar references the env var; there is no UI input for it.
- `firebase-key.json` (gitignored) is the Firebase service account credential used for **local** runs. On Streamlit Cloud, the same JSON must be provided as `st.secrets["firebase"]` instead — `init_firebase()` branches on which is present.

## Architecture

The entire app lives in `ombud_ai.py`. It is a single-page Streamlit chat UI backed by a LangChain agent. Three concerns are interleaved in that file and worth understanding before editing:

**1. Session lifecycle (Firestore-backed, URL-resumable).** Sessions are identified by a UUID stored in `st.session_state.session_id` and mirrored into `st.query_params["session_id"]` so a user can bookmark/return to a conversation. The landing page (gated by `session_started`) chooses between starting fresh or resuming via an entered ID. After every turn, the full message list is persisted to Firestore at `ombuds_sessions/{session_id}` as a `messages` array of `{role, content}` dicts. On load, those dicts are reconstructed into LangChain `HumanMessage`/`AIMessage` objects.

**2. RAG over policy PDFs.** `setup_ombud_system()` (cached via `@st.cache_resource`) loads a hardcoded list of PDFs (`salary.pdf`, `maternal_leave.pdf`, `fitness_for_duty.pdf`, `sick_time_pay.pdf`, `harassment.pdf`) into an `InMemoryVectorStore` using `OpenAIEmbeddings(text-embedding-3-large)`. Splits are 1000 chars / 200 overlap. **Adding a policy means editing the `file_paths` list inside this function** — there is no auto-discovery. The vector store is in-memory and rebuilt on cold start.

**3. Agent + tools + guardrails.** A `create_agent` (LangChain) wraps `gpt-4-turbo` with three tools:
- `retrieve_context(query)` — similarity search over the PDFs (k=2).
- `get_ombuds_contact()` — returns hardcoded contact info; used when the user wants a human but hasn't agreed to a handoff report.
- `connect_to_human_ombuds(preferred_name, incident_description, user_need)` — only called after the user explicitly agrees to generate a handoff report.

Guardrails are enforced in two places that both must be kept in sync if you change tone/policy: (a) the `system_prompt` passed to `create_agent`, and (b) `enforcement_text` which is **appended to the user's last message on every turn** (not sent as a system message). On the first turn (`len(messages) <= 2`), an additional onboarding block is appended that asks the model to greet and ask for a preferred name. This double-injection pattern is intentional — changes to behavior often need to land in both spots.

## Conventions worth knowing

- Sensitive content rules in the prompts (no PII, never give legal advice, escalate physical-harm mentions to law enforcement, never label conduct as harassment/discrimination) are load-bearing for the product's positioning as a neutral ombuds resource. Treat them as requirements, not stylistic suggestions.
- Streaming uses `agent.stream(..., stream_mode="values")` and only the final `AIMessage.content` is shown — intermediate tool messages are filtered out of the UI.
- `firebase-key.json` and `.env` are gitignored; never stage them.

