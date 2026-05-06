import os
import threading
import uuid # [NEW] For generating Session IDs
import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from analytics import update_session_metadata
from firebase_client import init_firebase

load_dotenv()

st.set_page_config(page_title="AI Ombuds Assistant", page_icon="⚖️")

db = init_firebase()

# --- 2. SESSION INITIALIZATION ---
if "session_started" not in st.session_state:
    st.session_state.session_started = False

if "session_id" in st.query_params and not st.session_state.session_started:
    st.session_state.session_id = st.query_params["session_id"]
    st.session_state.session_started = True

# --- 3. LANDING PAGE ---
# This MUST come before the sidebar and chat interface
if not st.session_state.session_started:
    st.title("⚖️ AI Ombuds Assistant")
    st.markdown("A confidential, empathetic, and neutral space to discuss workplace concerns.")
    st.divider()
    
    st.markdown("### Welcome to the AI Ombuds Assistant")
    st.markdown("Please choose how you would like to begin. Your privacy and anonymity are our top priority.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Option 1: Start Fresh")
        st.write("Begin a completely new, confidential conversation.")
        if st.button("Start New Conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.query_params["session_id"] = st.session_state.session_id
            st.session_state.session_started = True
            st.rerun()
            
    with col2:
        st.success("Option 2: Resume")
        st.write("Pick up where you left off using your anonymous ID.")
        entered_id = st.text_input("Session ID", placeholder="Enter your ID...", label_visibility="collapsed")
        
        if st.button("Load Previous Conversation", use_container_width=True):
            if entered_id.strip():
                st.session_state.session_id = entered_id.strip()
                st.query_params["session_id"] = st.session_state.session_id
                st.session_state.session_started = True
                st.rerun()
            else:
                st.warning("Please enter a valid Session ID to continue.")
                
    # st.stop() pauses the script here. The sidebar and chat WILL NOT load until a choice is made.
    st.stop() 


# ==========================================
# --- 4. MAIN APP (Runs only after starting) ---
# ==========================================
session_id = st.session_state.session_id

# Title for the main chat page
st.title("⚖️ AI Ombuds Assistant")
st.markdown("A confidential, empathetic, and neutral space to discuss workplace concerns.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    st.divider()
    st.header("Session Management")
    st.caption("Your anonymous Session ID:")
    st.code(session_id, language=None)
    
    if st.button("🚪 End Session & Start Over", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.query_params.clear()
        st.rerun()

# API Key check comes AFTER the landing page, but BEFORE the chat
if not openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

def save_chat_to_firestore(session_id, messages):
    """Saves the current conversation state to Firestore."""
    # Convert LangChain message objects into standard dictionaries for Firestore
    formatted_messages = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    # Store under the 'ombuds_sessions' collection using the session_id as the document ID
    db.collection("ombuds_sessions").document(session_id).set({"messages": formatted_messages})
    
def load_chat_from_firestore(session_id):
    """Loads a previous conversation from Firestore if it exists."""
    doc = db.collection("ombuds_sessions").document(session_id).get()
    if doc.exists:
        return doc.to_dict().get("messages", [])
    return None

@st.cache_resource(show_spinner="Initializing AI and Loading Policies...")
def setup_ombud_system():    
    model = init_chat_model("gpt-4o-mini")  # cheap default; bump to "gpt-4o" if empathy quality suffers
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)

    file_paths = [
        "salary.pdf",
        "maternal_leave.pdf",
        "fitness_for_duty.pdf",
        "sick_time_pay.pdf",
        "harassment.pdf"
    ]
    
    docs = []
    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except FileNotFoundError:
            pass # Suppressed warning for brevity

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=all_splits)

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from corporate policies to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    @tool(response_format="content_and_artifact")
    def get_ombuds_contact():
        """Use this tool immediately when the user asks to speak with a human ombuds..."""
        contact_info = {
            "office_name": "Corporate Ombuds Office",
            "phone": "1-800-555-0199",
            "email": "ombuds@company.com",
            "booking_link": "https://company.internal/ombuds/schedule"
        }
        content = "Successfully retrieved the Ombuds contact information. Please share these details with the user."
        return content, contact_info
    
    @tool(response_format="content_and_artifact")
    def connect_to_human_ombuds(preferred_name: str, incident_description: str, user_need: str):
        """Use this tool ONLY AFTER the user has explicitly said "yes" to generating a summary report..."""
        artifact = {
            "handoff_report": {
                "Preferred Name": preferred_name,
                "Incident Description": incident_description,
                "User Need": user_need
            },
            "contact_info": {
                "office_name": "Corporate Ombuds Office",
                "phone": "1-800-555-0199",
                "email": "ombuds@company.com",
            }
        }
        content = "Successfully generated the handoff report and retrieved contact info. Provide details to the user."
        return content, artifact

    tools = [retrieve_context, get_ombuds_contact, connect_to_human_ombuds]

    prompt = """
            You are a third-party AI Ombuds assistant.
            Your first goal is to establish a safe, confidential connection.

            You have access to a corporate policy retrieval tool that can search relevant policy documents. Use it when appropriate to inform your responses, but do not rely on it exclusively. Always provide thoughtful, empathetic guidance even if no specific policy applies.
            - Summarize policy language clearly.
            - Do not invent policy details. If the policy is unclear or unavailable, say so plainly.

            You have access to tools that connect user to human ombuds. Use these tools if the user asks for a human, or if the situation is highly sensitive (e.g., crime, safety threats).
            - Before you connect, ask if the user would like to generate a report to share with the human ombud. Reinforce information will not be shared. If there isn't enough information or nothing is provided, DO NOT ask the user to generate report. Just use the tool 'get_ombuds_contact'.
            - If the user agrees to generate and share report, use 'connect_to_human_ombuds'. Else if the user doesn't explicitly agree or there isn't enough context, use 'get_ombuds_contact'.

            Rules:
            - NEVER ask for more detail about specific events
            - Whenever the user mentions physical Harm or criminal Actions, inform them to contact local law-enforcement agenc immediately.
            - Use an empathetic, calm, neutral tone. DO NOT generate long paragraphs. Keep sentences short and readable.
            - Use few bullet points.
            - Preserve privacy and anonymity by default.
            - Do not ask for names, employee IDs, or other identifying details unless clearly necessary and optional.
            - Maintain clear independence from the employer, HR, legal, and management.
            - Never judge or conclude that conduct is harassment, discrimination, retaliation, misconduct, illegal, or criminal.
            - Never provide legal advice. If asked, explicitly state that you cannot provide legal advice.
            - If the matter appears serious, sensitive, urgent, or safety-related, recommend speaking with a human Ombuds, HR, or another appropriate support resource.
            - Use the corporate policy retrieval tool when policy documents are relevant.
            - Summarize policy documents clearly and accurately. Do not invent or infer missing policy details.
            """
    
    agent = create_agent(model, tools, system_prompt=prompt)
    return agent

# --- Chat Interface ---
if not openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

agent = setup_ombud_system()

# --- [NEW] Modified Chat Initialization to Load from Firestore ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Attempt to load historical messages using the active session_id
    past_messages = load_chat_from_firestore(session_id)
    
    if past_messages:
        # Reconstruct LangChain message objects from the loaded Firestore dictionaries
        for pm in past_messages:
            if pm["role"] == "user":
                st.session_state.messages.append(HumanMessage(content=pm["content"]))
            else:
                st.session_state.messages.append(AIMessage(content=pm["content"]))
    else:
        # If no history exists, append the intro text and save to Firestore
        intro_text = (
            """
            Hello! I am your AI ombuddy. Let me see how I can best help you. \n
            Before I find the best way to do that, please know that I am a completely private, anonymous resource designed to help you find information or discuss options when you have a question and aren’t quite sure who to ask in your organization. Please be assured that our conversation is completely private and inaccessible to your organization. \n
            Please also do not give me any personal or identifying information about yourself. Though NO conversations we have will ever be shared by us with anyone else, members of the Ombuddy team will occasionally view conversations to ensure the tool is as useful as it can be.
            For that reason, we prefer to keep these conversations anonymous. If you’d like to learn more about this, please look here www.ombudsprivacypolicy.com \n
            If you end up having more questions than I can answer for you, I’d highly recommend you speak with an Ombuds. An ombudsman is a completely confidential, neutral, third-party resource trained to help you navigate anything that might be concerning to you. They will only share information with your organization if you request that they do so. They are a free resource provided to you by your company.
            """
        )
        
        st.session_state.messages.append(AIMessage(content=intro_text))
        save_chat_to_firestore(session_id, st.session_state.messages)

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
        
# Accept user input
if query := st.chat_input("How can I support you today?"):
    # Add user message to state, UI, and save to Firestore
    st.session_state.messages.append(HumanMessage(content=query))
    save_chat_to_firestore(session_id, st.session_state.messages) # [NEW]
    
    with st.chat_message("user"):
        st.markdown(query)

    enforcement_text = (
        """Role: You are a calm, empathetic, and neutral ombudsman. Your primary goal is to provide supports while maintaining strict professional and safety boundaries.
            Operational Rules:
            - Privacy First: Never solicit the user’s name, contact details, or any personally identifiable information (PII).
            - Scope of Detail: Focus on high-level summaries. Do not probe for granular details, specific names, or the mechanics of events.
            - Communication Style: Use short, clear sentences. Avoid long sentences.

            Safety & Legal Guardrails:
            - Emergency Protocol: If a user mentions physical harm, violence, or criminal activity, your immediate priority is to direct them to contact local law enforcement or emergency services.
            - Non-Legal Status: You are an AI, not an attorney. Never provide legal interpretations, strategy, or advice. If legal matters arise, suggest consulting a qualified professional.
        """.strip()
    )
    
    if len(st.session_state.messages) <= 2:
        onboarding = (
            "\n\n[SYSTEM INSTRUCTION: INITIAL GREETING]\n"
            "This is the start of the conversation. Warmly introduce yourself. "
            "Ask the user what they would like to be called and if there is any information about their role or background they'd like to share to help you provide better context. "
            "Remind the user DO NOT to provide their real name or any personal information."
        )
        # Append the onboarding instructions to the enforcement text
        enforcement_text += onboarding

    messages_for_agent = st.session_state.messages.copy()
    messages_for_agent[-1] = HumanMessage(content=query + enforcement_text)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        tool_calls_seen = []

        with st.spinner("Thinking..."):
            try:
                for event in agent.stream({"messages": messages_for_agent}, stream_mode="values"):
                    last_message = event["messages"][-1]
                    if last_message.__class__.__name__ == "ToolMessage":
                        entry = {"tool": getattr(last_message, "name", None), "sources": []}
                        artifact = getattr(last_message, "artifact", None)
                        if entry["tool"] == "retrieve_context" and artifact:
                            entry["sources"] = sorted({
                                d.metadata.get("source", "Unknown") for d in artifact
                            })
                        tool_calls_seen.append(entry)
                    elif isinstance(last_message, AIMessage) and last_message.content:
                        full_response = last_message.content
                        message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

        if full_response:
            st.session_state.messages.append(AIMessage(content=full_response))
            save_chat_to_firestore(session_id, st.session_state.messages) # [NEW]

            threading.Thread(
                target=update_session_metadata,
                args=(db, session_id, tool_calls_seen),
                daemon=True,
            ).start()