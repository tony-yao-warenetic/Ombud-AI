import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore


@st.cache_resource(show_spinner="Connecting to Database...")
def init_firebase():
    if not firebase_admin._apps:
        cred_dict = None
        try:
            if "firebase" in st.secrets:
                cred_dict = dict(st.secrets["firebase"])
        except Exception:
            cred_dict = None
        if cred_dict:
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()
