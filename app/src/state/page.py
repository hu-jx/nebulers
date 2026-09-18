import streamlit as st
_KEY = "page"

def init(default="main"):
    st.session_state.setdefault(_KEY, default)

def current():
    return st.session_state[_KEY]

def switch_to(name):
    st.session_state[_KEY] = name