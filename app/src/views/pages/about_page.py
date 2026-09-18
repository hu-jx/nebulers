from textwrap import dedent
import streamlit as st

def render():
    html = dedent("""
    <div class='card'>
    <p><b>title</b> yap </p>
    <p><b>title</b> yap</p>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)