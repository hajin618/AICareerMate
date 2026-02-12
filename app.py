import streamlit as st
import requests

st.title("AI Career Coach")

file = st.file_uploader("Upload Resume")

job = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if file and job:

        files = {"file": file.getvalue()}

        res = requests.post(
            "http://localhost:8000/analyze",
            files={"file": file}
        )

        st.json(res.json())
