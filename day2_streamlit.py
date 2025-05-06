import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from streamlit.logger import get_logger
logger = get_logger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
st.title("My Gen-AI App")
repo_id = "microsoft/Phi-3-mini-4k-instruct"
temp = 1

print("Repo id: " + repo_id)
# print(f'API Token: {os.getenv("HUGGINGFACE_API_TOKEN")}')

with st.form("sample app"):
    # ← add this slider:
    temp = st.slider(
        "Temperature",      # label
        min_value=0.0,      # lower bound
        max_value=1.0,      # upper bound
        value=0.75,          # default
        step=0.1            # granularity
    )

    txt = st.text_area("Ask me anything. I dare you")
    submitted = st.form_submit_button("Submit")

    if submitted:
        llm = HuggingFaceEndpoint(
            repo_id = repo_id,
            task="text-generation",
            temperature=temp
        )
        chat = ChatHuggingFace(llm=llm, Verbose=True)
        logger.info("Invoking")
        ans = chat.invoke(txt)
        st.info(ans.content)
        logger.info("Done.")
