import os
import requests
import streamlit as st
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
api_key = os.environ.get("OLLAMA_API_KEY")

# Try cloud client if API key present
cloud_client = None
if api_key:
    try:
        cloud_client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"}
        )
    except Exception:
        cloud_client = None

def generate_text(prompt: str, model_name: str):
    if not prompt:
        return "Please enter a prompt."
    try:
        if "cloud" in model_name:
            if not cloud_client:
                return "Error: OLLAMA_API_KEY not set or cloud client unavailable."
            messages = [{"role": "user", "content": prompt}]
            response = cloud_client.chat(model_name.replace("-cloud", ""), messages=messages)
            return response["message"]["content"]
        else:
            data = {"model": model_name, "prompt": prompt, "stream": False}
            r = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "No response.")
    except Exception as e:
        return f"An error occurred: {e}"

st.set_page_config(page_title="Ollama LLM Text Generation", layout="wide")

# Optional: limit text area/container width
MAX_WIDTH = 900
st.markdown(
    f"""
    <style>
    .block-container{{max-width:{MAX_WIDTH}px; margin: auto;}}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Ollama LLM Text Generation")
st.write("Select a model. Local models use your running Ollama instance; cloud models use Ollama Cloud (requires OLLAMA_API_KEY).")

model = st.selectbox(
    "Model",
    ["qwen:0.5b", "gpt-oss:120b-cloud", "qwen3:4b", "mistral:latest"],
    index=2
)

prompt = st.text_area("Your prompt", placeholder="Enter a starting phrase...")

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_text(prompt, model)
    st.subheader("Result")
    st.write(output)