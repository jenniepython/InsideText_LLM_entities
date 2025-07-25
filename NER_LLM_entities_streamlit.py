#!/usr/bin/env python3
"""
Streamlit App using a Free Hugging Face LLM for Entity Recognition via Prompting
"""
import streamlit as st
import requests
import os
import json

# Configure Streamlit page
st.set_page_config(
    page_title="LLM-based Entity Recognition",
    layout="centered"
)

# Hugging Face API setup (store your token securely)
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
if HF_API_TOKEN is None:
    st.error("HF_API_TOKEN is not set!")
else:
    st.success("HF_API_TOKEN is correctly set.")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# Function to query the LLM API
def query_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.2, "max_new_tokens": 500}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()[0]['generated_text']

# Prompt construction for entity extraction
def construct_prompt(text):
    prompt = (
        "Extract named entities (people, organizations, locations, dates, etc.) from the following text. "
        "Provide the output as a JSON list with each entity having 'text', 'type', and 'context'.\n\n"
        f"Text: {text}\n\nEntities:"
    )
    return prompt

# Streamlit UI
st.title("Entity Recognition with Mistral LLM")

user_input = st.text_area("Enter text to analyze:", height=300)

if st.button("Extract Entities"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Extracting entities..."):
            prompt = construct_prompt(user_input)
            try:
                llm_response = query_llm(prompt)
                entities = json.loads(llm_response)

                if entities:
                    st.success(f"Extracted {len(entities)} entities.")
                    st.json(entities)
                else:
                    st.info("No entities found.")

            except Exception as e:
                st.error(f"Error extracting entities: {e}")
