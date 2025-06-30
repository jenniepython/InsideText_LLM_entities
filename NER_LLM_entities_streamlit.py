import streamlit as st
import torch
import pandas as pd
import requests
import re
import fitz  # PyMuPDF
import docx
import zipfile
import io
import json
from streamlit_folium import st_folium
import folium
import xml.etree.ElementTree as ET

# --- Branding ---
st.set_page_config(page_title="InsideText NER Tool", layout="centered")

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("### InsideText Named Entity Recognition")
    st.markdown("Extract, annotate, and explore named entities with LLM support.")
    st.markdown("---")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stButton>button, .stDownloadButton>button {
        background-color: #003f5c;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .highlight-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Hugging Face Inference API ---
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {st.secrets['hf_token']}"}

# --- Lookup Entities on Wikidata and Pelagios ---
def link_entity_to_wikidata(entity_text):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_text}&language=en&format=json"
    try:
        r = requests.get(url)
        results = r.json().get("search", [])
        if results:
            return f"https://www.wikidata.org/wiki/{results[0]['id']}"
    except:
        pass
    return None

def link_entity_to_pelagios(entity_text):
    url = f"https://pleiades.stoa.org/places/search?SearchableText={entity_text}"
    return url

# --- Extract Entities Using LLM Prompt ---
def extract_entities_with_types(text):
    prompt = (
        "You are an expert in named entity recognition. Extract all named entities from the text below. "
        "Return a JSON array of dictionaries. Each dictionary must contain two fields: "
        "\"text\" (the exact entity) and \"type\" (e.g., Person, Organisation, Place, Event, Object, Device, etc.).\n\n"
        f"Text:\n{text}\n\n"
        "Entities:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 1024}
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=HUGGINGFACE_HEADERS, json=payload)
        raw = response.json()[0].get("generated_text", "")
        # Use regex to find the first array-like JSON string
        match = re.search(r"\[\s*{.*?}\s*\]", raw, re.DOTALL)
        if not match:
            st.error("LLM response could not be parsed as JSON. Try adjusting the prompt or input length.")
            return []
        entities = json.loads(match.group(0))
        for ent in entities:
            ent["wikidata"] = link_entity_to_wikidata(ent["text"])
            if ent.get("type", "").lower() == "place":
                ent["pleiades"] = link_entity_to_pelagios(ent["text"])
        return entities
    except Exception as e:
        st.error(f"Failed to extract entities: {e}")
        return []


# --- Upload and Process File or Input Text ---
st.title("InsideText LLM Entity Extraction")

input_mode = st.radio("Choose input method:", ("Upload file", "Paste text"))

text = ""
if input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area("Paste your text here", height=300)

if text:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Text")
        st.text_area("", value=text, height=300)

    with col2:
        st.markdown("#### Highlighted Entities")
        def highlight_entities(text, entities):
            for ent in sorted(entities, key=lambda e: -len(e["text"])):
                text = re.sub(rf"\b{re.escape(ent['text'])}\b", f"<mark style='background-color:#ffff88;'>{ent['text']}</mark>", text)
            return f"<div class='highlight-box' style='white-space: pre-wrap;'>{text}</div>"

        entities = extract_entities_with_types(text)
        st.markdown(highlight_entities(text, entities), unsafe_allow_html=True)

    with st.expander("View Extracted Entities"):
        df = pd.DataFrame(entities)
        st.dataframe(df)

    # --- JSON-LD Export ---
    def generate_jsonld(entities):
        if not entities:
            return {"@context": "https://schema.org", "@type": "CreativeWork", "name": "No entities found."}
        return {
            "@context": "https://schema.org",
            "@type": "CreativeWork",
            "name": "Named Entity Extraction",
            "mentions": [
                {
                    "@type": "Thing",
                    "name": ent.get("text", ""),
                    "additionalType": ent.get("type", ""),
                    "sameAs": ent.get("wikidata") or ent.get("pleiades")
                }
                for ent in entities if ent is not None
            ]
        }

    jsonld_str = json.dumps(generate_jsonld(entities), indent=2)
    st.download_button("Download JSON-LD", data=jsonld_str, file_name="entities.jsonld", mime="application/ld+json")

    # --- TEI-XML Export ---
    def generate_tei_xml(text, entities):
        root = ET.Element("TEI")
        body = ET.SubElement(root, "text")
        for ent in entities:
            tag = ET.SubElement(body, "name", attrib={"type": ent["type"]})
            tag.text = ent["text"]
        return ET.tostring(root, encoding="unicode")

    tei_str = generate_tei_xml(text, entities)
    st.download_button("Download TEI-XML", data=tei_str, file_name="entities.tei.xml", mime="application/xml")

    # --- HTML Export ---
    def generate_html_output(text, entities):
        html = highlight_entities(text, entities)
        html = f"<html><head><meta charset='utf-8'><title>Entity Highlights</title></head><body>{html}</body></html>"
        return html

    html_str = generate_html_output(text, entities)
    st.download_button("Download HTML", data=html_str, file_name="highlighted_entities.html", mime="text/html")

