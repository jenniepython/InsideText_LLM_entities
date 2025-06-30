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
from transformers import AutoTokenizer, AutoModelForCausalLM
from streamlit_folium import st_folium
import folium
import xml.etree.ElementTree as ET

# --- Branding ---
st.set_page_config(page_title="InsideText NER Tool", layout="wide")

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("### InsideText Named Entity Recognition")
    st.markdown("This tool extracts, links, and maps named entities using a lightweight local model.")
    st.markdown("---")

st.markdown(
    """
    <style>
    body { color: #222; font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f8f8f8; }
    .block-container { padding: 2rem 2rem 2rem 2rem; }
    .stButton>button, .stDownloadButton>button { background-color: #003f5c; color: white; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model (small CPU-friendly model) ---
@st.cache_resource
def load_model():
    model_id = "distilgpt2"  # Small model for Streamlit Cloud compatibility
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

# --- Entity Extraction (simple pattern-based fallback for demo) ---
def extract_entities_with_types(text):
    # Simulate light LLM-based NER using simple heuristics
    tokens = list(set(re.findall(r"[A-Z][a-z]+(?:\\s[A-Z][a-z]+)*", text)))
    results = []
    for token in tokens:
        # Very naive typing
        if re.search(r"(Inc|Corp|Ltd|University|Institute|Council)", token):
            ent_type = "Organisation"
        elif re.search(r"(War|Conference|Summit|Treaty)", token):
            ent_type = "Event"
        elif re.search(r"(Street|Avenue|River|City|Village|Park|Hill|Mountain|Valley)", token):
            ent_type = "Place"
        elif re.match(r"^[A-Z][a-z]+(\\s[A-Z][a-z]+)?$", token):
            ent_type = "Person"
        else:
            ent_type = "Thing"
        results.append({"text": token, "type": ent_type})
    return results

# --- Wikidata Linking ---
def link_wikidata(entity_text):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": "en",
        "format": "json"
    }
    try:
        r = requests.get(url, params=params)
        results = r.json().get("search", [])
        if results:
            return results[0]["concepturi"]
    except Exception as e:
        print(f"Wikidata lookup failed for {entity_text}: {e}")
    return None

# --- Pleiades Linking ---
def link_pleiades(entity_text):
    url = f"https://pleiades.stoa.org/places/search?SearchableText={entity_text}"
    try:
        response = requests.get(url)
        if response.status_code == 200 and "places" in response.url:
            return response.url  # Pleiades redirects to entity page
    except Exception as e:
        print(f"Pleiades lookup failed for {entity_text}: {e}")
    return None

# --- Geocoding ---
def geocode_place(name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": name,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "accept-language": "en"
    }
    headers = {
        "User-Agent": "InsideTextNER/1.0"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if data:
            return {
                "place": name,
                "display_name": data[0]["display_name"],
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"])
            }
    except Exception as e:
        print(f"Geocoding failed for {name}: {e}")
    return None

# --- Map ---
def display_map(locations):
    if not locations:
        return
    m = folium.Map(location=[locations[0]["lat"], locations[0]["lon"]], zoom_start=2)
    for loc in locations:
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=loc["display_name"]
        ).add_to(m)
    st_folium(m, width=700, height=500)

# --- JSON-LD, TEI, HTML ---
TYPE_MAP = {
    "person": "Person",
    "place": "Place",
    "organisation": "Organization",
    "organization": "Organization",
    "event": "Event",
    "object": "Thing",
    "workofart": "CreativeWork",
    "date": "Date",
    "artifact": "Thing"
}

def generate_jsonld(entities):
    out = []
    for ent in entities:
        ent_type = TYPE_MAP.get(ent['type'].lower(), 'Thing')
        item = {
            "@context": "https://schema.org",
            "@type": ent_type,
            "name": ent["text"]
        }
        if ent.get("wikidata_uri"):
            item["sameAs"] = ent["wikidata_uri"]
        if ent.get("pleiades_uri"):
            item["identifier"] = ent["pleiades_uri"]
        out.append(item)
    return out

def generate_html(entities):
    html = "<ul>"
    for ent in entities:
        label = ent['text']
        link = ent.get("wikidata_uri") or ent.get("pleiades_uri")
        if link:
            html += f"<li><strong>{ent['type']}</strong>: <a href='{link}' target='_blank'>{label}</a></li>"
        else:
            html += f"<li><strong>{ent['type']}</strong>: {label}</li>"
    html += "</ul>"
    return html

def generate_tei(entities):
    root = ET.Element("TEI")
    text = ET.SubElement(root, "text")
    body = ET.SubElement(text, "body")
    list_name = ET.SubElement(body, "list")
    for ent in entities:
        name = ET.SubElement(list_name, "name")
        name.set("type", ent['type'])
        name.text = ent['text']
    return ET.tostring(root, encoding='unicode')

# --- Highlighting ---
def highlight_entities(text, entities):
    for ent in sorted(entities, key=lambda e: -len(e["text"])):
        text = re.sub(
            r'\\b({})\\b'.format(re.escape(ent["text"])),
            rf"<mark style='background-color:#ffff88;'>\\1</mark>",
            text
        )
    return f"<div style='white-space: pre-wrap;'>{text}</div>"

# --- File Upload ---
uploaded_files = st.file_uploader("Upload .txt, .pdf, or .docx file(s)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

def read_text_from_file(file):
    if file.name.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        docx_file = docx.Document(file)
        return "\n".join([p.text for p in docx_file.paragraphs])
    else:
        return file.read().decode("utf-8")

if uploaded_files:
    output_zip = io.BytesIO()
    with zipfile.ZipFile(output_zip, "w") as zf:
        for file in uploaded_files:
            text = read_text_from_file(file)
            entities = extract_entities_with_types(text)
            for ent in entities:
                ent["wikidata_uri"] = link_wikidata(ent["text"])
                if ent["type"].lower() == "place":
                    ent["pleiades_uri"] = link_pleiades(ent["text"])

            jsonld_str = pd.io.json.dumps(generate_jsonld(entities), indent=2)
            html_str = generate_html(entities)
            tei_str = generate_tei(entities)

            basename = file.name.rsplit(".", 1)[0]
            zf.writestr(f"{basename}.jsonld", jsonld_str)
            zf.writestr(f"{basename}.html", html_str)
            zf.writestr(f"{basename}.tei.xml", tei_str)

    st.download_button("Download All Outputs (ZIP)", data=output_zip.getvalue(), file_name="entities_output.zip", mime="application/zip")
