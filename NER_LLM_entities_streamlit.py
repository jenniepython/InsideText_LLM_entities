"""
Streamlit App: Text -> NER -> Geocoding -> JSON-LD & HTML Output (Gemini Only)
"""
import streamlit as st
import os
import json
import re
import time
import requests
from datetime import datetime
import urllib.parse

# Configure Streamlit page
st.set_page_config(
    page_title="Text Analysis Pipeline: NER + Geocoding + Structured Output",
    layout="wide"
)

# Model options - ONLY Gemini 1.5 Flash
MODEL_OPTIONS = {
    "Gemini 1.5 Flash": {
        "model_name": "gemini-1.5-flash",
        "description": "Google’s lightweight LLM for fast generation",
        "provider": "google"
    }
}

def construct_ner_prompt(text):
    prompt = f"""Extract named entities from the following text. Return only a JSON array with objects having \"text\", \"type\", and \"start_pos\" fields.

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISC

Text: \"{text}\"

JSON array:"""
    return prompt

def extract_json_from_response(response_text):
    response_text = response_text.strip()
    json_patterns = [r'\[.*?\]', r'\{.*?\}']
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                continue
    return None

def geocode_location(location_text):
    try:
        encoded_location = urllib.parse.quote(location_text)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_location}&format=json&limit=1"
        headers = {'User-Agent': 'StreamlitNERApp/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            result = data[0]
            return {
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"]),
                "display_name": result["display_name"],
                "confidence": float(result.get("importance", 0.5))
            }
        return None
    except Exception as e:
        st.warning(f"Geocoding failed for '{location_text}': {str(e)}")
        return None

def create_json_ld(entities, original_text):
    timestamp = datetime.now().isoformat()
    json_ld = {
        "@context": {
            "@vocab": "https://schema.org/",
            "geo": "https://schema.org/",
            "place": "https://schema.org/Place",
            "person": "https://schema.org/Person",
            "organization": "https://schema.org/Organization"
        },
        "@type": "TextDigitalDocument",
        "name": "Named Entity Recognition Analysis",
        "text": original_text,
        "dateCreated": timestamp,
        "mentions": []
    }
    for entity in entities:
        obj = {
            "@type": "Thing",
            "name": entity["text"],
            "additionalType": entity["type"],
            "startOffset": entity.get("start_pos", -1)
        }
        if entity["type"] == "PERSON":
            obj["@type"] = "Person"
        elif entity["type"] == "ORGANIZATION":
            obj["@type"] = "Organization"
        elif entity["type"] == "LOCATION":
            obj["@type"] = "Place"
            if entity.get("geocoding"):
                obj["geo"] = {
                    "@type": "GeoCoordinates",
                    "latitude": entity["geocoding"]["latitude"],
                    "longitude": entity["geocoding"]["longitude"]
                }
                obj["name"] = entity["geocoding"]["display_name"]
        elif entity["type"] == "DATE":
            obj["@type"] = "Date"
        json_ld["mentions"].append(obj)
    return json_ld

def create_html_output(entities, text):
    html = ["<html><head><meta charset='utf-8'><title>NER Output</title></head><body>"]
    html.append("<h2>Named Entities with Links</h2><ul>")
    for ent in entities:
        label = ent['type']
        value = ent['text']
        link = ""
        if label == "PERSON":
            link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(value)}"
        elif label == "ORGANIZATION":
            link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(value)}"
        elif label == "LOCATION" and ent.get("geocoding"):
            lat = ent['geocoding']['latitude']
            lon = ent['geocoding']['longitude']
            link = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=12"
        html.append(f"<li><strong>{label}</strong>: <a href='{link}' target='_blank'>{value}</a></li>")
    html.append("</ul><h3>Original Text</h3><p>{}</p>".format(text))
    html.append("</body></html>")
    return "\n".join(html)

# Streamlit UI
st.title("Text Analysis Pipeline: NER + Geocoding + Structured Output")
st.markdown("Enter text → Extract entities → Geocode locations → Generate JSON-LD & HTML")

selected_model = list(MODEL_OPTIONS.keys())[0]
geocode_locations = st.checkbox("Enable Geocoding for Locations", value=True)
user_input = st.text_area("Enter text to analyze:", height=200)

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing text through Gemini → Geocoding → Output generation..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = construct_ner_prompt(user_input)
                gemini_response = model.generate_content(prompt)
                llm_response = gemini_response.text
                entities = extract_json_from_response(llm_response)
                if not entities:
                    st.warning("Could not parse JSON from Gemini response.")
                    st.stop()
                st.success(f"✓ Extracted {len(entities)} entities")
                if geocode_locations:
                    for entity in entities:
                        if entity["type"] == "LOCATION":
                            geodata = geocode_location(entity["text"])
                            if geodata:
                                entity["geocoding"] = geodata
                json_ld_data = create_json_ld(entities, user_input)
                html_output = create_html_output(entities, user_input)

                st.subheader("JSON-LD Output")
                st.json(json_ld_data)
                st.download_button("Download JSON-LD", data=json.dumps(json_ld_data, indent=2), file_name="entities.jsonld", mime="application/ld+json")

                st.subheader("HTML Output")
                st.download_button("Download HTML with Links", data=html_output, file_name="entities.html", mime="text/html")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
