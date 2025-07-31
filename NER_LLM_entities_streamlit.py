"""
Streamlit App: Text -> NER -> Geocoding -> JSON-LD & HTML Output (Gemini Only)
"""
import streamlit as st

if 'entities' not in st.session_state:
    st.session_state['entities'] = None
if 'json_ld_data' not in st.session_state:
    st.session_state['json_ld_data'] = None
if 'html_output' not in st.session_state:
    st.session_state['html_output'] = None
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
    excerpt = text[:300] + ("..." if len(text) > 300 else "")
    prompt = f"""You are an expert assistant tasked with named entity recognition.

Given this excerpt from the text:
\"{excerpt}\"

Use it to identify entities in the full text below. For each entity, include its position in the full text using a field called \"start_pos\".

Return a JSON array of objects with \"text\", \"type\", and \"start_pos\" fields only.

Text to analyze:
{text}

JSON array:
"""
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

    def create_highlighted_html( text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities for display.
        
        Args:
            text: Original text
            entities: List of entity dictionaries
            
        Returns:
            HTML string with highlighted entities
        """
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Color scheme
        colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Highlight entities that have links OR coordinates
            has_links = (entity.get('britannica_url') or 
                         entity.get('wikidata_url') or 
                         entity.get('wikipedia_url') or     
                         entity.get('openstreetmap_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_links or has_coordinates):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#E7E2D2')
            
            # Create tooltip with entity information
            tooltip_parts = [f"Type: {entity['type']}"]
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link (priority: Wikipedia > Wikidata > Britannica > OpenStreetMap > Coordinates only)
            if entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('openstreetmap_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just highlight with coordinates (no link)
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    html = ["<html><head><meta charset='utf-8'><title>NER Output</title></head><body>"]
    html.append("<h2>Named Entities with Links</h2><ul>")
    for ent in entities:
        label = ent['type']
        value = ent['text']
        html.append(f"<li><strong>{label}</strong>: {value}</li>")
    html.append(f"</ul><h3>Annotated Text</h3><p>{text_html}</p>")
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








