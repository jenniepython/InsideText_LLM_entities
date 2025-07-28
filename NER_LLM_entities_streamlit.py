"""
Streamlit App: Text -> NER -> Geocoding -> JSON-LD & HTML Output
"""
import streamlit as st
import requests
import os
import json
import re
import time
from datetime import datetime
import urllib.parse

# Configure Streamlit page
st.set_page_config(
    page_title="Text Analysis Pipeline: NER + Geocoding + Structured Output",
    layout="wide"
)

# Hugging Face API setup
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

if HF_API_TOKEN is None:
    st.warning("HF_API_TOKEN environment variable not found.")
    manual_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    if manual_token:
        HF_API_TOKEN = manual_token
    else:
        st.error("Please provide a Hugging Face API token to continue.")
        st.info("Get your free token at: https://huggingface.co/settings/tokens")
        st.stop()

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Model options - ONLY generative LLMs for prompting-based NER
MODEL_OPTIONS = {
    "Google Flan-T5 (Small)": {
        "model_name": "google/flan-t5-small",
        "description": "Text-to-text model available via API",
        "inference_api": True
    }
}

def query_llm(prompt, model_url, max_retries=3):
    """Query the LLM API with retry logic"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 1000,
            "do_sample": True,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(model_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                st.warning(f"Model loading... Attempt {attempt + 1}/{max_retries}")
                time.sleep(10)
                continue
            elif response.status_code == 403:
                raise Exception("403 Forbidden: Check your API token")
            elif response.status_code == 429:
                st.warning(f"Rate limit exceeded. Waiting... Attempt {attempt + 1}/{max_retries}")
                time.sleep(20)
                continue
                
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            elif isinstance(result, dict):
                return result.get('generated_text', '')
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(5)
    
    raise Exception("Max retries exceeded")

def test_model_availability(model_url):
    """Test if model is available"""
    try:
        test_payload = {
            "inputs": "Test input",
            "parameters": {"max_new_tokens": 10}
        }
        response = requests.post(model_url, headers=headers, json=test_payload, timeout=15)
        
        if response.status_code == 404:
            return False, "Model not found"
        elif response.status_code == 403:
            return False, "Access forbidden - check API token"
        elif response.status_code == 503:
            return True, "Model loading (available but needs time)"
        elif response.status_code == 200:
            return True, "Model ready"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

# Test model availability
if st.button("🔍 Test All Models"):
    st.subheader("Testing LLM availability...")
    working_models = []
    
    for model_name, model_url in MODEL_OPTIONS.items():
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{model_name}**")
            with col2:
                available, message = test_model_availability(model_url)
                if available:
                    st.success(f"✓ {message}")
                    working_models.append(model_name)
                else:
                    st.error(f"✗ {message}")
    
    if working_models:
        st.info(f"Working models: {', '.join(working_models)}")
    else:
        st.warning("No models are currently accessible. Try checking your API token.")

def construct_ner_prompt(text):
    """General prompt that works with most models"""
    prompt = f"""Extract named entities from the following text. Return only a JSON array with objects having "text", "type", and "start_pos" fields.

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISC

Text: "{text}"

JSON array:"""
    return prompt

def construct_t5_prompt(text):
    """T5-specific prompt format"""
    return f"Extract entities: {text}"

def extract_json_from_response(response_text):
    """Extract JSON from LLM response"""
    # Clean up the response
    response_text = response_text.strip()
    
    # Try to find complete JSON array
    json_patterns = [
        r'\[.*?\]',  # Standard array
        r'\{.*?\}',  # Single object (wrap in array)
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                continue
    
    return None

def geocode_location(location_text):
    """Geocode a location using Nominatim (free)"""
    try:
        # Use Nominatim (OpenStreetMap) - free geocoding service
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
    """Create JSON-LD structured data"""
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
        "about": [],
        "mentions": []
    }
    
    for entity in entities:
        entity_obj = {
            "@type": "Thing",
            "name": entity["text"],
            "additionalType": entity["type"],
            "startOffset": entity.get("start_pos", -1)
        }
        
        # Add specific schema.org types
        if entity["type"] == "PERSON":
            entity_obj["@type"] = "Person"
        elif entity["type"] == "ORGANIZATION":
            entity_obj["@type"] = "Organization"
        elif entity["type"] == "LOCATION":
            entity_obj["@type"] = "Place"
            if entity.get("geocoding"):
                entity_obj["geo"] = {
                    "@type": "GeoCoordinates",
                    "latitude": entity["geocoding"]["latitude"],
                    "longitude": entity["geocoding"]["longitude"]
                }
                entity_obj["name"] = entity["geocoding"]["display_name"]
        elif entity["type"] == "DATE":
            entity_obj["@type"] = "Date"
        
        json_ld["mentions"].append(entity_obj)
    
    return json_ld

def create_html_output(entities, original_text, json_ld_data):
    """Create rich HTML output with embedded JSON-LD"""
    
    # Create highlighted text
    highlighted_text = original_text
    offset = 0
    
    # Sort entities by start position (reverse order to maintain positions)
    sorted_entities = sorted(
        [e for e in entities if e.get("start_pos", -1) >= 0], 
        key=lambda x: x["start_pos"], 
        reverse=True
    )
    
    for entity in sorted_entities:
        start = entity["start_pos"]
        end = start + len(entity["text"])
        
        # Color coding for different entity types
        colors = {
            "PERSON": "#e1f5fe",
            "ORGANIZATION": "#f3e5f5", 
            "LOCATION": "#e8f5e8",
            "DATE": "#fff3e0",
            "TIME": "#fff3e0",
            "MONEY": "#f1f8e9",
            "PERCENT": "#f1f8e9",
            "MISC": "#fafafa"
        }
        
        color = colors.get(entity["type"], "#fafafa")
        
        # Create highlighted span
        highlighted_span = f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 3px; border-left: 3px solid #666;" title="{entity["type"]}">{entity["text"]}</span>'
        
        # Replace in text
        highlighted_text = highlighted_text[:start] + highlighted_span + highlighted_text[end:]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Named Entity Recognition Results</title>
        <script type="application/ld+json">
        {json.dumps(json_ld_data, indent=2)}
        </script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .original-text {{
                background: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
                margin: 20px 0;
            }}
            .highlighted-text {{
                background: white;
                padding: 20px;
                border-radius: 5px;
                border: 1px solid #ddd;
                margin: 20px 0;
                line-height: 1.8;
            }}
            .entities-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .entity-card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .entity-type {{
                background: #333;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .entity-text {{
                font-size: 1.1em;
                font-weight: bold;
                margin: 10px 0;
                color: #333;
            }}
            .geocoding-info {{
                background: #e8f5e8;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                font-size: 0.9em;
            }}
            .legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 1px solid #ccc;
            }}
            .metadata {{
                background: #f0f8ff;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Named Entity Recognition Analysis</h1>
            
            <div class="metadata">
                <strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                <strong>Total Entities Found:</strong> {len(entities)}<br>
                <strong>Text Length:</strong> {len(original_text)} characters
            </div>
            
            <h2>Original Text</h2>
            <div class="original-text">
                {original_text}
            </div>
            
            <h2>Highlighted Text with Entities</h2>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e1f5fe;"></div>
                    <span>Person</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f3e5f5;"></div>
                    <span>Organization</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e8f5e8;"></div>
                    <span>Location</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fff3e0;"></div>
                    <span>Date/Time</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f1f8e9;"></div>
                    <span>Money/Percent</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fafafa;"></div>
                    <span>Other</span>
                </div>
            </div>
            
            <div class="highlighted-text">
                {highlighted_text}
            </div>
            
            <h2>Extracted Entities</h2>
            <div class="entities-grid">
    """
    
    for entity in entities:
        geocoding_html = ""
        if entity.get("geocoding"):
            geo = entity["geocoding"]
            geocoding_html = f"""
                <div class="geocoding-info">
                    <strong>Location:</strong> {geo["display_name"]}<br>
                    <strong>Coordinates:</strong> {geo["latitude"]:.6f}, {geo["longitude"]:.6f}<br>
                    <strong>Confidence:</strong> {geo["confidence"]:.2f}
                </div>
            """
        
        confidence_html = ""
        if entity.get("confidence"):
            confidence_html = f"<br><strong>Confidence:</strong> {entity['confidence']:.2f}"
        
        html_content += f"""
                <div class="entity-card">
                    <span class="entity-type">{entity["type"]}</span>
                    <div class="entity-text">{entity["text"]}</div>
                    <strong>Position:</strong> {entity.get("start_pos", "N/A")}{confidence_html}
                    {geocoding_html}
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Structured Data (JSON-LD)</h2>
            <details>
                <summary>Click to view embedded JSON-LD data</summary>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">""" + json.dumps(json_ld_data, indent=2) + """</pre>
            </details>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Streamlit UI
st.title("Text Analysis Pipeline: NER + Geocoding + Structured Output")
st.markdown("Enter text → Extract entities → Geocode locations → Generate JSON-LD & HTML")

# Model selection
col1, col2 = st.columns(2)
with col1:
    selected_model = st.selectbox(
        "Choose NER Model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )

with col2:
    geocode_locations = st.checkbox("Enable Geocoding for Locations", value=True)

# Text input
user_input = st.text_area(
    "Enter text to analyze:", 
    height=200,
    placeholder="Example: Apple Inc. was founded by Steve Jobs in Cupertino, California in April 1976. The company is now worth over $2 trillion."
)

# Example text loader
if st.button("Load Example Text"):
    example_text = "Microsoft Corporation was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico on April 4, 1975. The company later moved to Redmond, Washington and is now valued at over $3 trillion. CEO Satya Nadella announced new AI initiatives in Seattle last month, investing $10 billion in partnerships."
    user_input = example_text

# Main processing
if st.button("Analyze Text", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Processing text through NER → Geocoding → Output generation..."):
            try:
                # Step 1: Named Entity Recognition using LLM prompting only
                st.subheader("Step 1: Named Entity Recognition (LLM Prompting)")
                # model_url = MODEL_OPTIONS[selected_model]
                model_url = f"https://api-inference.huggingface.co/models/{MODEL_OPTIONS[selected_model]['model_name']}"
                
                entities = []
                
                # Use LLM prompting for all models
                if "t5" in selected_model.lower():
                    prompt = construct_t5_prompt(user_input)
                else:
                    prompt = construct_ner_prompt(user_input)
                
                llm_response = query_llm(prompt, model_url)
                
                st.write("**LLM Response:**")
                st.text(llm_response)
                
                # Parse response
                entities = extract_json_from_response(llm_response)
                if not entities:
                    st.warning("Could not parse JSON from LLM response. Trying fallback extraction...")
                    entities = fallback_entity_extraction(user_input)
                
                st.success(f"✓ Extracted {len(entities)} entities using LLM prompting")
                
                # Display extracted entities
                if entities:
                    for entity in entities:
                        st.write(f"**{entity['text']}** ({entity['type']})")
                
                # Step 2: Geocoding
                if geocode_locations and entities:
                    st.subheader("Step 2: Geocoding Locations")
                    location_entities = [e for e in entities if e["type"] == "LOCATION"]
                    
                    if location_entities:
                        geocoded_count = 0
                        for entity in location_entities:
                            with st.spinner(f"Geocoding {entity['text']}..."):
                                geocoding_result = geocode_location(entity["text"])
                                if geocoding_result:
                                    entity["geocoding"] = geocoding_result
                                    geocoded_count += 1
                                    st.success(f"✓ Geocoded {entity['text']}: {geocoding_result['display_name']}")
                                else:
                                    st.warning(f"✗ Could not geocode {entity['text']}")
                        
                        st.success(f"✓ Successfully geocoded {geocoded_count}/{len(location_entities)} locations")
                    else:
                        st.info("No location entities found for geocoding")
                
                # Step 3: Generate JSON-LD
                st.subheader("Step 3: Generate Structured Data")
                json_ld_data = create_json_ld(entities, user_input)
                
                # Step 4: Generate HTML
                html_output = create_html_output(entities, user_input, json_ld_data)
                
                # Display results
                st.subheader("Results")
                
                # Tabs for different outputs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🗂️ JSON-LD", "🌐 HTML Preview", "📄 Downloads"])
                
                with tab1:
                    st.write("### Analysis Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Entities", len(entities))
                    with col2:
                        location_count = len([e for e in entities if e["type"] == "LOCATION"])
                        st.metric("Locations", location_count)
                    with col3:
                        geocoded_count = len([e for e in entities if e.get("geocoding")])
                        st.metric("Geocoded", geocoded_count)
                    
                    # Entity breakdown
                    entity_types = {}
                    for entity in entities:
                        entity_types[entity["type"]] = entity_types.get(entity["type"], 0) + 1
                    
                    st.write("### Entity Types")
                    for entity_type, count in entity_types.items():
                        st.write(f"**{entity_type}**: {count}")
                
                with tab2:
                    st.write("### JSON-LD Structured Data")
                    st.json(json_ld_data)
                    
                    # Download button for JSON-LD
                    st.download_button(
                        label="Download JSON-LD",
                        data=json.dumps(json_ld_data, indent=2),
                        file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonld",
                        mime="application/ld+json"
                    )
                
                with tab3:
                    st.write("### HTML Output Preview")
                    st.components.v1.html(html_output, height=600, scrolling=True)
                
                with tab4:
                    st.write("### Download Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download HTML",
                            data=html_output,
                            file_name=f"ner_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📊 Download JSON-LD",
                            data=json.dumps(json_ld_data, indent=2),
                            file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonld",
                            mime="application/ld+json"
                        )
                    
                    with col3:
                        # Raw entities JSON
                        st.download_button(
                            label="🗃️ Download Raw Entities",
                            data=json.dumps(entities, indent=2),
                            file_name=f"raw_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Instructions
with st.expander("Instructions & Features"):
    st.markdown("""
    ### How this works:
    
    1. **Text Input**: Enter any text containing entities (people, places, organizations, etc.)
    2. **NER Processing**: Uses LLM prompting or dedicated NER models to extract entities
    3. **Geocoding**: Automatically geocodes location entities using OpenStreetMap
    4. **Structured Output**: Generates both JSON-LD and rich HTML output
    
    ### Features:
    - **LLM Prompting Only**: Uses generative language models with carefully crafted prompts
    - **Multiple LLM Options**: GPT-2, DistilGPT-2, T5, and FLAN-T5 models
    - **Free Geocoding**: Uses Nominatim (OpenStreetMap) for location data
    - **JSON-LD Output**: Schema.org compliant structured data
    - **Rich HTML**: Interactive visualization with embedded structured data
    - **Download Options**: Get results in multiple formats
    
    ### Why LLM Prompting?
    - More flexible than dedicated NER models
    - Can adapt to different entity types through prompting
    - Better context understanding for ambiguous entities
    - Can be fine-tuned through prompt engineering
    
    ### JSON-LD Benefits:
    - SEO-friendly structured data
    - Machine-readable entity information
    - Schema.org compliance
    - Easy integration with knowledge graphs
    
    ### Use Cases:
    - Content analysis and SEO optimization
    - Knowledge graph construction
    - Geographic information extraction
    - Entity relationship mapping
    """)
