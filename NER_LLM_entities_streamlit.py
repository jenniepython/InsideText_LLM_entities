#!/usr/bin/env python3
"""
Streamlit App using Free Hugging Face LLMs for Entity Recognition via Prompting
"""
import streamlit as st
import requests
import os
import json
import re
import time

# Configure Streamlit page
st.set_page_config(
    page_title="LLM-based Entity Recognition",
    layout="centered"
)

# Hugging Face API setup
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
if HF_API_TOKEN is None:
    st.error("HF_API_TOKEN is not set! Please set it in your environment variables.")
    st.stop()

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Model options - these are free and work well for NER
MODEL_OPTIONS = {
    "Mistral-7B-Instruct": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-2-7B-Chat": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
    "Zephyr-7B-Beta": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "CodeLlama-7B-Instruct": "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-Instruct-hf"
}

def query_llm(prompt, model_url, max_retries=3):
    """Query the LLM API with retry logic"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 800,
            "do_sample": True,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(model_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                st.warning(f"Model is loading... Attempt {attempt + 1}/{max_retries}")
                time.sleep(10)
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

def construct_prompt_v1(text):
    """Simple JSON format prompt"""
    prompt = f"""Extract named entities from the following text. Return ONLY a valid JSON array where each entity has "text", "type", and "start_pos" fields.

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISC

Text: "{text}"

JSON:"""
    return prompt

def construct_prompt_v2(text):
    """More structured prompt with examples"""
    prompt = f"""<s>[INST] You are an expert at named entity recognition. Extract all named entities from the given text and format them as a JSON array.

Each entity should have:
- "text": the actual entity text
- "type": one of PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISC
- "start_pos": character position where entity starts

Example:
Text: "John Smith works at Microsoft in Seattle since 2020."
JSON: [
  {{"text": "John Smith", "type": "PERSON", "start_pos": 0}},
  {{"text": "Microsoft", "type": "ORGANIZATION", "start_pos": 20}},
  {{"text": "Seattle", "type": "LOCATION", "start_pos": 33}},
  {{"text": "2020", "type": "DATE", "start_pos": 47}}
]

Now extract entities from this text:
Text: "{text}"

JSON: [/INST]"""
    return prompt

def construct_prompt_v3(text):
    """Simple list format - more reliable for smaller models"""
    prompt = f"""Extract named entities from this text. List each entity on a new line in this format:
ENTITY_TYPE: entity_text

Types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISC

Text: "{text}"

Entities:
"""
    return prompt

def parse_simple_format(response_text):
    """Parse simple list format response"""
    entities = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            try:
                entity_type, entity_text = line.split(':', 1)
                entity_type = entity_type.strip().upper()
                entity_text = entity_text.strip()
                
                if entity_type in ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'MISC']:
                    entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "start_pos": -1  # Position not available in this format
                    })
            except:
                continue
    
    return entities

def extract_json_from_response(response_text):
    """Extract JSON from LLM response"""
    # Try to find JSON array in the response
    json_pattern = r'\[.*?\]'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON found, try to parse as simple format
    return None

def fallback_entity_extraction(text):
    """Simple regex-based fallback for basic entity types"""
    entities = []
    
    # Simple patterns for common entities
    patterns = {
        'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
        'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        'PERCENT': r'\d+(?:\.\d+)?%'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "text": match.group(),
                "type": entity_type,
                "start_pos": match.start()
            })
    
    return entities

# Streamlit UI
st.title("Entity Recognition with Free LLMs")
st.markdown("Extract named entities from text using Hugging Face's free inference API")

# Model selection
selected_model = st.selectbox(
    "Choose a model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

# Prompt strategy selection
prompt_strategy = st.selectbox(
    "Choose prompting strategy:",
    options=["Simple JSON", "Structured with Examples", "Simple List Format"],
    index=2
)

# Text input
user_input = st.text_area(
    "Enter text to analyze:", 
    height=200,
    placeholder="Example: John Smith works at Microsoft in Seattle. He started there in January 2020 and earns $75,000 per year."
)

# Example texts
if st.button("Load Example Text"):
    example_text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Los Altos, California. The company is now headquartered in Cupertino and is worth over $2 trillion as of 2024."
    st.session_state.example_text = example_text

if 'example_text' in st.session_state:
    user_input = st.text_area(
        "Enter text to analyze:", 
        value=st.session_state.example_text,
        height=200
    )

# Extract entities button
if st.button("Extract Entities", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Extracting entities..."):
            try:
                # Select prompt based on strategy
                if prompt_strategy == "Simple JSON":
                    prompt = construct_prompt_v1(user_input)
                elif prompt_strategy == "Structured with Examples":
                    prompt = construct_prompt_v2(user_input)
                else:
                    prompt = construct_prompt_v3(user_input)
                
                # Query the selected model
                model_url = MODEL_OPTIONS[selected_model]
                llm_response = query_llm(prompt, model_url)
                
                st.subheader("Raw LLM Response:")
                st.text(llm_response)
                
                # Parse the response
                entities = None
                
                if prompt_strategy == "Simple List Format":
                    entities = parse_simple_format(llm_response)
                else:
                    entities = extract_json_from_response(llm_response)
                
                if entities and len(entities) > 0:
                    st.success(f"Extracted {len(entities)} entities")
                    
                    # Display entities in a nice format
                    st.subheader("Extracted Entities:")
                    for i, entity in enumerate(entities):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{entity['text']}**")
                            with col2:
                                st.write(f"`{entity['type']}`")
                            with col3:
                                if entity.get('start_pos', -1) >= 0:
                                    st.write(f"Pos: {entity['start_pos']}")
                                else:
                                    st.write("Pos: N/A")
                    
                    # JSON view
                    with st.expander("View as JSON"):
                        st.json(entities)
                        
                else:
                    st.warning("No entities found or failed to parse response. Trying fallback method...")
                    
                    # Try fallback extraction
                    fallback_entities = fallback_entity_extraction(user_input)
                    if fallback_entities:
                        st.info(f"Fallback method found {len(fallback_entities)} entities")
                        st.json(fallback_entities)
                    else:
                        st.error("No entities could be extracted")
                        
            except Exception as e:
                st.error(f"Error extracting entities: {str(e)}")
                
                # Show fallback option
                st.warning("Trying fallback extraction method...")
                try:
                    fallback_entities = fallback_entity_extraction(user_input)
                    if fallback_entities:
                        st.success(f"Fallback method extracted {len(fallback_entities)} entities")
                        st.json(fallback_entities)
                    else:
                        st.error("Fallback method also failed")
                except Exception as fallback_error:
                    st.error(f"Fallback method error: {str(fallback_error)}")

# Instructions
with st.expander("Instructions"):
    st.markdown("""
    ### How to use:
    1. **Set up your HF API token**: Set the `HF_API_TOKEN` environment variable with your Hugging Face API token
    2. **Choose a model**: Different models may perform differently for entity recognition
    3. **Select prompting strategy**: 
       - **Simple List Format**: Most reliable for smaller models
       - **Simple JSON**: Attempts to get structured JSON output
       - **Structured with Examples**: More detailed prompting with examples
    4. **Enter your text** and click "Extract Entities"
    
    ### Tips:
    - The "Simple List Format" strategy tends to work best with free models
    - If one model doesn't work well, try another
    - The app includes a regex-based fallback for basic entity types
    - Some models may take time to load (you'll see a loading message)
    """)
