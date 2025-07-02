#!/usr/bin/env python3
"""
Pure Prompt-Based Entity Extraction and Linking

No hardcoding - everything is handled by comprehensive prompts sent to language models.
The LLM handles extraction, classification, and linking instructions all in one call.
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="Pure Prompt Entity Extraction",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Simplified authentication (remove if not needed)
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    if os.path.exists('config.yaml'):
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
        
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'], 
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
        
        if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
            authenticator.login(location='main')
            if not st.session_state.get('authentication_status'):
                st.stop()
        else:
            authenticator.logout("Logout", "sidebar")
            
except (ImportError, FileNotFoundError):
    # Skip authentication if not available
    pass

class PurePromptEntityExtractor:
    """
    Entity extraction using only comprehensive prompts - zero hardcoding.
    """
    
    def __init__(self):
        if 'extraction_cache' not in st.session_state:
            st.session_state.extraction_cache = {}
        
        self.api_calls_made = 0

    def extract_and_link_entities(self, text: str, domain_context: str = "") -> Dict[str, Any]:
        """
        Extract and link entities using a single comprehensive prompt.
        
        Args:
            text: Input text to analyze
            domain_context: Optional domain hint
            
        Returns:
            Complete results including entities with links
        """
        # Check cache
        cache_key = hashlib.md5(f"{text}_{domain_context}".encode()).hexdigest()
        if cache_key in st.session_state.extraction_cache:
            return st.session_state.extraction_cache[cache_key]
        
        # Create comprehensive prompt
        prompt = self._create_comprehensive_prompt(text, domain_context)
        
        # Send to language model
        response = self._call_language_model(prompt)
        
        if response:
            # Parse the comprehensive response
            results = self._parse_comprehensive_response(response, text)
            
            # Cache results
            st.session_state.extraction_cache[cache_key] = results
            
            return results
        
        return {"entities": [], "summary": "No entities extracted", "confidence": 0}

    def _create_comprehensive_prompt(self, text: str, domain_context: str) -> str:
        """Create a single comprehensive prompt that handles everything including geocoding."""
        
        domain_instruction = ""
        if domain_context:
            domain_instruction = f"\nDomain context: This text is about {domain_context}. Adjust your analysis accordingly."
        
        prompt = f"""You are an expert entity extraction, linking, and geocoding system. Analyze the following text and provide a comprehensive response.

{domain_instruction}

INSTRUCTIONS:
1. Extract all named entities (people, places, organizations, facilities, addresses)
2. For each entity, determine the most appropriate knowledge base links
3. For geographical entities, provide coordinates when possible
4. Provide linking priority: Pelagios (for historical/geographical) > Wikidata > Wikipedia
5. Include confidence scores and reasoning
6. Provide a summary of findings

TEXT TO ANALYZE:
{text}

RESPONSE FORMAT:
Return a JSON structure with this exact format:

{{
  "entities": [
    {{
      "text": "entity name as it appears in text",
      "type": "PERSON|ORGANIZATION|LOCATION|FACILITY|GPE|ADDRESS",
      "start_position": number,
      "confidence": 0.0-1.0,
      "reasoning": "why this is an entity and this type",
      "links": {{
        "pelagios_search": "https://pleiades.stoa.org/places/search?SearchableText=entityname",
        "wikidata_search": "suggested search term for wikidata",
        "wikipedia_search": "suggested search term for wikipedia",
        "priority": "pelagios|wikidata|wikipedia - which to try first"
      }},
      "coordinates": {{
        "latitude": number_or_null,
        "longitude": number_or_null,
        "source": "historical_knowledge|modern_knowledge|uncertain",
        "precision": "exact|approximate|region",
        "notes": "explanation of coordinate source and accuracy"
      }},
      "geographical_context": "if applicable, geographical region or context",
      "historical_period": "if applicable, time period or era"
    }}
  ],
  "summary": "brief summary of what entities were found",
  "text_domain": "inferred domain/topic of the text",
  "extraction_confidence": 0.0-1.0,
  "geographical_focus": "main geographical area if applicable",
  "time_period": "main time period if applicable"
}}

LINKING GUIDELINES:
- For ancient/historical places: prioritize Pelagios/Pleiades
- For modern entities: prioritize Wikidata/Wikipedia  
- For people: prioritize Wikipedia then Wikidata
- For organizations: prioritize Wikidata then Wikipedia
- Provide search terms that will find the most relevant results
- Consider context when determining link priority

GEOCODING GUIDELINES:
- Provide coordinates for any recognizable geographical entity
- For historical places, use your knowledge of ancient geography
- For modern places, use current geographical knowledge
- Indicate coordinate source: historical_knowledge, modern_knowledge, or uncertain
- Specify precision: exact (for specific sites), approximate (for general areas), region (for large areas)
- Include explanatory notes about coordinate accuracy and source
- Examples:
  * Ancient city: provide approximate historical location
  * Modern city: provide city center coordinates
  * Region/country: provide central coordinates with "region" precision
  * Uncertain locations: mark as uncertain with best estimate

COORDINATE EXAMPLES:
- "Argos": {{"latitude": 37.6333, "longitude": 22.7333, "source": "historical_knowledge", "precision": "approximate", "notes": "Ancient Greek city, approximate center of archaeological site"}}
- "London": {{"latitude": 51.5074, "longitude": -0.1278, "source": "modern_knowledge", "precision": "exact", "notes": "Modern city center coordinates"}}
- "Hellas": {{"latitude": 39.0742, "longitude": 21.8243, "source": "historical_knowledge", "precision": "region", "notes": "Ancient Greece, central coordinates of historical region"}}

QUALITY STANDARDS:
- Only extract clear, unambiguous named entities
- Provide honest confidence scores
- Include reasoning for each decision
- Suggest the most appropriate knowledge bases for each entity type
- For coordinates, be transparent about accuracy and source
- Don't guess coordinates for completely unknown places

JSON:"""

        return prompt

    def _call_language_model(self, prompt: str) -> Optional[str]:
        """Call language model with the comprehensive prompt."""
        
        models_to_try = [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "google/flan-t5-large",
            "microsoft/DialoGPT-medium"
        ]
        
        for model in models_to_try:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1000,
                        "temperature": 0.1,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 503:
                    st.warning(f"Model {model} is loading...")
                    time.sleep(5)
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        response_text = result[0].get("generated_text", "")
                        if response_text and len(response_text) > 50:
                            self.api_calls_made += 1
                            st.success(f"Response received from {model}")
                            return response_text
                
                st.warning(f"No valid response from {model}")
                
            except Exception as e:
                st.error(f"Error with {model}: {e}")
                continue
        
        st.error("All language models failed")
        return None

    def _parse_comprehensive_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """Parse the comprehensive JSON response from the language model."""
        
        # Try to extract JSON from response
        import re
        
        # Look for JSON structure
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Validate and enhance the parsed results
                if isinstance(parsed, dict) and 'entities' in parsed:
                    # Fix entity positions if needed
                    for entity in parsed.get('entities', []):
                        if 'text' in entity:
                            # Find actual position in text if not provided or incorrect
                            entity_text = entity['text']
                            actual_pos = original_text.find(entity_text)
                            if actual_pos != -1:
                                entity['start_position'] = actual_pos
                                entity['end_position'] = actual_pos + len(entity_text)
                            
                            # Generate actual links if URLs not provided
                            if 'links' in entity:
                                links = entity['links']
                                
                                # Generate Pleiades search URL
                                import urllib.parse
                                encoded_text = urllib.parse.quote(entity_text)
                                links['pelagios_search'] = f"https://pleiades.stoa.org/places/search?SearchableText={encoded_text}"
                                
                                # Generate Wikipedia URL
                                wiki_encoded = urllib.parse.quote(entity_text.replace(' ', '_'))
                                links['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{wiki_encoded}"
                    
                    return parsed
                
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
        
        # Fallback: try to extract entities from plain text
        return self._fallback_parse(response, original_text)

    def _fallback_parse(self, response: str, original_text: str) -> Dict[str, Any]:
        """Fallback parsing if JSON fails."""
        
        entities = []
        
        # Look for entity mentions in various formats
        import re
        
        # Try to find entity patterns in the response
        patterns = [
            r'"text":\s*"([^"]+)".*?"type":\s*"([^"]+)"',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([A-Z]+)\)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    entity_text, entity_type = match[0].strip(), match[1].strip().upper()
                    
                    # Find position in original text
                    start_pos = original_text.find(entity_text)
                    if start_pos != -1:
                        import urllib.parse
                        encoded_text = urllib.parse.quote(entity_text)
                        
                        entities.append({
                            "text": entity_text,
                            "type": entity_type,
                            "start_position": start_pos,
                            "end_position": start_pos + len(entity_text),
                            "confidence": 0.7,
                            "reasoning": "Extracted from model response",
                            "links": {
                                "pelagios_search": f"https://pleiades.stoa.org/places/search?SearchableText={encoded_text}",
                                "wikipedia_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(entity_text.replace(' ', '_'))}",
                                "priority": "pelagios" if entity_type in ["LOCATION", "GPE", "FACILITY"] else "wikipedia"
                            }
                        })
        
        return {
            "entities": entities,
            "summary": f"Fallback parsing found {len(entities)} entities",
            "extraction_confidence": 0.6,
            "text_domain": "unknown"
        }


class StreamlitApp:
    """Streamlit app for pure prompt entity extraction."""
    
    def __init__(self):
        self.extractor = PurePromptEntityExtractor()
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None

    def render_header(self):
        """Render application header."""
        # Logo
        try:
            if os.path.exists("logo.png"):
                st.image("logo.png", width=300)
        except:
            pass
        
        st.header("Pure Prompt Entity Extraction")
        st.markdown("**Zero hardcoding - everything handled by language model prompts**")

    def render_sidebar(self):
        """Render sidebar."""
        st.sidebar.subheader("Pure Prompt Approach")
        st.sidebar.info("• Single comprehensive prompt handles everything\n• No hardcoded patterns or rules\n• Language model determines entity types and links\n• Completely generic and adaptable")
        
        st.sidebar.subheader("Domain Context")
        domain = st.sidebar.selectbox(
            "Optional context hint:",
            ["", "historical", "academic", "business", "technical", "literary", "geographical"],
            help="Optional hint to help the language model understand context"
        )
        
        return domain

    def render_input(self):
        """Render input section."""
        st.subheader("Input Text")
        
        # Sample text
        sample = """The Persian learned men say that the Phoenicians were the cause of the dispute. These came to our seas from the sea which is called Red, and having settled in the country which they still occupy, at once began to make long voyages. Among other places to which they carried Egyptian and Assyrian merchandise, they came to Argos, which was at that time preeminent in every way among the people of what is now called Hellas. The Phoenicians came to Argos, and set out their cargo. On the fifth or sixth day after their arrival, when their wares were almost all sold, many women came to the shore and among them especially the daughter of the king, whose name was Io (according to Persians and Greeks alike), the daughter of Inachus."""
        
        text = st.text_area(
            "Text to analyze:",
            value=sample,
            height=200,
            help="The language model will extract entities and determine appropriate links"
        )
        
        return text

    def render_results(self, results):
        """Render results section."""
        if not results:
            return
        
        st.subheader("Results")
        
        # Summary
        if results.get('summary'):
            st.info(f"**Summary:** {results['summary']}")
        
        # Metrics
        entities = results.get('entities', [])
        geocoded_count = len([e for e in entities if e.get('coordinates', {}).get('latitude')])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entities Found", len(entities))
        with col2:
            confidence = results.get('extraction_confidence', 0)
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("Geocoded", geocoded_count)
        with col4:
            st.metric("API Calls", self.extractor.api_calls_made)
        
        # Context information
        if results.get('text_domain') or results.get('geographical_focus'):
            st.subheader("Context Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if results.get('text_domain'):
                    st.write(f"**Domain:** {results['text_domain']}")
            with col2:
                if results.get('geographical_focus'):
                    st.write(f"**Geography:** {results['geographical_focus']}")
        
        # Entities table
        if entities:
            st.subheader("Extracted Entities")
            
            df_data = []
            for entity in entities:
                links = entity.get('links', {})
                coords = entity.get('coordinates', {})
                priority = links.get('priority', 'unknown')
                
                row = {
                    'Entity': entity.get('text', ''),
                    'Type': entity.get('type', ''),
                    'Confidence': f"{entity.get('confidence', 0):.1%}",
                    'Coordinates': f"{coords.get('latitude', 'N/A')}, {coords.get('longitude', 'N/A')}" if coords.get('latitude') else 'None',
                    'Link Priority': priority,
                    'Reasoning': entity.get('reasoning', '')[:100] + "..." if len(entity.get('reasoning', '')) > 100 else entity.get('reasoning', '')
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Show links and coordinates for each entity
            with st.expander("Entity Links & Coordinates"):
                for entity in entities:
                    st.write(f"**{entity.get('text')}** ({entity.get('type')})")
                    
                    # Links
                    links = entity.get('links', {})
                    if links.get('pelagios_search'):
                        st.write(f"[Pleiades Search]({links['pelagios_search']})")
                    if links.get('wikipedia_url'):
                        st.write(f"[Wikipedia]({links['wikipedia_url']})")
                    if links.get('wikidata_search'):
                        st.write(f"Wikidata search: `{links['wikidata_search']}`")
                    
                    # Coordinates
                    coords = entity.get('coordinates', {})
                    if coords.get('latitude') and coords.get('longitude'):
                        lat, lon = coords['latitude'], coords['longitude']
                        precision = coords.get('precision', 'unknown')
                        source = coords.get('source', 'unknown')
                        notes = coords.get('notes', '')
                        
                        st.write(f"**Coordinates:** {lat}, {lon}")
                        st.write(f"   • **Precision:** {precision}")
                        st.write(f"   • **Source:** {source}")
                        if notes:
                            st.write(f"   • **Notes:** {notes}")
                        
                        # Show map link
                        map_url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=12"
                        st.write(f"   • [View on Map]({map_url})")
                    else:
                        st.write("No coordinates available")
                    
                    st.write("---")

    def run(self):
        """Run the application."""
        # Custom CSS
        st.markdown("""
        <style>
        .stApp { background-color: #F5F0DC !important; }
        .stButton > button {
            background-color: #C4A998 !important;
            color: black !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover {
            background-color: #B5998A !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.render_header()
        domain = self.render_sidebar()
        text = self.render_input()
        
        if st.button("Extract Entities with Pure Prompts", type="primary", use_container_width=True):
            if text.strip():
                with st.spinner("Sending comprehensive prompt to language model..."):
                    results = self.extractor.extract_and_link_entities(text, domain)
                    st.session_state.results = results
            else:
                st.warning("Please enter some text to analyze.")
        
        if st.session_state.results:
            self.render_results(st.session_state.results)


def main():
    """Main function."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
