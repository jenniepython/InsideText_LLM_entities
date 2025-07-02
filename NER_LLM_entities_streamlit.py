#!/usr/bin/env python3
"""
Sustainable Multi-Tier Entity Extraction

Progressive approach:
1. Efficient small prompts first (minimal energy)
2. Intelligent caching (zero repeat energy) 
3. Complex prompts only if simple ones fail (energy conscious)
4. Local fallbacks (zero energy)
5. Energy usage tracking and optimization
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import hashlib
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Sustainable Entity Extraction",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Simplified authentication
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
    pass

class SustainableEntityExtractor:
    """
    Multi-tier sustainable entity extraction with progressive complexity.
    """
    
    def __init__(self):
        # Initialize caches
        if 'entity_cache' not in st.session_state:
            st.session_state.entity_cache = {}
        if 'linking_cache' not in st.session_state:
            st.session_state.linking_cache = {}
        if 'geocoding_cache' not in st.session_state:
            st.session_state.geocoding_cache = {}
        
        # Energy tracking
        self.energy_metrics = {
            'tier1_calls': 0,  # Simple prompts
            'tier2_calls': 0,  # Medium prompts  
            'tier3_calls': 0,  # Complex prompts
            'cache_hits': 0,
            'local_fallbacks': 0,
            'total_tokens': 0
        }

    def extract_entities(self, text: str, domain_hint: str = "") -> List[Dict[str, Any]]:
        """
        Multi-tier extraction: simple → medium → complex → local fallback.
        """
        # Check cache first (zero energy)
        cache_key = hashlib.md5(f"{text}_{domain_hint}".encode()).hexdigest()
        if cache_key in st.session_state.entity_cache:
            self.energy_metrics['cache_hits'] += 1
            return st.session_state.entity_cache[cache_key]
        
        # Tier 1: Simple, efficient prompt (minimal energy)
        entities = self._tier1_simple_extraction(text, domain_hint)
        if entities and len(entities) >= 3:  # Good enough result
            st.session_state.entity_cache[cache_key] = entities
            return entities
        
        # Tier 2: Medium complexity prompt (moderate energy)
        entities = self._tier2_structured_extraction(text, domain_hint)
        if entities and len(entities) >= 2:  # Acceptable result
            st.session_state.entity_cache[cache_key] = entities
            return entities
        
        # Tier 3: Complex prompt only if really needed (high energy)
        entities = self._tier3_detailed_extraction(text, domain_hint)
        if entities:
            st.session_state.entity_cache[cache_key] = entities
            return entities
        
        # Local fallback: Pattern matching (zero energy)
        entities = self._local_fallback_extraction(text)
        self.energy_metrics['local_fallbacks'] += 1
        st.session_state.entity_cache[cache_key] = entities
        return entities

    def _tier1_simple_extraction(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Tier 1: Minimal, efficient prompt (20-30 tokens)."""
        
        domain_prefix = f"This is {domain_hint} text. " if domain_hint else ""
        
        prompt = f"""{domain_prefix}Find named entities:

{text[:500]}...

List: Name (Type)
Types: PERSON, PLACE, GROUP

Names:"""

        response = self._make_efficient_call(prompt, max_tokens=100, tier=1)
        if response:
            return self._parse_simple_format(response, text)
        return []

    def _tier2_structured_extraction(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Tier 2: Structured prompt (50-80 tokens)."""
        
        context = f" Focus on {domain_hint} entities." if domain_hint else ""
        
        prompt = f"""Extract entities from this text.{context}

Text: {text}

Format:
PERSON: [names]
PLACE: [locations] 
GROUP: [organizations]

Output:"""

        response = self._make_efficient_call(prompt, max_tokens=150, tier=2)
        if response:
            return self._parse_structured_format(response, text)
        return []

    def _tier3_detailed_extraction(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Tier 3: Detailed prompt only when needed (100+ tokens)."""
        
        prompt = f"""Analyze this text and extract named entities with types.

Domain: {domain_hint if domain_hint else 'general'}

Text: {text}

Return JSON array:
[{{"text": "name", "type": "PERSON|GPE|ORGANIZATION|LOCATION|FACILITY"}}]

Focus on clear, unambiguous entities only.

JSON:"""

        response = self._make_efficient_call(prompt, max_tokens=200, tier=3)
        if response:
            return self._parse_json_format(response, text)
        return []

    def _local_fallback_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Local pattern matching fallback (zero energy cost)."""
        
        entities = []
        
        # Simple capitalization patterns
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entity_text = match.group()
            
            # Skip common words
            if entity_text.lower() in ['the', 'this', 'that', 'when', 'where', 'which', 'what']:
                continue
            
            # Basic type inference from context
            context = text[max(0, match.start()-20):match.start()+len(entity_text)+20].lower()
            
            if any(indicator in context for indicator in ['king', 'daughter', 'son', 'according to']):
                entity_type = 'PERSON'
            elif any(indicator in context for indicator in ['came to', 'from', 'sea', 'country']):
                entity_type = 'GPE'
            else:
                entity_type = 'ORGANIZATION'
            
            entities.append({
                'text': entity_text,
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.6,
                'source': 'local_fallback'
            })
        
        # Limit to prevent noise
        return entities[:10]

    def _make_efficient_call(self, prompt: str, max_tokens: int, tier: int) -> Optional[str]:
        """Make efficient API call with tier-appropriate models."""
        
        # Use smaller models for lower tiers (more sustainable)
        model_tiers = {
            1: ["google/flan-t5-base"],  # Most efficient
            2: ["google/flan-t5-large"],  # Moderate
            3: ["mistralai/Mistral-7B-Instruct-v0.1"]  # Complex, use sparingly
        }
        
        models = model_tiers.get(tier, ["google/flan-t5-base"])
        
        for model in models:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": max_tokens,
                        "temperature": 0.1,
                        "do_sample": False  # Deterministic for efficiency
                    }
                }
                
                response = requests.post(url, json=payload, timeout=15)
                
                if response.status_code == 503:
                    continue  # Try next model
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        response_text = result[0].get("generated_text", "")
                        if response_text:
                            # Track energy usage
                            self.energy_metrics[f'tier{tier}_calls'] += 1
                            self.energy_metrics['total_tokens'] += len(prompt.split()) + len(response_text.split())
                            return response_text
                
            except Exception:
                continue
        
        return None

    def _parse_simple_format(self, response: str, text: str) -> List[Dict[str, Any]]:
        """Parse 'Name (Type)' format."""
        entities = []
        pattern = r'([A-Za-z\s]+)\s*\(([A-Z]+)\)'
        matches = re.findall(pattern, response)
        
        for name, entity_type in matches:
            name = name.strip()
            if len(name) > 1 and name in text:
                start_pos = text.find(name)
                entities.append({
                    'text': name,
                    'type': entity_type,
                    'start': start_pos,
                    'end': start_pos + len(name),
                    'confidence': 0.8,
                    'source': 'tier1_simple'
                })
        
        return entities

    def _parse_structured_format(self, response: str, text: str) -> List[Dict[str, Any]]:
        """Parse structured PERSON:/PLACE: format."""
        entities = []
        
        type_mapping = {
            'PERSON': 'PERSON',
            'PLACE': 'GPE', 
            'GROUP': 'ORGANIZATION'
        }
        
        for type_key, entity_type in type_mapping.items():
            pattern = f'{type_key}:\\s*\\[([^\\]]+)\\]'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                content = match.group(1)
                names = [name.strip() for name in content.split(',')]
                
                for name in names:
                    if len(name) > 1 and name in text:
                        start_pos = text.find(name)
                        entities.append({
                            'text': name,
                            'type': entity_type,
                            'start': start_pos,
                            'end': start_pos + len(name),
                            'confidence': 0.9,
                            'source': 'tier2_structured'
                        })
        
        return entities

    def _parse_json_format(self, response: str, text: str) -> List[Dict[str, Any]]:
        """Parse JSON format from tier 3."""
        entities = []
        
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    if isinstance(item, dict) and 'text' in item:
                        name = item['text'].strip()
                        if name in text:
                            start_pos = text.find(name)
                            entities.append({
                                'text': name,
                                'type': item.get('type', 'UNKNOWN'),
                                'start': start_pos,
                                'end': start_pos + len(name),
                                'confidence': 0.95,
                                'source': 'tier3_detailed'
                            })
            except json.JSONDecodeError:
                pass
        
        return entities

    def add_efficient_linking(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add efficient linking with caching."""
        
        for entity in entities:
            cache_key = f"link_{entity['text']}"
            
            # Check cache first
            if cache_key in st.session_state.linking_cache:
                entity.update(st.session_state.linking_cache[cache_key])
                self.energy_metrics['cache_hits'] += 1
                continue
            
            # Generate efficient links
            import urllib.parse
            encoded_text = urllib.parse.quote(entity['text'])
            
            links = {
                'pleiades_search': f"https://pleiades.stoa.org/places/search?SearchableText={encoded_text}",
                'wikipedia_url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(entity['text'].replace(' ', '_'))}"
            }
            
            entity.update(links)
            st.session_state.linking_cache[cache_key] = links

        return entities

    def add_efficient_geocoding(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add basic geocoding for place entities."""
        
        for entity in entities:
            if entity['type'] not in ['GPE', 'LOCATION', 'FACILITY']:
                continue
            
            cache_key = f"geo_{entity['text']}"
            
            # Check cache
            if cache_key in st.session_state.geocoding_cache:
                entity.update(st.session_state.geocoding_cache[cache_key])
                self.energy_metrics['cache_hits'] += 1
                continue
            
            # Simple geocoding lookup (minimal energy)
            coords = self._simple_geocoding_lookup(entity['text'])
            if coords:
                entity.update(coords)
                st.session_state.geocoding_cache[cache_key] = coords

        return entities

    def _simple_geocoding_lookup(self, place_name: str) -> Optional[Dict[str, Any]]:
        """Simple, efficient geocoding."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': place_name,
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    return {
                        'latitude': float(result['lat']),
                        'longitude': float(result['lon']),
                        'location_name': result['display_name'],
                        'geocoding_source': 'openstreetmap_simple'
                    }
        except Exception:
            pass
        
        return None

    def get_sustainability_metrics(self) -> Dict[str, Any]:
        """Get detailed sustainability metrics."""
        total_calls = sum([
            self.energy_metrics['tier1_calls'],
            self.energy_metrics['tier2_calls'], 
            self.energy_metrics['tier3_calls']
        ])
        
        return {
            'total_api_calls': total_calls,
            'tier1_calls': self.energy_metrics['tier1_calls'],
            'tier2_calls': self.energy_metrics['tier2_calls'],
            'tier3_calls': self.energy_metrics['tier3_calls'],
            'cache_hits': self.energy_metrics['cache_hits'],
            'local_fallbacks': self.energy_metrics['local_fallbacks'],
            'total_tokens': self.energy_metrics['total_tokens'],
            'efficiency_ratio': self.energy_metrics['cache_hits'] / max(1, total_calls + self.energy_metrics['cache_hits'])
        }


class SustainableApp:
    """Sustainable Streamlit application."""
    
    def __init__(self):
        self.extractor = SustainableEntityExtractor()
        
        if 'results' not in st.session_state:
            st.session_state.results = None

    def render_header(self):
        """Render header with sustainability focus."""
        try:
            if os.path.exists("logo.png"):
                st.image("logo.png", width=300)
        except:
            pass
        
        st.header("Sustainable Entity Extraction")
        st.markdown("**Progressive complexity: simple prompts first, complex only when needed**")
        
        # Show sustainability metrics only after activity
        metrics = self.extractor.get_sustainability_metrics()
        if metrics['total_api_calls'] > 0 or metrics['cache_hits'] > 0:
            st.subheader("Sustainability Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("API Calls", metrics['total_api_calls'])
            with col2:
                st.metric("Cache Hits", metrics['cache_hits'])
            with col3:
                st.metric("Efficiency", f"{metrics['efficiency_ratio']:.1%}")
            with col4:
                st.metric("Tokens Used", metrics['total_tokens'])
            
            # Tier breakdown
            with st.expander("Energy Usage Breakdown"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tier 1 (Efficient)", metrics['tier1_calls'])
                with col2:
                    st.metric("Tier 2 (Moderate)", metrics['tier2_calls'])
                with col3:
                    st.metric("Tier 3 (Complex)", metrics['tier3_calls'])
                
                st.metric("Local Fallbacks (Zero Energy)", metrics['local_fallbacks'])

    def render_sidebar(self):
        """Render sidebar."""
        st.sidebar.subheader("Sustainable Approach")
        st.sidebar.info("• Tier 1: Simple prompts (20-30 tokens)\n• Tier 2: Structured prompts (50-80 tokens)\n• Tier 3: Complex prompts (100+ tokens)\n• Local fallback: Zero energy patterns\n• Intelligent caching throughout")
        
        st.sidebar.subheader("Domain Context")
        domain = st.sidebar.selectbox(
            "Optional domain hint:",
            ["", "historical", "academic", "business", "technical", "geographical"]
        )
        
        return domain

    def render_input(self):
        """Render input section."""
        st.subheader("Input Text")
        
        sample = """The Persian learned men say that the Phoenicians were the cause of the dispute. These came to our seas from the sea which is called Red, and having settled in the country which they still occupy, at once began to make long voyages. Among other places to which they carried Egyptian and Assyrian merchandise, they came to Argos, which was at that time preeminent in every way among the people of what is now called Hellas."""
        
        text = st.text_area(
            "Text to analyze:",
            value=sample,
            height=150,
            help="Uses progressive complexity: simple extraction first, complex only if needed"
        )
        
        return text

    def render_results(self, results):
        """Render results."""
        if not results:
            return
        
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entities Found", len(results))
        with col2:
            geocoded = len([e for e in results if e.get('latitude')])
            st.metric("Geocoded", geocoded)
        with col3:
            sources = set(e.get('source', 'unknown') for e in results)
            st.metric("Sources Used", len(sources))
        
        # Results table
        if results:
            df_data = []
            for entity in results:
                row = {
                    'Entity': entity['text'],
                    'Type': entity['type'],
                    'Confidence': f"{entity.get('confidence', 0):.1%}",
                    'Source': entity.get('source', 'unknown'),
                    'Coordinates': f"{entity.get('latitude', 'N/A')}, {entity.get('longitude', 'N/A')}" if entity.get('latitude') else 'None'
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

    def run(self):
        """Run the sustainable application."""
        # Farrow & Ball styling
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
        .stButton > button:hover { background-color: #B5998A !important; }
        </style>
        """, unsafe_allow_html=True)
        
        self.render_header()
        domain = self.render_sidebar()
        text = self.render_input()
        
        if st.button("Extract Entities Sustainably", type="primary", use_container_width=True):
            if text.strip():
                with st.spinner("Using progressive extraction tiers..."):
                    # Extract entities
                    entities = self.extractor.extract_entities(text, domain)
                    
                    # Add linking and geocoding
                    entities = self.extractor.add_efficient_linking(entities)
                    entities = self.extractor.add_efficient_geocoding(entities)
                    
                    st.session_state.results = entities
                    
                    # Show which tier was used
                    metrics = self.extractor.get_sustainability_metrics()
                    if metrics['tier1_calls'] > 0:
                        st.success("Tier 1 (efficient) extraction succeeded!")
                    elif metrics['tier2_calls'] > 0:
                        st.info("Tier 2 (moderate) extraction used")
                    elif metrics['tier3_calls'] > 0:
                        st.warning("Tier 3 (complex) extraction required")
                    else:
                        st.info("Local fallback patterns used (zero energy)")
            else:
                st.warning("Please enter text to analyze.")
        
        if st.session_state.results:
            self.render_results(st.session_state.results)


def main():
    """Main function."""
    app = SustainableApp()
    app.run()


if __name__ == "__main__":
    main()
