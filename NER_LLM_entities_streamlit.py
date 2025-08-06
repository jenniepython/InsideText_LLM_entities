#!/usr/bin/env python3
"""
Streamlit LLM Entity Linker Application

A web interface for entity extraction using LLM (Gemini) with linking and geocoding.
This application provides the same look and feel as the NLTK version but uses
LLM for entity recognition.

Author: Enhanced from NER_LLM_entities_streamlit.py
Version: 2.0
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="From Text to Linked Data using LLM",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Add custom CSS for Farrow & Ball Slipper Satin background - same as NLTK app
st.markdown("""
<style>
.stApp {
    background-color: #F5F0DC !important;
}
.main .block-container {
    background-color: #F5F0DC !important;
}
.stSidebar {
    background-color: #F5F0DC !important;
}
.stSelectbox > div > div {
    background-color: white !important;
}
.stTextInput > div > div > input {
    background-color: white !important;
}
.stTextArea > div > div > textarea {
    background-color: white !important;
}
.stExpander {
    background-color: white !important;
    border: 1px solid #E0D7C0 !important;
    border-radius: 4px !important;
}
.stDataFrame {
    background-color: white !important;
}
.stButton > button {
    background-color: #C4A998 !important;
    color: black !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background-color: #B5998A !important;
    color: black !important;
}
.stButton > button:active {
    background-color: #A68977 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
import pandas as pd
import json
import os
import re
import time
import requests
from datetime import datetime
import urllib.parse
import html as html_module
from typing import List, Dict, Any
import hashlib

# LLM Model configuration - ONLY Gemini 1.5 Flash
MODEL_OPTIONS = {
    "Gemini 1.5 Flash": {
        "model_name": "gemini-1.5-flash",
        "description": "Google's lightweight LLM for fast generation",
        "provider": "google"
    }
}

class LLMEntityLinker:
    """
    Main class for LLM-based entity linking functionality.
    
    This class uses Gemini for entity extraction and provides the same
    linking and geocoding capabilities as the NLTK version.
    """
    
    def __init__(self):
        """Initialize the LLM Entity Linker."""
        # Color scheme for different entity types in HTML output - expanded for more entity types (removed QUANTITY)
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground. 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA',         # F&B Oxford stone
            'PRODUCT': '#E6D7C9',         # F&B Skimming stone
            'EVENT': '#D4C5B9',          # F&B Mouse's back
            'WORK_OF_ART': '#E8E1D4',    # F&B Strong white
            'LANGUAGE': '#F0EAE2',       # F&B Pointing
            'LAW': '#DDD6CE',            # F&B Elephant's breath (lighter)
            'DATE': '#E3DDD7',          # F&B Dimity
            'MONEY': '#D6CFCA'          # F&B Joa's white
        }

    def construct_ner_prompt(self, text):
        """Construct an improved NER prompt for Gemini with few-shot examples relevant to digital humanities and historical texts."""
        
        # Few-shot examples tailored for digital humanities and historical texts
        examples = """
Examples:

Text: "The Whitechapel Bell Foundry was established in 1570 and cast the Liberty Bell for Philadelphia. Master founder Robert Mot created bells for St. Paul's Cathedral in the 17th century."
Output:
[
    {"text": "Whitechapel Bell Foundry", "type": "ORGANIZATION", "start_pos": 4},
    {"text": "1570", "type": "DATE", "start_pos": 44},
    {"text": "Liberty Bell", "type": "WORK_OF_ART", "start_pos": 62},
    {"text": "Philadelphia", "type": "GPE", "start_pos": 79},
    {"text": "Robert Mot", "type": "PERSON", "start_pos": 108},
    {"text": "St. Paul's Cathedral", "type": "FACILITY", "start_pos": 139},
    {"text": "17th century", "type": "DATE", "start_pos": 167}
]

Text: "In the manuscript British Library MS Cotton Vitellius A.xv, Beowulf battles the dragon. The codex dates to circa 1000 CE and contains Old English verse."
Output:
[
    {"text": "British Library MS Cotton Vitellius A.xv", "type": "WORK_OF_ART", "start_pos": 17},
    {"text": "Beowulf", "type": "PERSON", "start_pos": 60},
    {"text": "circa 1000 CE", "type": "DATE", "start_pos": 108},
    {"text": "Old English", "type": "LANGUAGE", "start_pos": 135}
]

Text: "The Theatre Royal Drury Lane staged Richard Sheridan's The School for Scandal in 1777. David Garrick had previously managed this playhouse from 1747 to 1776."
Output:
[
    {"text": "Theatre Royal Drury Lane", "type": "FACILITY", "start_pos": 4},
    {"text": "Richard Sheridan", "type": "PERSON", "start_pos": 36},
    {"text": "The School for Scandal", "type": "WORK_OF_ART", "start_pos": 55},
    {"text": "1777", "type": "DATE", "start_pos": 81},
    {"text": "David Garrick", "type": "PERSON", "start_pos": 87},
    {"text": "1747", "type": "DATE", "start_pos": 135},
    {"text": "1776", "type": "DATE", "start_pos": 143}
]
"""
        
        prompt = f"""You are an expert named entity recognition system specialized in historical documents, manuscripts, and cultural heritage materials. Your task is to identify and extract ALL relevant entities from historical and literary texts.

TASK DEFINITION:
Extract entities with their exact positions in the text. Be thorough and identify as many relevant entities as possible, paying special attention to historical names, places, dates, and cultural artifacts.

ENTITY TYPES TO IDENTIFY:
- PERSON: Historical figures, authors, artists, craftsmen, performers, rulers, religious figures
- ORGANIZATION: Historical institutions, guilds, companies, religious orders, theatrical companies
- GPE: Historical places, cities, kingdoms, regions, counties, parishes, districts
- LOCATION: Geographic locations, neighborhoods, historical sites, battlefields, landmarks
- FACILITY: Historical buildings, theaters, churches, castles, bridges, workshops, stages
- ADDRESS: Historical addresses, street names, property descriptions
- PRODUCT: Historical artifacts, manufactured goods, instruments, tools, materials
- EVENT: Historical events, performances, ceremonies, festivals, wars, meetings
- WORK_OF_ART: Plays, operas, paintings, sculptures, manuscripts, books, musical compositions
- LANGUAGE: Historical languages, dialects, scripts, linguistic terms
- LAW: Historical laws, charters, acts, regulations, legal documents
- DATE: Historical dates, periods, reigns, centuries, years, eras
- MONEY: Historical currencies, amounts, prices, wages, costs

IMPORTANT INSTRUCTIONS FOR HISTORICAL TEXTS:
1. Extract ALL entities you can identify - be comprehensive and thorough
2. Include historical terminology and period-specific language
3. Capture complete historical names and titles (e.g., "Theatre Royal Drury Lane", "Master founder")
4. Include manuscript references, architectural features, and technical terms
5. Don't miss dates, measurements, or specialized vocabulary
6. Pay attention to historical context and period-appropriate entities
7. Include both major and minor historical figures, places, and events

{examples}

Now extract entities from this historical text:

Text: "{text}"

Output (JSON array only):
"""
        return prompt

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response with improved parsing."""
        response_text = response_text.strip()
        
        # Try to find JSON array patterns
        json_patterns = [
            r'\[.*?\]',  # Array pattern
            r'\{.*?\}'   # Object pattern
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
        
        # If no JSON found, try to extract from code blocks
        code_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        return [parsed]
                except json.JSONDecodeError:
                    continue
        
        return None

    def extract_entities(self, text: str):
        """Extract named entities from text using Gemini LLM."""
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found!")
                return []
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = self.construct_ner_prompt(text)
            gemini_response = model.generate_content(prompt)
            llm_response = gemini_response.text
            
            entities_raw = self.extract_json_from_response(llm_response)
            if not entities_raw:
                st.warning("Could not parse JSON from Gemini response.")
                return []
            
            # Convert to consistent format and remove duplicates
            entities = []
            seen_entities = set()  # Track (text, type) pairs to avoid duplicates
            
            for entity_raw in entities_raw:
                if 'text' in entity_raw and 'type' in entity_raw:
                    entity_text = entity_raw['text'].strip()
                    entity_type = entity_raw['type']
                    
                    # Create unique key for duplicate detection
                    unique_key = (entity_text.lower(), entity_type)
                    
                    # Skip if we've already seen this entity
                    if unique_key in seen_entities:
                        continue
                    
                    seen_entities.add(unique_key)
                    
                    # Find actual position in text
                    start_pos = text.find(entity_text)
                    if start_pos == -1:
                        # Try with start_pos from LLM if provided
                        start_pos = entity_raw.get('start_pos', 0)
                    
                    entity = {
                        'text': entity_text,
                        'type': entity_type,
                        'start': start_pos,
                        'end': start_pos + len(entity_text)
                    }
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            st.error(f"Error in LLM entity extraction: {e}")
            return []

    def analyze_text_context(self, text: str) -> Dict[str, Any]:
        """Analyze text to determine historical, geographical, and cultural context."""
        context = {
            'period': None,
            'region': None,
            'culture': None,
            'subject_matter': None,
            'language_style': 'modern'
        }
        
        text_lower = text.lower()
        
        # Detect historical periods
        period_indicators = {
            'ancient': ['ancient', 'bc', 'bce', 'antiquity', 'classical', 'pharaoh', 'caesar', 'emperor', 'temple', 'oracle'],
            'medieval': ['medieval', 'middle ages', 'feudal', 'knight', 'monastery', 'crusade', 'pilgrim', 'manuscript'],
            'renaissance': ['renaissance', '15th century', '16th century', 'leonardo', 'michelangelo', 'medici'],
            'early_modern': ['17th century', '18th century', 'enlightenment', 'baroque', 'reformation'],
            'victorian': ['victorian', '19th century', 'industrial revolution', 'steam', 'railway', 'empire'],
            'modern': ['20th century', '21st century', 'world war', 'internet', 'digital']
        }
        
        for period, indicators in period_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                context['period'] = period
                break
        
        # Detect geographical/cultural regions
        region_indicators = {
            'ancient_greece': ['greece', 'greek', 'athens', 'sparta', 'delphi', 'olympia', 'hellas', 'argos', 'thebes', 'corinth', 'phoenician'],
            'ancient_rome': ['rome', 'roman', 'latin', 'caesar', 'senate', 'consul', 'legion', 'colosseum'],
            'ancient_egypt': ['egypt', 'egyptian', 'pharaoh', 'nile', 'pyramid', 'hieroglyph', 'alexandria'],
            'ancient_persia': ['persia', 'persian', 'zoroaster', 'cyrus', 'darius'],
            'mesopotamia': ['babylon', 'assyria', 'mesopotamia', 'tigris', 'euphrates'],
            'britain': ['britain', 'british', 'england', 'london', 'scotland', 'wales', 'uk'],
            'france': ['france', 'french', 'paris', 'versailles'],
            'germany': ['germany', 'german', 'berlin', 'bavaria'],
            'italy': ['italy', 'italian', 'florence', 'venice', 'milan', 'naples']
        }
        
        for region, indicators in region_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                context['region'] = region
                break
        
        # Detect subject matter
        subject_indicators = {
            'theater': ['theatre', 'theater', 'stage', 'play', 'drama', 'actor', 'performance', 'audience'],
            'architecture': ['building', 'architecture', 'cathedral', 'church', 'castle', 'palace'],
            'literature': ['book', 'manuscript', 'author', 'poem', 'novel', 'text', 'writing'],
            'history': ['battle', 'war', 'king', 'queen', 'empire', 'dynasty', 'reign'],
            'mythology': ['myth', 'legend', 'god', 'goddess', 'hero', 'oracle', 'prophecy']
        }
        
        for subject, indicators in subject_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                context['subject_matter'] = subject
                break
        
        # Detect language style
        if any(word in text_lower for word in ['thee', 'thou', 'thy', 'hath', 'doth', 'forsooth']):
            context['language_style'] = 'archaic'
        elif any(word in text_lower for word in ['learned men say', 'it is said', 'according to']):
            context['language_style'] = 'historical_narrative'
        
        return context
        """
        Detect geographical context from the text to improve geocoding accuracy.
        Dynamic approach without hardcoded mappings.
        """
        context_clues = []
        text_lower = text.lower()
        
        # Extract from entities that are already identified as places
        geographical_entities = []
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION', 'FACILITY']:
                geographical_entities.append(entity['text'].lower())
        
        # Look for common geographical indicators in the text
        # Countries - common ones that appear frequently
        common_countries = ['uk', 'united kingdom', 'britain', 'great britain', 'england', 'scotland', 'wales', 'northern ireland',
                           'usa', 'united states', 'america', 'us', 'canada', 'australia', 'france', 'germany', 'italy', 
                           'spain', 'japan', 'china', 'india', 'brazil', 'russia', 'mexico', 'netherlands', 'belgium', 
                           'switzerland', 'austria', 'sweden', 'norway', 'denmark', 'poland', 'portugal', 'greece',
                           'ireland', 'finland', 'czech republic', 'hungary', 'romania', 'bulgaria', 'croatia',
                           'south africa', 'egypt', 'israel', 'turkey', 'iran', 'iraq', 'saudi arabia', 'uae',
                           'thailand', 'vietnam', 'malaysia', 'singapore', 'indonesia', 'philippines', 'south korea',
                           'north korea', 'taiwan', 'hong kong', 'new zealand', 'argentina', 'chile', 'colombia',
                           'peru', 'venezuela', 'ecuador', 'bolivia', 'uruguay', 'paraguay']
        
        # Major cities - extract dynamically from entities and common patterns
        for country in common_countries:
            if country in text_lower:
                context_clues.append(country)
        
        # Add geographical entities found by the LLM
        for geo_entity in geographical_entities:
            if geo_entity not in context_clues:
                context_clues.append(geo_entity)
        
        # Look for postal codes to infer country
        postal_patterns = {
            'uk': [
                r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK postcodes
                r'\b[A-Z]{2}\d{1,2}\s*\d[A-Z]{2}\b'
            ],
            'usa': [
                r'\b\d{5}(-\d{4})?\b'  # US ZIP codes
            ],
            'canada': [
                r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b'  # Canadian postal codes
            ]
        }
        
        for country, patterns in postal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if country not in context_clues:
                        context_clues.append(country)
                    break
        
        # Return the most relevant context clues (limit to avoid over-constraining)
        return context_clues[:3]

    def get_coordinates(self, entities, processed_text=""):
        """Enhanced coordinate lookup with geographical context detection."""
        # Detect geographical context from the full text
        context_clues = self._detect_geographical_context(processed_text, entities)
        
        if context_clues:
            print(f"Detected geographical context: {', '.join(context_clues)}")
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try geocoding with context
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Fall back to OpenStreetMap
                if self._try_openstreetmap(entity):
                    continue
                    
                # If still no coordinates, try a more aggressive search
                self._try_aggressive_geocoding(entity)
        
        return entities

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context."""
        if not context_clues:
            return False
        
        # Create context-aware search terms
        search_variations = [entity['text']]
        
        # Add context to search terms
        for context in context_clues:
            context_mapping = {
                'uk': ['UK', 'United Kingdom', 'England', 'Britain'],
                'usa': ['USA', 'United States', 'US'],
                'canada': ['Canada'],
                'australia': ['Australia'],
                'france': ['France'],
                'germany': ['Germany'],
                'london': ['London, UK', 'London, England'],
                'new york': ['New York, USA', 'New York, NY'],
                'paris': ['Paris, France'],
                'tokyo': ['Tokyo, Japan'],
                'sydney': ['Sydney, Australia'],
            }
            
            context_variants = context_mapping.get(context, [context])
            for variant in context_variants:
                search_variations.append(f"{entity['text']}, {variant}")
        
        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))
        
        # Try OpenStreetMap with context
        for search_term in search_variations[:3]:  # Try top 3 with OSM
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_contextual'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.3)  # Rate limiting
            except Exception:
                continue
        
        return False

    def _try_openstreetmap(self, entity):
        """Fall back to direct OpenStreetMap Nominatim API."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/1.0'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            pass
        
        return False

    def _try_aggressive_geocoding(self, entity):
        """Try more aggressive geocoding with different search terms."""
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",
            f"{entity['text']}, England",
            f"{entity['text']}, Scotland",
            f"{entity['text']}, Wales",
            f"{entity['text']} city",
            f"{entity['text']} town"
        ]
        
        for search_term in search_variations:
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_aggressive'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.2)  # Rate limiting between attempts
            except Exception:
                continue
        
        return False

    def link_to_wikidata(self, entities):
        """Add basic Wikidata linking."""
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'search': entity['text'],
                    'language': 'en',
                    'limit': 1,
                    'type': 'item'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search') and len(data['search']) > 0:
                        result = data['search'][0]
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                        entity['wikidata_description'] = result.get('description', '')
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Detect geographical context from the text to improve geocoding accuracy.
        Dynamic approach without hardcoded mappings.
        """
        context_clues = []
        text_lower = text.lower()
        
        # Extract from entities that are already identified as places
        geographical_entities = []
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION', 'FACILITY']:
                geographical_entities.append(entity['text'].lower())
        
        # Look for common geographical indicators in the text
        # Countries - common ones that appear frequently
        common_countries = ['uk', 'united kingdom', 'britain', 'great britain', 'england', 'scotland', 'wales', 'northern ireland',
                           'usa', 'united states', 'america', 'us', 'canada', 'australia', 'france', 'germany', 'italy', 
                           'spain', 'japan', 'china', 'india', 'brazil', 'russia', 'mexico', 'netherlands', 'belgium', 
                           'switzerland', 'austria', 'sweden', 'norway', 'denmark', 'poland', 'portugal', 'greece',
                           'ireland', 'finland', 'czech republic', 'hungary', 'romania', 'bulgaria', 'croatia',
                           'south africa', 'egypt', 'israel', 'turkey', 'iran', 'iraq', 'saudi arabia', 'uae',
                           'thailand', 'vietnam', 'malaysia', 'singapore', 'indonesia', 'philippines', 'south korea',
                           'north korea', 'taiwan', 'hong kong', 'new zealand', 'argentina', 'chile', 'colombia',
                           'peru', 'venezuela', 'ecuador', 'bolivia', 'uruguay', 'paraguay']
        
        # Major cities - extract dynamically from entities and common patterns
        for country in common_countries:
            if country in text_lower:
                context_clues.append(country)
        
        # Add geographical entities found by the LLM
        for geo_entity in geographical_entities:
            if geo_entity not in context_clues:
                context_clues.append(geo_entity)
        
        # Look for postal codes to infer country
        postal_patterns = {
            'uk': [
                r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK postcodes
                r'\b[A-Z]{2}\d{1,2}\s*\d[A-Z]{2}\b'
            ],
            'usa': [
                r'\b\d{5}(-\d{4})?\b'  # US ZIP codes
            ],
            'canada': [
                r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b'  # Canadian postal codes
            ]
        }
        
        for country, patterns in postal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if country not in context_clues:
                        context_clues.append(country)
                    break
        
        # Return the most relevant context clues (limit to avoid over-constraining)
        return context_clues[:3]

    def link_to_wikipedia_contextual(self, entities, text_context):
        """Add contextual Wikipedia linking based on text analysis."""
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Create context-aware search terms
                search_terms = [entity['text']]
                
                # Add contextual modifiers based on text analysis
                if text_context['region'] == 'ancient_greece' and entity['type'] in ['GPE', 'LOCATION']:
                    search_terms.extend([
                        f"{entity['text']} ancient Greece",
                        f"{entity['text']} ancient Greek city",
                        f"ancient {entity['text']}"
                    ])
                elif text_context['region'] == 'ancient_rome' and entity['type'] in ['GPE', 'LOCATION']:
                    search_terms.extend([
                        f"{entity['text']} ancient Rome",
                        f"{entity['text']} Roman",
                        f"ancient {entity['text']}"
                    ])
                elif text_context['period'] == 'ancient' and entity['type'] == 'PERSON':
                    search_terms.extend([
                        f"{entity['text']} ancient",
                        f"{entity['text']} mythology",
                        f"{entity['text']} classical"
                    ])
                elif text_context['period'] == 'victorian' and entity['type'] in ['FACILITY', 'ORGANIZATION']:
                    search_terms.extend([
                        f"{entity['text']} Victorian",
                        f"{entity['text']} 19th century"
                    ])
                elif text_context['subject_matter'] == 'theater' and entity['type'] in ['FACILITY', 'PERSON']:
                    search_terms.extend([
                        f"{entity['text']} theatre",
                        f"{entity['text']} theater",
                        f"{entity['text']} drama"
                    ])
                
                # Try each search term
                for search_term in search_terms[:3]:  # Limit to top 3
                    search_url = "https://en.wikipedia.org/w/api.php"
                    search_params = {
                        'action': 'query',
                        'format': 'json',
                        'list': 'search',
                        'srsearch': search_term,
                        'srlimit': 1
                    }
                    
                    headers = {'User-Agent': 'EntityLinker/1.0'}
                    response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('query', {}).get('search'):
                            result = data['query']['search'][0]
                            page_title = result['title']
                            
                            # Create Wikipedia URL
                            encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                            entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                            entity['wikipedia_title'] = page_title
                            
                            # Get a snippet/description from the search result
                            if result.get('snippet'):
                                snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                                entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                            
                            # Mark which search term worked
                            entity['search_context'] = search_term
                            break
                    
                    time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                pass
        
        return entities
        """Add Wikipedia linking for entities without Wikidata links."""
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Use Wikipedia's search API
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': entity['text'],
                    'srlimit': 1
                }
                
                headers = {'User-Agent': 'EntityLinker/1.0'}
                response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        # Get the first search result
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        # Create Wikipedia URL
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                        entity['wikipedia_title'] = page_title
                        
                        # Get a snippet/description from the search result
                        if result.get('snippet'):
                            # Clean up the snippet (remove HTML tags)
                            snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                            entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add basic Britannica linking.""" 
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                params = {'query': entity['text']}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Look for article links
                    pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                    matches = re.findall(pattern, response.text)
                    
                    for url_path, link_text in matches:
                        if (entity['text'].lower() in link_text.lower() or 
                            link_text.lower() in entity['text'].lower()):
                            entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                            entity['britannica_title'] = link_text.strip()
                            break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses."""
        for entity in entities:
            # Only process ADDRESS entities
            if entity['type'] != 'ADDRESS':
                continue
                
            try:
                # Search OpenStreetMap Nominatim for the address
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': entity['text'],
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
                
                response = requests.get(url, params=params, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        # Create OpenStreetMap link
                        lat = result['lat']
                        lon = result['lon']
                        entity['openstreetmap_url'] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
                        entity['openstreetmap_display_name'] = result['display_name']
                        
                        # Also add coordinates
                        entity['latitude'] = float(lat)
                        entity['longitude'] = float(lon)
                        entity['location_name'] = result['display_name']
                
                time.sleep(0.2)  # Rate limiting
            except Exception:
                pass
        
        return entities


class StreamlitLLMEntityLinker:
    """
    Streamlit wrapper for the LLM Entity Linker class.
    
    Provides the same interface as the NLTK version but uses LLM for entity extraction.
    """
    
    def __init__(self):
        """Initialize the Streamlit LLM Entity Linker."""
        self.entity_linker = LLMEntityLinker()
        
        # Initialize session state
        if 'entities' not in st.session_state:
            st.session_state.entities = []
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""
        if 'html_content' not in st.session_state:
            st.session_state.html_content = ""
        if 'analysis_title' not in st.session_state:
            st.session_state.analysis_title = "text_analysis"
        if 'last_processed_hash' not in st.session_state:
            st.session_state.last_processed_hash = ""

    @st.cache_data
    def cached_extract_entities(_self, text: str) -> str:
        """Cached entity extraction to avoid reprocessing same text."""
        entities = _self.entity_linker.extract_entities(text)
        return json.dumps(entities, default=str)
    
    @st.cache_data  
    def cached_link_to_wikidata(_self, entities_json: str) -> str:
        """Cached Wikidata linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikidata(entities)
        return json.dumps(linked_entities, default=str)
    
    @st.cache_data
    def cached_link_to_britannica(_self, entities_json: str) -> str:
        """Cached Britannica linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_britannica(entities)
        return json.dumps(linked_entities, default=str)

    @st.cache_data
    def cached_link_to_wikipedia(_self, entities_json: str) -> str:
        """Cached Wikipedia linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikipedia(entities)
        return json.dumps(linked_entities, default=str)

    def render_header(self):
        """Render the application header with logo - same as NLTK app."""
        # Display logo if it exists
        try:
            # Try to load and display the logo
            logo_path = "logo.png"  # You can change this filename as needed
            if os.path.exists(logo_path):
                # Logo naturally aligns to the left without columns
                st.image(logo_path, width=300)  # Adjust width as needed
            else:
                # If logo file doesn't exist, show a placeholder or message
                st.info("💡 Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            # If there's any error loading the logo, continue without it
            st.warning(f"Could not load logo: {e}")        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using LLM")
        st.markdown("**Extract and link named entities from text using Gemini LLM**")
        
        # Create a simple process diagram - same as NLTK app but with LLM
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Gemini LLM Entity Recognition</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Link to Knowledge Bases:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata</strong><br><small>Structured knowledge</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikipedia/Britannica</strong><br><small>Encyclopedia articles</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Geocoding</strong><br><small>Coordinates & locations</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Output Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #EFCA89;">
                         <strong>JSON-LD Export</strong><br><small>Structured data format</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #C3B5AC;">
                         <strong>HTML Export</strong><br><small>Portable web format</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with minimal information - same as NLTK app."""
        # Entity linking information
        st.sidebar.subheader("Entity Linking & Geocoding")
        st.sidebar.info("Entities are extracted using Gemini LLM and linked to Wikidata first, then Wikipedia, then Britannica as fallbacks. Places and addresses are geocoded using multiple services for accurate coordinates.")

    def render_input_section(self):
        """Render the text input section - same as NLTK app."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Sample text for demonstration
        sample_text = ""       
        # Text input area - always shown and editable
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,  # Pre-populate with sample text
            height=200,  # Reduced height for mobile
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        # File upload option in expander for mobile
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md) to replace the text above"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text  # Override the text area content
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    # Set default title from filename if no title provided
                    if not analysis_title:
                        import os
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Use suggested title if no title provided
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "text_analysis"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """
        Process the input text using the LLM EntityLinker with contextual analysis.
        
        Args:
            text: Input text to process
            title: Analysis title
        """
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Check if we've already processed this exact text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text and extracting entities..."):
            try:
                # Create a progress bar for the different steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Analyze text context for better linking
                status_text.text("Analyzing text context...")
                progress_bar.progress(10)
                text_context = self.entity_linker.analyze_text_context(text)
                
                # Step 2: Extract entities using LLM (cached)
                status_text.text("Extracting entities using Gemini LLM...")
                progress_bar.progress(25)
                entities_json = self.cached_extract_entities(text)
                entities = json.loads(entities_json)
                
                if not entities:
                    st.warning("No entities found in the text.")
                    return
                
                # Step 3: Link to Wikidata (cached)
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Contextual Wikipedia linking
                status_text.text("Linking to Wikipedia with context...")
                progress_bar.progress(60)
                entities = self.entity_linker.link_to_wikipedia_contextual(entities, text_context)
                
                # Step 5: Link to Britannica (cached)
                status_text.text("Linking to Britannica...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 6: Get coordinates
                status_text.text("Getting coordinates...")
                progress_bar.progress(85)
                # Geocode all place entities more aggressively
                place_entities = [e for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']]
                
                if place_entities:
                    try:
                        # Use the get_coordinates method which handles multiple geocoding services
                        geocoded_entities = self.entity_linker.get_coordinates(place_entities, text)
                        
                        # Update the entities list with geocoded results
                        for geocoded_entity in geocoded_entities:
                            # Find the corresponding entity in the main list and update it
                            for idx, entity in enumerate(entities):
                                if (entity['text'] == geocoded_entity['text'] and 
                                    entity['type'] == geocoded_entity['type'] and
                                    entity['start'] == geocoded_entity['start']):
                                    entities[idx] = geocoded_entity
                                    break
                    except Exception as e:
                        st.warning(f"Some geocoding failed: {e}")
                        # Continue with processing even if geocoding fails
                
                # Step 7: Link addresses to OpenStreetMap
                status_text.text("Linking addresses to OpenStreetMap...")
                progress_bar.progress(90)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 8: Generate visualization
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                st.session_state.text_context = text_context  # Store context for reference
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show context analysis results
                if text_context['period'] or text_context['region'] or text_context['subject_matter']:
                    st.success(f"Processing complete! Found {len(entities)} entities.")
                    context_info = []
                    if text_context['period']:
                        context_info.append(f"Period: {text_context['period'].replace('_', ' ').title()}")
                    if text_context['region']:
                        context_info.append(f"Region: {text_context['region'].replace('_', ' ').title()}")
                    if text_context['subject_matter']:
                        context_info.append(f"Subject: {text_context['subject_matter'].title()}")
                    
                    st.info(f"Context detected: {' | '.join(context_info)}")
                else:
                    st.success(f"Processing complete! Found {len(entities)} entities.")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities for display.
        Fixed to ensure proper HTML link syntax.
        """
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Color scheme - expanded for all entity types (removed QUANTITY)
        colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA',         # F&B Oxford stone
            'PRODUCT': '#E6D7C9',         # F&B Skimming stone
            'EVENT': '#D4C5B9',          # F&B Mouse's back
            'WORK_OF_ART': '#E8E1D4',    # F&B Strong white
            'LANGUAGE': '#F0EAE2',       # F&B Pointing
            'LAW': '#DDD6CE',            # F&B Elephant's breath (lighter)
            'DATE': '#E3DDD7',          # F&B Dimity
            'MONEY': '#D6CFCA'          # F&B Joa's white
        }
        
        # Replace entities from end to start to avoid position shifting
        for entity in sorted_entities:
            # Only highlight entities that have links OR coordinates
            has_links = (entity.get('britannica_url') or 
                         entity.get('wikidata_url') or 
                         entity.get('wikipedia_url') or     
                         entity.get('openstreetmap_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_links or has_coordinates):
                continue
                
            start = entity['start']
            end = entity['end']
            
            # Get the original text of the entity from the source text
            original_entity_text = text[start:end]
            # Escape it for HTML
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#E7E2D2')
            
            # Create tooltip with entity information
            tooltip_parts = [f"Type: {entity['type']}"]
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = html_module.escape(" | ".join(tooltip_parts))
            
            # Create the replacement HTML with proper link syntax
            # Priority: Wikipedia > Wikidata > Britannica > OpenStreetMap > Coordinates only
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
            
            # Find and replace the entity text in the HTML
            # We need to find the escaped version of the entity text
            escaped_original = html_module.escape(original_entity_text)
            
            # Replace only the first occurrence to avoid issues with duplicate entities
            if escaped_original in highlighted:
                highlighted = highlighted.replace(escaped_original, replacement, 1)
        
        return highlighted

    def render_results(self):
        """Render the results section with entities and visualizations - same as NLTK app."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Entity details in collapsible section for mobile
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        # Export options in collapsible section for mobile
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details - same as NLTK app."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare data for table
        table_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Links': self.format_entity_links(entity)
            }
            
            if entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            
            if entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table - same as NLTK app."""
        links = []
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results - same as NLTK app."""
        # Stack buttons vertically for mobile
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export - create JSON-LD format
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "entities": []
            }
            
            # Format entities for JSON-LD
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end']
                }
                
                if entity.get('wikidata_url'):
                    entity_data['sameAs'] = entity['wikidata_url']
                
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                
                if entity.get('wikipedia_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['wikipedia_url']]
                        else:
                            entity_data['sameAs'].append(entity['wikipedia_url'])
                    else:
                        entity_data['sameAs'] = entity['wikipedia_url']
                
                if entity.get('britannica_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['britannica_url']]
                        else:
                            entity_data['sameAs'].append(entity['britannica_url'])
                    else:
                        entity_data['sameAs'] = entity['britannica_url']
                
                if entity.get('openstreetmap_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['openstreetmap_url']]
                        else:
                            entity_data['sameAs'].append(entity['openstreetmap_url'])
                    else:
                        entity_data['sameAs'] = entity['openstreetmap_url']
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.jsonld",
                mime="application/ld+json",
                use_container_width=True
            )
        
        with col2:
            # HTML export - clean version with just the text and proper links
            if st.session_state.html_content:
                html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Entity Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    {st.session_state.html_content}
</body>
</html>"""
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html",
                    use_container_width=True
                )

    def run(self):
        """Main application runner - same as NLTK app."""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Single column layout for mobile compatibility
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Process button
        if st.button("Process Text", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Add some spacing
        st.markdown("---")
        
        # Results section
        self.render_results()


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitLLMEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
