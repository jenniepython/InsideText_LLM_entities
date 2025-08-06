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

    def construct_ner_prompt(self, text: str, context: Dict[str, Any] = None):
        """Construct a context-aware NER prompt that works for any text."""
        
        # Analyze context if not provided
        if context is None:
            context = self.analyze_text_context(text)
        
        # Create dynamic context instructions based on detected patterns
        context_instructions = ""
        
        # Add period-based instructions
        if context.get('period') == 'ancient':
            context_instructions += """
ANCIENT/HISTORICAL CONTEXT DETECTED:
- Place names in historical contexts = ancient locations (GPE), not modern companies
- Ancient peoples (like Phoenicians, Romans, etc.) = civilizations (ORGANIZATION)
- If people "came to", "sailed to", "traveled to" a place → that place is GPE/LOCATION
- Names ending in -us, -os, -es often = ancient people or places
- Kings, emperors, pharaohs = historical figures (PERSON)
"""
        elif context.get('period') in ['medieval', 'victorian']:
            context_instructions += f"""
{context.get('period').upper()} PERIOD CONTEXT DETECTED:
- Historical building names = period facilities (FACILITY)
- Technical/architectural terms = historical products/features (PRODUCT)
- Period-appropriate interpretation of ambiguous names
"""
        
        # Add subject matter instructions
        if context.get('subject_matter') == 'theater':
            context_instructions += """
THEATER CONTEXT DETECTED:
- Stage terminology (mezzanine, proscenium, trap, etc.) = technical features (PRODUCT)
- Theater names = performance venues (FACILITY) 
- Play titles = works of art (WORK_OF_ART)
- Performers, directors = people (PERSON)
"""
        elif context.get('subject_matter') == 'architecture':
            context_instructions += """
ARCHITECTURE CONTEXT DETECTED:
- Building components, materials = architectural products (PRODUCT)
- Architectural styles, periods = historical context
- Building names = facilities (FACILITY)
"""
        elif context.get('subject_matter') == 'literature':
            context_instructions += """
LITERATURE CONTEXT DETECTED:
- Book titles, manuscript names = works of art (WORK_OF_ART)
- Authors, poets, writers = people (PERSON)
- Literary characters = people (PERSON) if clearly fictional
"""
        
        # Add geographical context
        if context.get('region'):
            region_name = context['region'].replace('_', ' ').title()
            context_instructions += f"""
GEOGRAPHICAL CONTEXT: {region_name}
- Interpret place names in appropriate regional/cultural context
- Historical places should link to period-appropriate entries
"""
        
        # Dynamic examples based on detected context
        examples = """
Examples:

Text: "The merchants sailed from their homeland to the great city, bringing goods to trade with the local rulers."
Context: Historical/ancient context detected
Output:
[
    {"text": "merchants", "type": "PERSON", "start_pos": 4},
    {"text": "homeland", "type": "LOCATION", "start_pos": 27},
    {"text": "great city", "type": "GPE", "start_pos": 44},
    {"text": "rulers", "type": "PERSON", "start_pos": 97}
]

Text: "The theater's main stage featured a revolving platform and hydraulic lifts for scene changes."
Context: Theater context detected
Output:
[
    {"text": "theater", "type": "FACILITY", "start_pos": 4},
    {"text": "main stage", "type": "PRODUCT", "start_pos": 14},
    {"text": "revolving platform", "type": "PRODUCT", "start_pos": 35},
    {"text": "hydraulic lifts", "type": "PRODUCT", "start_pos": 58}
]

Text: "The manuscript contains poems written by the court poet during the reign of King Edward."
Context: Literature/historical context detected
Output:
[
    {"text": "manuscript", "type": "WORK_OF_ART", "start_pos": 4},
    {"text": "poems", "type": "WORK_OF_ART", "start_pos": 24},
    {"text": "court poet", "type": "PERSON", "start_pos": 45},
    {"text": "King Edward", "type": "PERSON", "start_pos": 75}
]
"""
        
        prompt = f"""You are an expert named entity recognition system that adapts to different text types and contexts. Your task is to identify and extract ALL relevant entities while interpreting them correctly based on context.

{context_instructions}

CORE CONTEXTUAL RULES (apply to ANY text):
1. If people "went to", "came to", "sailed to", "traveled to" a place → that place is GPE/LOCATION
2. In historical contexts, interpret names as historical entities, not modern companies
3. Technical terms related to the subject matter = PRODUCT (tools, components, features)
4. Titles of creative works = WORK_OF_ART
5. Names of buildings, venues, institutions = FACILITY
6. Groups of people, civilizations, organizations = ORGANIZATION
7. Individual people, historical figures, characters = PERSON

ENTITY TYPES:
- PERSON: People, historical figures, characters, roles, titles
- ORGANIZATION: Groups, institutions, civilizations, companies, teams
- GPE: Cities, countries, regions, kingdoms, territories, political entities  
- LOCATION: Geographic places, landmarks, natural features
- FACILITY: Buildings, venues, structures, stages, theaters
- ADDRESS: Street addresses, property descriptions
- PRODUCT: Objects, tools, components, materials, technical features
- EVENT: Named events, ceremonies, performances, battles
- WORK_OF_ART: Books, plays, poems, manuscripts, artworks
- LANGUAGE: Languages, dialects, scripts
- LAW: Legal documents, laws, regulations
- DATE: Dates, periods, years, eras, reigns
- MONEY: Currencies, amounts, prices

{examples}

Now extract entities from this text, using context clues to determine correct interpretations:

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
        """Extract named entities from text using Gemini LLM with context awareness."""
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found!")
                return []
            
            # First, analyze the text context
            context = self.analyze_text_context(text)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Use context-aware prompt
            prompt = self.construct_ner_prompt(text, context)
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
                        'end': start_pos + len(entity_text),
                        'context': context  # Store context for linking
                    }
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            st.error(f"Error in LLM entity extraction: {e}")
            return []

    def analyze_text_context(self, text: str) -> Dict[str, Any]:
        """Analyze text to determine historical, geographical, and cultural context - works for any text."""
        context = {
            'period': None,
            'region': None,
            'culture': None,
            'subject_matter': None,
            'language_style': 'modern',
            'time_indicators': [],
            'place_indicators': [],
            'subject_indicators': []
        }
        
        text_lower = text.lower()
        
        # Detect time periods - broader patterns
        if any(indicator in text_lower for indicator in ['bc', 'bce', 'ancient', 'antiquity', 'classical', 'pharaoh', 'emperor', 'temple', 'oracle', 'sailed', 'came to', 'kingdom']):
            context['period'] = 'ancient'
            context['time_indicators'].extend(['ancient', 'classical', 'historical'])
        elif any(indicator in text_lower for indicator in ['medieval', 'middle ages', 'feudal', 'knight', 'monastery', 'manuscript', 'abbey', 'monk']):
            context['period'] = 'medieval'
            context['time_indicators'].extend(['medieval', 'historical'])
        elif any(indicator in text_lower for indicator in ['victorian', '19th century', 'industrial', 'railway', 'steam', 'empire', 'theatre royal']):
            context['period'] = 'victorian'
            context['time_indicators'].extend(['victorian', '19th century'])
        elif any(indicator in text_lower for indicator in ['20th century', '21st century', 'modern', 'contemporary', 'digital', 'internet']):
            context['period'] = 'modern'
        
        # Detect geographical context - broader patterns
        # European context
        if any(indicator in text_lower for indicator in ['europe', 'european', 'britain', 'france', 'germany', 'italy', 'spain']):
            context['region'] = 'european'
        # Ancient Mediterranean
        elif any(indicator in text_lower for indicator in ['mediterranean', 'greece', 'greek', 'rome', 'roman', 'egypt', 'phoenician']):
            context['region'] = 'mediterranean'
            if any(indicator in text_lower for indicator in ['greece', 'greek', 'hellas', 'athens', 'sparta']):
                context['culture'] = 'greek'
            elif any(indicator in text_lower for indicator in ['rome', 'roman', 'latin', 'caesar']):
                context['culture'] = 'roman'
        # Asian context
        elif any(indicator in text_lower for indicator in ['asia', 'china', 'japan', 'india', 'persia', 'persian']):
            context['region'] = 'asian'
        
        # Detect subject matter - broader patterns
        if any(indicator in text_lower for indicator in ['theatre', 'theater', 'stage', 'play', 'drama', 'performance', 'actor', 'audience', 'curtain']):
            context['subject_matter'] = 'theater'
            context['subject_indicators'].extend(['theater', 'performance', 'drama'])
        elif any(indicator in text_lower for indicator in ['building', 'architecture', 'cathedral', 'church', 'castle', 'palace', 'construction']):
            context['subject_matter'] = 'architecture'
            context['subject_indicators'].extend(['architecture', 'building'])
        elif any(indicator in text_lower for indicator in ['book', 'manuscript', 'text', 'author', 'writing', 'literature', 'poem', 'novel']):
            context['subject_matter'] = 'literature'
            context['subject_indicators'].extend(['literature', 'writing'])
        elif any(indicator in text_lower for indicator in ['battle', 'war', 'army', 'soldier', 'military', 'conflict', 'victory']):
            context['subject_matter'] = 'military'
            context['subject_indicators'].extend(['military', 'war', 'battle'])
        elif any(indicator in text_lower for indicator in ['king', 'queen', 'royal', 'court', 'politics', 'government', 'ruler']):
            context['subject_matter'] = 'politics'
            context['subject_indicators'].extend(['politics', 'royal', 'government'])
        
        # Detect language style patterns
        if any(pattern in text_lower for pattern in ['thee', 'thou', 'thy', 'hath', 'doth', 'forsooth', 'wherefore']):
            context['language_style'] = 'archaic'
        elif any(pattern in text_lower for pattern in ['it is said', 'according to', 'the learned men', 'historians say']):
            context['language_style'] = 'historical_narrative'
        elif any(pattern in text_lower for pattern in ['recording', 'survey', 'analysis', 'study', 'research']):
            context['language_style'] = 'academic'
        
        return context

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
        """Add contextual Wikipedia linking that works for any text type."""
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Get entity context from extraction
                entity_context = entity.get('context', text_context)
                
                # Create context-aware search terms - generalized approach
                search_terms = [entity['text']]
                
                # Add context based on detected patterns, not hardcoded regions
                context_modifiers = []
                
                # Add period context
                if entity_context.get('time_indicators'):
                    context_modifiers.extend(entity_context['time_indicators'])
                
                # Add subject matter context
                if entity_context.get('subject_indicators'):
                    context_modifiers.extend(entity_context['subject_indicators'])
                
                # Add place context
                if entity_context.get('place_indicators'):
                    context_modifiers.extend(entity_context['place_indicators'])
                
                # Add general historical context if period is detected
                if entity_context.get('period') and entity_context['period'] != 'modern':
                    context_modifiers.extend(['historical', entity_context['period']])
                
                # Create contextual search terms for different entity types
                if entity['type'] in ['GPE', 'LOCATION'] and context_modifiers:
                    for modifier in context_modifiers[:3]:  # Top 3 modifiers
                        search_terms.append(f"{entity['text']} {modifier}")
                        if modifier != 'historical':  # Avoid double historical
                            search_terms.append(f"{modifier} {entity['text']}")
                
                elif entity['type'] == 'PERSON' and context_modifiers:
                    for modifier in context_modifiers[:3]:
                        search_terms.append(f"{entity['text']} {modifier}")
                        # Add biographical terms
                        if entity_context.get('period') and entity_context['period'] != 'modern':
                            search_terms.append(f"{entity['text']} biography")
                
                elif entity['type'] == 'ORGANIZATION' and context_modifiers:
                    for modifier in context_modifiers[:2]:
                        search_terms.append(f"{entity['text']} {modifier}")
                
                elif entity['type'] in ['FACILITY', 'PRODUCT'] and 'theater' in context_modifiers:
                    search_terms.extend([
                        f"{entity['text']} theatre",
                        f"{entity['text']} theater architecture"
                    ])
                
                # Remove duplicates and limit search terms
                search_terms = list(dict.fromkeys(search_terms))[:4]
                
                # Try each contextual search term
                for search_term in search_terms:
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
                            
                            # Basic validation - check if result seems relevant
                            snippet = result.get('snippet', '').lower()
                            title_lower = page_title.lower()
                            
                            # Skip clearly irrelevant results
                            skip_terms = ['video game', 'software', 'app', 'company', 'corporation', 'brand']
                            if entity_context.get('period') and entity_context['period'] != 'modern':
                                if any(term in snippet or term in title_lower for term in skip_terms):
                                    continue  # Try next search term
                            
                            # Accept result if it seems reasonable
                            encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                            entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                            entity['wikipedia_title'] = page_title
                            
                            # Get a snippet/description from the search result
                            if result.get('snippet'):
                                snippet_clean = re.sub(r'<[^>]+>', '', result['snippet'])
                                entity['wikipedia_description'] = snippet_clean[:200] + "..." if len(snippet_clean) > 200 else snippet_clean
                            
                            # Mark which search term worked
                            entity['search_context'] = search_term
                            break
                    
                    time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                pass
        
        return entities

    def link_to_wikipedia(self, entities):
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
