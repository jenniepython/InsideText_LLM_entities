#!/usr/bin/env python3
"""
Streamlit LLM Entity Linker Application - Enhanced Version with Overlap Resolution

Enhanced with:
1. Geocoding only for place entities (GPE, LOCATION, FACILITY, ADDRESS)
2. Contextual linking using surrounding words for better knowledge base disambiguation
3. Fixed overlapping entity extraction (no more duplicate Amazon, etc.)

Author: Enhanced from NER_LLM_entities_streamlit.py
Version: 2.2
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

try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
    st.warning("pycountry not installed. Using fallback country detection. Install with: pip install pycountry")

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
    Main class for LLM-based entity linking functionality with contextual linking and overlap resolution.
    
    Enhanced with:
    - Geocoding only for place entities
    - Contextual linking using surrounding words for better disambiguation
    - Fixed overlapping entity extraction
    """
    
    def __init__(self):
        """Initialize the LLM Entity Linker."""
        # Color scheme for different entity types in HTML output
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

        # Define which entity types should be geocoded (PLACES ONLY)
        self.geocodable_types = {'GPE', 'LOCATION', 'FACILITY', 'ADDRESS'}

    def extract_context_window(self, text: str, entity_start: int, entity_end: int, window_size: int = 50) -> Dict[str, str]:
        """
        Extract context window around an entity for better disambiguation.
        
        Args:
            text: Full text
            entity_start: Start position of entity
            entity_end: End position of entity  
            window_size: Number of characters to extract on each side
            
        Returns:
            Dict with context information
        """
        # Extract surrounding context
        context_start = max(0, entity_start - window_size)
        context_end = min(len(text), entity_end + window_size)
        
        # Get words before and after the entity
        before_text = text[context_start:entity_start].strip()
        after_text = text[entity_end:context_end].strip()
        entity_text = text[entity_start:entity_end]
        
        # Extract meaningful words (filter out common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'}
        
        # Extract contextual keywords
        before_words = [w.strip('.,;:!?"()[]{}') for w in before_text.split() if w.lower().strip('.,;:!?"()[]{}') not in stop_words and len(w) > 2]
        after_words = [w.strip('.,;:!?"()[]{}') for w in after_text.split() if w.lower().strip('.,;:!?"()[]{}') not in stop_words and len(w) > 2]
        
        # Get the most recent meaningful words (up to 3 before, 3 after)
        context_before = ' '.join(before_words[-3:]) if before_words else ''
        context_after = ' '.join(after_words[:3]) if after_words else ''
        
        return {
            'before': context_before,
            'after': context_after,
            'full_before': before_text,
            'full_after': after_text,
            'entity': entity_text,
            'context_snippet': f"{before_text} [{entity_text}] {after_text}".strip()
        }

    def construct_ner_prompt(self, text: str, context: Dict[str, Any] = None):
        """Construct an improved NER prompt that avoids substring overlaps."""
        
        prompt = f"""Extract named entities from the following text. Only extract proper nouns and named things.

ENTITY TYPES:
- PERSON: Named individuals (e.g., "John Smith", "Caesar")
- ORGANIZATION: Named groups, companies, civilizations (e.g., "Apple Inc", "the Phoenicians") 
- GPE: Countries, cities, regions (e.g., "France", "London", "ancient Egypt")
- LOCATION: Geographic features (e.g., "Red Sea", "Mount Everest", "Amazon rainforest")
- FACILITY: Named buildings, venues (e.g., "Empire State Building")
- PRODUCT: Named objects, brands (e.g., "iPhone", "merchandise" only if specifically named)
- EVENT: Named events (e.g., "World War II", "the Olympics")
- WORK_OF_ART: Titles of books, movies, songs, etc.
- DATE: Specific dates, years, periods
- MONEY: Specific amounts with currency

CRITICAL RULES:
1. Only extract proper nouns - things with specific names
2. Don't extract adjectives or descriptive words (e.g., skip "Scottish" in "Scottish goods")
3. Don't extract common job titles or roles unless they're part of a proper name
4. Don't extract generic terms like "king", "merchants", "women" unless they're part of a proper name
5. IMPORTANT: Only list each unique entity ONCE in your response, even if it appears multiple times in the text
6. CRITICAL: When you find a compound entity like "Edinburgh castle", do NOT also extract just "Edinburgh" separately
7. CRITICAL: Choose the most complete and specific form of each entity (e.g., prefer "Edinburgh castle" over "Edinburgh" when referring to the location)
8. CRITICAL: If the same word appears in different contexts (like "Amazon rainforest" vs "Amazon company"), extract both as separate entities
9. CONTEXT-AWARE CLASSIFICATION: The same word can have different entity types based on surrounding context. Always check the immediate context before assigning entity type.
10. GEOGRAPHICAL INDICATORS - MANDATORY OVERRIDE: When a word appears immediately after "the country of", "nation of", "state of", "republic of", it MUST be classified as GPE regardless of any previous occurrences. This rule overrides all other classifications.
11. DISAMBIGUATION PATTERNS: Look for context clues:
    - "President/King/Queen of X" → X is GPE
    - "X River/Mountain/Sea/Castle" → LOCATION  
    - "country/nation/state of X" → X is GPE
    - "person/student/friend named X" → X is PERSON
12. MULTI-OCCURRENCE ANALYSIS: When the same word appears multiple times, analyze each occurrence independently based on its specific context, not previous classifications.

EXAMPLES:

Input: "The Roman merchants sailed to Egypt with their cargo."
Output: [
  {{"text": "Roman", "type": "ORGANIZATION", "start_pos": 4}},
  {{"text": "Egypt", "type": "GPE", "start_pos": 29}}
]

Input: "At dinner, a guest spoke about the Amazon rainforest, but her friend assumed she was talking about Amazon, the company."
Output: [
  {{"text": "Amazon rainforest", "type": "LOCATION", "start_pos": 39}},
  {{"text": "Amazon", "type": "ORGANIZATION", "start_pos": 98}}
]

Input: "Jordan River flows through Jordan, and my friend Jordan visited there."
Output: [
  {{"text": "Jordan River", "type": "LOCATION", "start_pos": 0}},
  {{"text": "Jordan", "type": "GPE", "start_pos": 21}},
  {{"text": "Jordan", "type": "PERSON", "start_pos": 46}}
]

Now extract entities from this text:

"{text}"

Output only a JSON array with the entities found:"""
        
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

    def _remove_overlapping_entities(self, entities):
        """
        Remove overlapping entities, keeping the longer/more specific ones.
        
        For example, if we have both "Amazon rainforest" and "Amazon", 
        we keep "Amazon rainforest" and remove the "Amazon" that overlaps with it.
        """
        if not entities:
            return entities
            
        # Sort entities by start position and length (longer first for same position)
        sorted_entities = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        non_overlapping = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for i, accepted in enumerate(non_overlapping):
                # Check for overlap: entities overlap if one starts before the other ends
                if not (entity['end'] <= accepted['start'] or entity['start'] >= accepted['end']):
                    overlaps = True
                    
                    # If the current entity is longer and completely contains the accepted one, replace it
                    current_length = entity['end'] - entity['start']
                    accepted_length = accepted['end'] - accepted['start']
                    
                    if current_length > accepted_length:
                        # Remove the shorter entity and add the longer one
                        non_overlapping.pop(i)
                        non_overlapping.append(entity)
                        overlaps = False  # We've handled this overlap
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        # Sort back by original position
        non_overlapping.sort(key=lambda x: x['start'])
        
        # Debug info
        if len(entities) != len(non_overlapping):
            removed_count = len(entities) - len(non_overlapping)
            st.info(f"Removed {removed_count} overlapping entities for cleaner results")
        
        return non_overlapping

    def extract_entities(self, text: str):
        """Extract named entities from text using Gemini LLM with context extraction and overlap resolution."""
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found!")
                return []
            
            # First, analyse the text context
            context = self.analyse_text_context(text)
            
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
            
            # Debug: Show what LLM found
            st.info(f"LLM found {len(entities_raw)} raw entities: {[e.get('text', 'N/A') for e in entities_raw]}")
            
            # Deduplicate entities from LLM response FIRST
            seen_entities = set()
            deduplicated_entities_raw = []
            
            for entity in entities_raw:
                if 'text' in entity and 'type' in entity:
                    entity_key = (entity['text'].strip().lower(), entity['type'])
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        deduplicated_entities_raw.append(entity)
            
            entities_raw = deduplicated_entities_raw
            
            # Convert to consistent format and find ALL occurrences with context
            entities = []
            
            for entity_raw in entities_raw:
                if 'text' in entity_raw and 'type' in entity_raw:
                    entity_text = entity_raw['text'].strip()
                    entity_type = entity_raw['type']
                    
                    # Use regex with word boundaries to find exact matches
                    import re
                    pattern = r'\b' + re.escape(entity_text) + r'\b'
                    
                    # Find all non-overlapping matches
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    
                    if matches:
                        # Create one entity for each actual occurrence with context
                        for match in matches:
                            # Extract context window for this specific occurrence
                            context_window = self.extract_context_window(
                                text, match.start(), match.end()
                            )
                            
                            entity = {
                                'text': match.group(),  # Preserves original case
                                'type': entity_type,
                                'start': match.start(),
                                'end': match.end(),
                                'context': context,
                                'context_window': context_window  # Add contextual information
                            }
                            entities.append(entity)
                    else:
                        # Fallback: try case-sensitive exact search
                        start_pos = 0
                        while True:
                            pos = text.find(entity_text, start_pos)
                            if pos == -1:
                                break
                            
                            # Extract context for this occurrence
                            context_window = self.extract_context_window(
                                text, pos, pos + len(entity_text)
                            )
                            
                            entity = {
                                'text': entity_text,
                                'type': entity_type,
                                'start': pos,
                                'end': pos + len(entity_text),
                                'context': context,
                                'context_window': context_window
                            }
                            entities.append(entity)
                            
                            # Move past the entire entity
                            start_pos = pos + len(entity_text)
                        
                        # If no matches found at all, use LLM position as last resort
                        if not any(e['text'].lower() == entity_text.lower() for e in entities):
                            start_pos = entity_raw.get('start_pos', 0)
                            if 0 <= start_pos < len(text):
                                context_window = self.extract_context_window(
                                    text, start_pos, start_pos + len(entity_text)
                                )
                                
                                entity = {
                                    'text': entity_text,
                                    'type': entity_type,
                                    'start': start_pos,
                                    'end': start_pos + len(entity_text),
                                    'context': context,
                                    'context_window': context_window
                                }
                                entities.append(entity)
            
            # CRITICAL: Remove overlapping entities to fix the duplication issue
            entities = self._remove_overlapping_entities(entities)
            
            return entities
            
        except Exception as e:
            st.error(f"Error in LLM entity extraction: {e}")
            return []

    def analyse_text_context(self, text: str) -> Dict[str, Any]:
        """Analyse text to determine historical, geographical, and cultural context - works for any text."""
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

    def _get_all_countries(self) -> List[str]:
        """
        Get comprehensive list of all countries using pycountry library.
        Falls back to minimal list if pycountry not available.
        """
        if not PYCOUNTRY_AVAILABLE:
            # Minimal fallback list (still biased but smaller impact)
            return ['usa', 'united states', 'uk', 'united kingdom', 'france', 'germany', 'china', 'japan', 'india', 'australia', 'canada', 'brazil', 'russia']
        
        countries = []
        
        # Get all countries from pycountry
        for country in pycountry.countries:
            # Add official name
            countries.append(country.name.lower())
            
            # Add common alternative names and codes
            if hasattr(country, 'common_name') and country.common_name:
                countries.append(country.common_name.lower())
            
            # Add 2-letter country code (ISO 3166-1 alpha-2)
            countries.append(country.alpha_2.lower())
            
            # Add 3-letter country code (ISO 3166-1 alpha-3)  
            countries.append(country.alpha_3.lower())
            
            # Add some common alternative names manually for major countries
            name_variations = {
                'united states': ['usa', 'america', 'us'],
                'united kingdom': ['uk', 'britain', 'great britain', 'england', 'scotland', 'wales', 'northern ireland'],
                'russia': ['russian federation'],
                'south korea': ['republic of korea', 'korea south'],
                'north korea': ['democratic people\'s republic of korea', 'korea north'],
                'czech republic': ['czechia'],
                'myanmar': ['burma'],
                'ivory coast': ['cÃƒÂ´te d\'ivoire'],
                'democratic republic of the congo': ['drc', 'congo kinshasa'],
                'republic of the congo': ['congo brazzaville'],
                'united arab emirates': ['uae'],
                'saudi arabia': ['kingdom of saudi arabia'],
                'vatican city': ['holy see'],
                'bosnia and herzegovina': ['bosnia'],
                'north macedonia': ['macedonia', 'former yugoslav republic of macedonia'],
                'timor-leste': ['east timor'],
                'eswatini': ['swaziland'],
                'cabo verde': ['cape verde']
            }
            
            country_name_lower = country.name.lower()
            if country_name_lower in name_variations:
                countries.extend(name_variations[country_name_lower])
        
        # Remove duplicates and return
        return list(set(countries))

    def get_coordinates(self, entities, processed_text=""):
        """Enhanced coordinate lookup ONLY for place entities (GPE, LOCATION, FACILITY, ADDRESS)."""
        # Filter to only geocodable entity types
        place_entities = [e for e in entities if e['type'] in self.geocodable_types]
        
        if not place_entities:
            st.info("No place entities found for geocoding.")
            return entities
        
        # Detect geographical context from the full text
        context_clues = self._detect_geographical_context(processed_text, place_entities)
        
        if context_clues:
            print(f"Detected geographical context: {', '.join(context_clues)}")
        
        geocoded_count = 0
        for entity in place_entities:
            # Skip if already has coordinates
            if entity.get('latitude') is not None:
                continue
            
            # Try geocoding with context
            if self._try_contextual_geocoding(entity, context_clues):
                geocoded_count += 1
                continue
                
            # Fall back to OpenStreetMap
            if self._try_openstreetmap(entity):
                geocoded_count += 1
                continue
        
        st.info(f"Geocoded {geocoded_count}/{len(place_entities)} place entities")
        return entities

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Detect geographical context from the text using pycountry for comprehensive country detection.
        This provides unbiased global coverage instead of hardcoded northern hemisphere bias.
        """
        context_clues = []
        text_lower = text.lower()
        
        # Extract from entities that are already identified as places
        geographical_entities = []
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION', 'FACILITY']:
                geographical_entities.append(entity['text'].lower())
        
        # Get comprehensive country list using pycountry
        all_countries = self._get_all_countries()
        
        # Look for any countries mentioned in the text
        for country in all_countries:
            if country in text_lower:
                context_clues.append(country)
                # Limit to avoid too many context clues
                if len(context_clues) >= 10:
                    break
        
        # Add geographical entities found by the LLM
        for geo_entity in geographical_entities:
            if geo_entity not in context_clues:
                context_clues.append(geo_entity)
        
        # Look for postal codes to infer country (expanded coverage)
        postal_patterns = {
            'uk': [r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b'],  # UK postcodes
            'usa': [r'\b\d{5}(-\d{4})?\b'],  # US ZIP codes
            'canada': [r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b'],  # Canadian postal codes
            'germany': [r'\b\d{5}\b'],  # German postal codes
            'france': [r'\b\d{5}\b'],  # French postal codes  
            'australia': [r'\b\d{4}\b'],  # Australian postal codes
            'netherlands': [r'\b\d{4}\s*[A-Z]{2}\b'],  # Dutch postal codes
            'sweden': [r'\b\d{3}\s*\d{2}\b'],  # Swedish postal codes
            'norway': [r'\b\d{4}\b'],  # Norwegian postal codes
            'denmark': [r'\b\d{4}\b'],  # Danish postal codes
            'switzerland': [r'\b\d{4}\b'],  # Swiss postal codes
            'austria': [r'\b\d{4}\b'],  # Austrian postal codes
            'belgium': [r'\b\d{4}\b'],  # Belgian postal codes
            'spain': [r'\b\d{5}\b'],  # Spanish postal codes
            'italy': [r'\b\d{5}\b'],  # Italian postal codes
            'portugal': [r'\b\d{4}-\d{3}\b'],  # Portuguese postal codes
            'japan': [r'\b\d{3}-\d{4}\b'],  # Japanese postal codes
            'south korea': [r'\b\d{5}\b'],  # South Korean postal codes
            'singapore': [r'\b\d{6}\b'],  # Singapore postal codes
            'brazil': [r'\b\d{5}-\d{3}\b'],  # Brazilian postal codes (CEP)
            'mexico': [r'\b\d{5}\b'],  # Mexican postal codes
            'india': [r'\b\d{6}\b'],  # Indian PIN codes
            'china': [r'\b\d{6}\b'],  # Chinese postal codes
            'south africa': [r'\b\d{4}\b'],  # South African postal codes
            'new zealand': [r'\b\d{4}\b'],  # New Zealand postal codes
            'russia': [r'\b\d{6}\b'],  # Russian postal codes
            'poland': [r'\b\d{2}-\d{3}\b'],  # Polish postal codes
            'czech republic': [r'\b\d{3}\s*\d{2}\b'],  # Czech postal codes
            'finland': [r'\b\d{5}\b'],  # Finnish postal codes
            'israel': [r'\b\d{7}\b'],  # Israeli postal codes
            'turkey': [r'\b\d{5}\b'],  # Turkish postal codes
            'argentina': [r'\b[A-Z]\d{4}[A-Z]{3}\b'],  # Argentine postal codes
            'chile': [r'\b\d{7}\b'],  # Chilean postal codes
            'colombia': [r'\b\d{6}\b'],  # Colombian postal codes
            'peru': [r'\b\d{5}\b'],  # Peruvian postal codes
            'venezuela': [r'\b\d{4}\b'],  # Venezuelan postal codes
            'ukraine': [r'\b\d{5}\b'],  # Ukrainian postal codes
            'romania': [r'\b\d{6}\b'],  # Romanian postal codes
            'greece': [r'\b\d{5}\b'],  # Greek postal codes
            'hungary': [r'\b\d{4}\b'],  # Hungarian postal codes
            'bulgaria': [r'\b\d{4}\b'],  # Bulgarian postal codes
            'croatia': [r'\b\d{5}\b'],  # Croatian postal codes
            'serbia': [r'\b\d{5}\b'],  # Serbian postal codes
            'thailand': [r'\b\d{5}\b'],  # Thai postal codes
            'vietnam': [r'\b\d{6}\b'],  # Vietnamese postal codes
            'malaysia': [r'\b\d{5}\b'],  # Malaysian postal codes
            'indonesia': [r'\b\d{5}\b'],  # Indonesian postal codes
            'philippines': [r'\b\d{4}\b'],  # Philippine postal codes
            'egypt': [r'\b\d{5}\b'],  # Egyptian postal codes
            'morocco': [r'\b\d{5}\b'],  # Moroccan postal codes
            'kenya': [r'\b\d{5}\b'],  # Kenyan postal codes
            'nigeria': [r'\b\d{6}\b'],  # Nigerian postal codes
            'ghana': [r'\b[A-Z]{2}-\d{3}-\d{4}\b'],  # Ghanaian postal codes
            'ethiopia': [r'\b\d{4}\b'],  # Ethiopian postal codes
            'tanzania': [r'\b\d{5}\b'],  # Tanzanian postal codes
            'uganda': [r'\b\d{5}\b'],  # Ugandan postal codes
        }
        
        for country, patterns in postal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if country not in context_clues:
                        context_clues.append(country)
                    break
        
        # Return the most relevant context clues (limit to avoid over-constraining)
        return context_clues[:5]  # Increased slightly due to better coverage

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context using contextual information and comprehensive country mapping."""
        if not context_clues:
            return False
        
        # Create context-aware search terms using both global and local context
        search_variations = [entity['text']]
        
        # Add context from surrounding words
        context_window = entity.get('context_window', {})
        if context_window.get('before') or context_window.get('after'):
            contextual_keywords = []
            if context_window.get('before'):
                contextual_keywords.extend(context_window['before'].split()[-2:])  # Last 2 words
            if context_window.get('after'):
                contextual_keywords.extend(context_window['after'].split()[:2])   # First 2 words
            
            # Create searches with local context
            for keyword in contextual_keywords:
                if len(keyword) > 2 and keyword.lower() not in ['the', 'and', 'with', 'from', 'near']:
                    search_variations.append(f"{entity['text']} {keyword}")
                    search_variations.append(f"{keyword} {entity['text']}")
        
        # Add global context to search terms - improved mapping
        for context in context_clues:
            context_lower = context.lower().strip()
            
            # Create smarter context mappings using pycountry if available
            if PYCOUNTRY_AVAILABLE:
                # Try to find the country by various identifiers
                country_variants = []
                try:
                    # Try by name
                    country = pycountry.countries.get(name=context_lower.title())
                    if country:
                        country_variants.extend([country.name, country.alpha_2, country.alpha_3])
                        if hasattr(country, 'common_name') and country.common_name:
                            country_variants.append(country.common_name)
                except:
                    pass
                
                try:
                    # Try by alpha_2 code
                    country = pycountry.countries.get(alpha_2=context_lower.upper())
                    if country:
                        country_variants.extend([country.name, country.alpha_2, country.alpha_3])
                        if hasattr(country, 'common_name') and country.common_name:
                            country_variants.append(country.common_name)
                except:
                    pass
                
                try:
                    # Try by alpha_3 code
                    country = pycountry.countries.get(alpha_3=context_lower.upper())
                    if country:
                        country_variants.extend([country.name, country.alpha_2, country.alpha_3])
                        if hasattr(country, 'common_name') and country.common_name:
                            country_variants.append(country.common_name)
                except:
                    pass
                
                # Use the variants we found
                for variant in list(set(country_variants))[:3]:  # Limit to avoid too many searches
                    search_variations.append(f"{entity['text']}, {variant}")
            else:
                # Fallback mapping for common countries if pycountry not available
                fallback_mapping = {
                    'uk': ['UK', 'United Kingdom', 'England', 'Britain'],
                    'usa': ['USA', 'United States', 'US'],
                    'us': ['USA', 'United States', 'US'],
                    'canada': ['Canada'],
                    'australia': ['Australia'],
                    'france': ['France'],
                    'germany': ['Germany'],
                    'china': ['China'],
                    'japan': ['Japan'],
                    'india': ['India'],
                    'brazil': ['Brazil'],
                    'russia': ['Russia'],
                    'mexico': ['Mexico'],
                    'south africa': ['South Africa'],
                    'new zealand': ['New Zealand'],
                }
                
                context_variants = fallback_mapping.get(context_lower, [context])
                for variant in context_variants[:2]:  # Limit variants
                    search_variations.append(f"{entity['text']}, {variant}")
        
        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))
        
        # Try OpenStreetMap with context (limit to top 5 to avoid rate limits)
        for search_term in search_variations[:5]:
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

    def link_to_wikidata(self, entities):
        """Add context-aware Wikidata linking using surrounding words for disambiguation."""
        for entity in entities:
            try:
                # Get context from entity if available
                entity_context = entity.get('context', {})
                context_window = entity.get('context_window', {})
                
                # Prepare search query with enhanced context
                search_query = entity['text']
                
                # Build contextual search query using surrounding words
                contextual_terms = []
                
                # Add meaningful words from context window
                if context_window.get('before'):
                    contextual_terms.extend([w for w in context_window['before'].split() if len(w) > 2])
                if context_window.get('after'):
                    contextual_terms.extend([w for w in context_window['after'].split() if len(w) > 2])
                
                # Add global context terms
                if entity_context.get('period') == 'ancient':
                    contextual_terms.append('ancient')
                if entity_context.get('region') == 'mediterranean':
                    contextual_terms.append('Greek')
                if entity_context.get('subject_matter'):
                    contextual_terms.append(entity_context['subject_matter'])
                
                # Create enhanced search queries
                search_queries = [search_query]  # Start with basic query
                
                # Add contextual variants
                if contextual_terms:
                    # Use most relevant contextual terms (limit to avoid overly specific queries)
                    for term in contextual_terms[:3]:
                        search_queries.append(f"{search_query} {term}")
                
                # Add entity-type specific context
                if entity['type'] == 'GPE':
                    # For places, add geographical context
                    if entity_context.get('period') == 'ancient':
                        search_queries.append(f"{entity['text']} ancient city")
                    elif entity_context.get('region') == 'mediterranean':
                        search_queries.append(f"{entity['text']} Greece")
                    
                    # Special cases for known ancient places
                    if entity['text'].lower() == 'argos':
                        search_queries.insert(1, "Argos Greece ancient city")
                    elif entity['text'].lower() == 'hellas':
                        search_queries.insert(1, "ancient Greece Hellas")
                
                elif entity['type'] == 'PERSON':
                    # For people, use contextual clues from surrounding words
                    if 'mythology' in ' '.join(contextual_terms).lower():
                        search_queries.append(f"{entity['text']} mythology")
                    elif entity_context.get('period') == 'ancient':
                        if entity['text'].lower() == 'io':
                            search_queries.insert(1, "Io mythology")
                        elif entity['text'].lower() == 'inachus':
                            search_queries.insert(1, "Inachus mythology river god")
                        else:
                            search_queries.append(f"{entity['text']} ancient history")
                
                elif entity['type'] == 'LOCATION':
                    # For locations like "Red Sea", use contextual information
                    if 'sea' in entity['text'].lower():
                        search_queries.append(f"{entity['text']} body of water")
                
                # Try each search query until we find a good match
                for search_query in search_queries:
                    url = "https://www.wikidata.org/w/api.php"
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'search': search_query,
                        'language': 'en',
                        'limit': 5,
                        'type': 'item'
                    }
                    
                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('search') and len(data['search']) > 0:
                            # Try to find the best match based on context
                            best_match = self._find_best_wikidata_match(data['search'], entity, entity_context)
                            
                            if best_match:
                                entity['wikidata_url'] = f"http://www.wikidata.org/entity/{best_match['id']}"
                                entity['wikidata_description'] = best_match.get('description', '')
                                entity['search_query_used'] = search_query  # Track which query worked
                                break  # Found a good match, stop searching
                    
                    time.sleep(0.1)  # Rate limiting between queries
                
                time.sleep(0.1)  # Rate limiting between entities
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def _find_best_wikidata_match(self, search_results: List[Dict], entity: Dict, entity_context: Dict) -> Dict:
        """
        Find the best Wikidata match using contextual information.
        
        Args:
            search_results: List of Wikidata search results
            entity: The entity being linked
            entity_context: Global context information
            
        Returns:
            Best matching result or None
        """
        context_window = entity.get('context_window', {})
        contextual_words = []
        
        # Collect contextual words for scoring
        if context_window.get('before'):
            contextual_words.extend(context_window['before'].lower().split())
        if context_window.get('after'):
            contextual_words.extend(context_window['after'].lower().split())
        
        best_match = None
        best_score = 0
        
        for result in search_results:
            description = result.get('description', '').lower()
            label = result.get('label', '').lower()
            score = 0
            
            # Base score for exact label match
            if label == entity['text'].lower():
                score += 10
            
            # Score based on entity type appropriateness
            if entity['type'] == 'GPE':
                if any(term in description for term in ['city', 'town', 'ancient', 'greece', 'greek', 'historical', 'archaeological', 'country', 'region']):
                    score += 5
                # Penalty for clearly wrong types
                if any(skip in description for skip in ['video game', 'game', 'software', 'album', 'film', 'movie', 'book']):
                    score -= 10
            
            elif entity['type'] == 'PERSON':
                if entity_context.get('period') == 'ancient':
                    if any(term in description for term in ['mythology', 'mythological', 'ancient', 'greek', 'deity', 'god', 'goddess', 'hero', 'king', 'queen']):
                        score += 5
                # Penalty for non-persons
                elif any(skip in description for skip in ['genus', 'species', 'asteroid', 'crater', 'company']):
                    score -= 5
            
            elif entity['type'] == 'LOCATION':
                if any(term in description for term in ['sea', 'ocean', 'river', 'mountain', 'lake', 'water', 'geographic']):
                    score += 5
            
            elif entity['type'] == 'ORGANIZATION':
                if any(term in description for term in ['people', 'ethnic', 'ancient', 'historical', 'civilization']):
                    score += 5
            
            # Contextual scoring based on surrounding words
            for word in contextual_words:
                if len(word) > 3:  # Only meaningful words
                    if word in description or word in label:
                        score += 2
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = result
        
        # Only return if we have a reasonable confidence score
        return best_match if best_score > 0 else None

    def link_to_wikipedia_contextual(self, entities, text_context):
        """Add contextual Wikipedia linking using surrounding words for better disambiguation."""
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Get entity context from extraction
                entity_context = entity.get('context', text_context)
                context_window = entity.get('context_window', {})
                
                # Create context-aware search terms using surrounding words
                search_terms = [entity['text']]
                
                # Add contextual information from surrounding words
                contextual_keywords = []
                if context_window.get('before'):
                    words = [w for w in context_window['before'].split() if len(w) > 2]
                    contextual_keywords.extend(words[-2:])  # Last 2 meaningful words
                if context_window.get('after'):
                    words = [w for w in context_window['after'].split() if len(w) > 2]
                    contextual_keywords.extend(words[:2])   # First 2 meaningful words
                
                # Create contextual search terms
                for keyword in contextual_keywords:
                    keyword_clean = keyword.strip('.,;:!?"()[]{}').lower()
                    # Skip common words
                    if keyword_clean not in ['the', 'and', 'with', 'from', 'near', 'this', 'that', 'was', 'were', 'are']:
                        search_terms.append(f"{entity['text']} {keyword}")
                
                # Add context modifiers from global context
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
                    for modifier in context_modifiers[:2]:  # Top 2 modifiers
                        search_terms.append(f"{entity['text']} {modifier}")
                        if modifier != 'historical':  # Avoid double historical
                            search_terms.append(f"{modifier} {entity['text']}")
                
                elif entity['type'] == 'PERSON' and context_modifiers:
                    for modifier in context_modifiers[:2]:
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
                search_terms = list(dict.fromkeys(search_terms))[:5]  # Limit to 5 searches
                
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
                            
                            # Enhanced validation using contextual information
                            snippet = result.get('snippet', '').lower()
                            title_lower = page_title.lower()
                            
                            # Skip clearly irrelevant results using context
                            skip_terms = ['video game', 'software', 'app', 'company', 'corporation', 'brand']
                            if entity_context.get('period') and entity_context['period'] != 'modern':
                                if any(term in snippet or term in title_lower for term in skip_terms):
                                    continue  # Try next search term
                            
                            # Additional contextual validation
                            relevant = False
                            
                            # Check if result matches contextual expectations
                            if contextual_keywords:
                                for keyword in contextual_keywords:
                                    if keyword.lower() in snippet or keyword.lower() in title_lower:
                                        relevant = True
                                        break
                            
                            # If no contextual match found but it's the basic entity search, accept it
                            if not relevant and search_term == entity['text']:
                                relevant = True
                            
                            # Accept result if it seems reasonable and relevant
                            if relevant:
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

    def link_to_britannica(self, entities):
        """Add basic Britannica linking with contextual search.""" 
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                # Create contextual search terms
                search_terms = [entity['text']]
                
                # Add context from surrounding words
                context_window = entity.get('context_window', {})
                if context_window.get('before') or context_window.get('after'):
                    contextual_words = []
                    if context_window.get('before'):
                        contextual_words.extend(context_window['before'].split()[-1:])
                    if context_window.get('after'):
                        contextual_words.extend(context_window['after'].split()[:1])
                    
                    for word in contextual_words:
                        if len(word) > 2 and word.lower() not in ['the', 'and', 'with']:
                            search_terms.append(f"{entity['text']} {word}")
                
                # Try each search term
                for search_term in search_terms[:3]:  # Limit to 3 attempts
                    search_url = "https://www.britannica.com/search"
                    params = {'query': search_term}
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
                                entity['britannica_search_term'] = search_term
                                break
                        
                        if entity.get('britannica_url'):
                            break  # Found a match, stop searching
                
                    time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses ONLY."""
        for entity in entities:
            # Only process ADDRESS entities (places are handled by get_coordinates)
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
    Streamlit wrapper for the enhanced LLM Entity Linker class.
    
    Provides the same interface as the NLTK version but uses LLM for entity extraction
    with contextual linking, unbiased global geocoding, and overlap resolution.
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
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("🖼️ Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using LLM")
        st.markdown("**Extract and link named entities from text using Gemini LLM with contextual disambiguation and overlap resolution**")
        
        # Create enhanced process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Gemini LLM Entity Recognition + Context Analysis + Overlap Resolution</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Contextual Linking to Knowledge Bases:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata</strong>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikipedia/Britannica</strong>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Global Geocoding</strong>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Output Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #EFCA89;">
                         <strong>JSON-LD Export</strong>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #C3B5AC;">
                         <strong>HTML Export</strong>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with enhanced information."""
        st.sidebar.subheader("Enhanced Entity Linking")
        st.sidebar.info("Entities are extracted using Gemini LLM with contextual analysis of surrounding words for better disambiguation. Overlapping entities are resolved to prevent duplicates. Links to Wikidata first, then Wikipedia, then Britannica as fallbacks.")
        
        st.sidebar.subheader("Global Geocoding")
        if PYCOUNTRY_AVAILABLE:
            st.sidebar.success("Global coverage: 195+ countries using pycountry library. Only place entities (GPE, LOCATION, FACILITY, ADDRESS) are geocoded.")
        else:
            st.sidebar.warning("Limited country detection. Install pycountry for full global coverage: `pip install pycountry`")

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        sample_text = ""       
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,
            height=200,
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md) to replace the text above"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    if not analysis_title:
                        import os
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "text_analysis"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """Process the input text using the enhanced LLM EntityLinker with contextual linking and overlap resolution."""
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text and extracting entities with contextual analysis and overlap resolution..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: analyse text context
                status_text.text("Analyzing text context...")
                progress_bar.progress(10)
                text_context = self.entity_linker.analyse_text_context(text)
                
                # Step 2: Extract entities with context and overlap resolution
                status_text.text("Extracting entities with context analysis and overlap resolution using Gemini LLM...")
                progress_bar.progress(25)
                entities_json = self.cached_extract_entities(text)
                entities = json.loads(entities_json)
                
                if not entities:
                    st.warning("No entities found in the text.")
                    return
                
                place_entities = [e for e in entities if e['type'] in self.entity_linker.geocodable_types]
                
                # Step 3: Link to Wikidata with contextual disambiguation
                status_text.text("Linking to Wikidata with contextual disambiguation...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Contextual Wikipedia linking
                status_text.text("Linking to Wikipedia with surrounding word context...")
                progress_bar.progress(60)
                entities = self.entity_linker.link_to_wikipedia_contextual(entities, text_context)
                
                # Step 5: Link to Britannica with context
                status_text.text("Linking to Britannica with contextual search...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 6: Geocode PLACE ENTITIES ONLY
                if place_entities:
                    status_text.text(f"Geocoding {len(place_entities)} place entities (global coverage)...")
                    progress_bar.progress(85)
                    
                    try:
                        entities = self.entity_linker.get_coordinates(entities, text)
                        geocoded_count = len([e for e in entities if e.get('latitude') is not None and e['type'] in self.entity_linker.geocodable_types])
                        
                        if geocoded_count > 0:
                            status_text.text(f"Successfully geocoded {geocoded_count}/{len(place_entities)} places")
                        else:
                            status_text.text("Geocoding completed (no coordinates found)")
                            
                    except Exception as e:
                        st.warning(f"Some geocoding failed: {e}")
                else:
                    status_text.text("No place entities found for geocoding")
                    progress_bar.progress(85)
                
                # Step 7: Link addresses to OpenStreetMap
                status_text.text("Linking addresses to OpenStreetMap...")
                progress_bar.progress(90)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 8: Generate visualisation
                status_text.text("Generating enhanced visualisation...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                st.session_state.text_context = text_context
                
                progress_bar.empty()
                status_text.empty()
                
                # Count linked entities
                linked_entities = []
                for entity in entities:
                    has_links = (entity.get('britannica_url') or 
                                entity.get('wikidata_url') or 
                                entity.get('wikipedia_url') or     
                                entity.get('openstreetmap_url'))
                    has_coordinates = entity.get('latitude') is not None
                    
                    if has_links or has_coordinates:
                        linked_entities.append(entity)
                
                geocoded_places = len([e for e in linked_entities if e.get('latitude') is not None])
                total_places = len([e for e in linked_entities if e['type'] in self.entity_linker.geocodable_types])
                
                success_message = f"Processing complete! Found {len(linked_entities)} contextually linked entities"
                if total_places > 0:
                    success_message += f" ({geocoded_places}/{total_places} places geocoded)"
                
                unlinked_count = len(entities) - len(linked_entities)
                if unlinked_count > 0:
                    success_message += f" ({unlinked_count} entities found but not linked)"
                
                st.success(success_message)
                
                if text_context['period'] or text_context['region'] or text_context['subject_matter']:
                    context_info = []
                    if text_context['period']:
                        context_info.append(f"Period: {text_context['period'].replace('_', ' ').title()}")
                    if text_context['region']:
                        context_info.append(f"Region: {text_context['region'].replace('_', ' ').title()}")
                    if text_context['subject_matter']:
                        context_info.append(f"Subject: {text_context['subject_matter'].title()}")
                    
                    st.info(f"Context detected: {' | '.join(context_info)}")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities for display."""
        colors = self.entity_linker.colors
        
        char_entity_map = [None] * len(text)
        
        for entity in entities:
            has_links = (entity.get('britannica_url') or 
                        entity.get('wikidata_url') or 
                        entity.get('wikipedia_url') or     
                        entity.get('openstreetmap_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_links or has_coordinates):
                continue
                
            start = entity.get('start', -1)
            end = entity.get('end', -1)
            
            if start < 0 or end > len(text) or start >= end:
                continue
                
            for i in range(start, min(end, len(text))):
                if char_entity_map[i] is None:
                    char_entity_map[i] = entity
        
        result = []
        i = 0
        while i < len(text):
            if char_entity_map[i] is not None:
                entity = char_entity_map[i]
                entity_start = i
                
                while i < len(text) and char_entity_map[i] == entity:
                    i += 1
                entity_end = i
                
                entity_text = text[entity_start:entity_end]
                escaped_text = html_module.escape(entity_text)
                
                color = colors.get(entity['type'], '#E7E2D2')
                
                tooltip_parts = [f"Type: {entity['type']}"]
                if entity.get('wikidata_description'):
                    desc = entity['wikidata_description'][:100]
                    tooltip_parts.append(f"Description: {desc}")
                if entity.get('location_name'):
                    loc = entity['location_name'][:100]
                    tooltip_parts.append(f"Location: {loc}")
                if entity.get('search_context'):
                    tooltip_parts.append(f"Context: {entity['search_context']}")
                
                tooltip = html_module.escape(" | ".join(tooltip_parts))
                
                url = (entity.get('wikipedia_url') or
                      entity.get('wikidata_url') or
                      entity.get('britannica_url') or
                      entity.get('openstreetmap_url'))
                
                if url:
                    url = html_module.escape(url)
                    result.append(
                        f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; '
                        f'border-radius: 3px; text-decoration: none; color: black;" '
                        f'target="_blank" title="{tooltip}">{escaped_text}</a>'
                    )
                else:
                    result.append(
                        f'<span style="background-color: {color}; padding: 2px 4px; '
                        f'border-radius: 3px;" title="{tooltip}">{escaped_text}</span>'
                    )
            else:
                result.append(html_module.escape(text[i]))
                i += 1
        
        return ''.join(result)

    def render_results(self):
        """Render the results section with entities and visualisations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        st.subheader("Highlighted Text with Contextual Links")
        if st.session_state.html_content:
            st.markdown(st.session_state.html_content, unsafe_allow_html=True)
        else:
            st.info("No highlighted text available. Process some text first.")
        
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details with contextual information."""
        if not entities:
            st.info("No entities found.")
            return
        
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
        
        table_data = []
        for entity in sorted_entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Context': entity.get('context_window', {}).get('context_snippet', '')[:100] + "..." if entity.get('context_window', {}).get('context_snippet', '') else 'N/A',
                'Links': self.format_entity_links(entity)
            }
            
            if entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            
            if entity.get('latitude') is not None and entity['type'] in self.entity_linker.geocodable_types:
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
        links = []
        
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        
        if not links:
            return "No links"
        
        return " | ".join(links)

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results with enhanced JSON-LD."""
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export - create enhanced JSON-LD format
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "entities": [],
                "processingInfo": {
                    "entityExtraction": "Gemini LLM with contextual analysis and overlap resolution",
                    "geocodingMethod": "Global coverage (places only)",
                    "linkingStrategy": "Contextual disambiguation using surrounding words",
                    "globalCoverage": PYCOUNTRY_AVAILABLE
                }
            }
            
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end']
                }
                
                if entity.get('context_window'):
                    entity_data['context'] = {
                        "before": entity['context_window'].get('before', ''),
                        "after": entity['context_window'].get('after', ''),
                        "snippet": entity['context_window'].get('context_snippet', '')
                    }
                
                if entity.get('wikidata_url'):
                    entity_data['sameAs'] = entity['wikidata_url']
                
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                
                if entity.get('latitude') and entity.get('longitude') and entity['type'] in self.entity_linker.geocodable_types:
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                
                additional_links = []
                if entity.get('wikipedia_url'):
                    additional_links.append(entity['wikipedia_url'])
                if entity.get('britannica_url'):
                    additional_links.append(entity['britannica_url'])
                if entity.get('openstreetmap_url'):
                    additional_links.append(entity['openstreetmap_url'])
                
                if additional_links:
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs']] + additional_links
                        else:
                            entity_data['sameAs'].extend(additional_links)
                    else:
                        entity_data['sameAs'] = additional_links
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download Enhanced JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities_fixed.jsonld",
                mime="application/ld+json",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.html_content:
                html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Enhanced Entity Analysis with Overlap Resolution</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .metadata {{
            background-color: #f5f5f5;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="metadata">
        <h3>Processing Information</h3>
        <p><strong>Method:</strong> LLM Entity Extraction with Contextual Linking and Overlap Resolution</p>
        <p><strong>Geocoding:</strong> Global coverage (places only) - {"195+ countries" if PYCOUNTRY_AVAILABLE else "Limited coverage"}</p>
        <p><strong>Disambiguation:</strong> Uses surrounding words for better linking accuracy</p>
        <p><strong>Overlap Resolution:</strong> Eliminates duplicate entities like "Amazon" vs "Amazon rainforest"</p>
    </div>
    {st.session_state.html_content}
</body>
</html>"""
                
                st.download_button(
                    label="Download Enhanced HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities_fixed.html",
                    mime="text/html",
                    use_container_width=True
                )

    def run(self):
        """Main application runner."""
        self.render_header()
        self.render_sidebar()
        
        text_input, analysis_title = self.render_input_section()
        
        if st.button("Process Text", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyse.")
        
        st.markdown("---")
        self.render_results()


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitLLMEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
