#!/usr/bin/env python3
"""
Streamlit LLM Entity Linker Application

A web interface for entity extraction using LLM (Gemini) with intelligent linking and geocoding.
This application uses LLM for both entity extraction AND disambiguation of Wikipedia links.

Author: Enhanced from NER_LLM_entities_streamlit.py
Version: 3.0 - Added LLM-powered disambiguation
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
    
    This class uses Gemini for entity extraction AND intelligent disambiguation
    of Wikipedia links based on context.
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
        
        # analyse context if not provided
        if context is None:
            context = self.analyse_text_context(text)
        
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
8. IMPORTANT: Adjectives and demonyms (Egyptian, Persian, Greek, Roman, etc.) when used as modifiers (e.g., "Egyptian merchandise", "Persian learned men") should NOT be tagged as entities. Only tag them when they refer to the people as a group (e.g., "The Egyptians built pyramids")
9. Only extract proper nouns and named entities, not common descriptive adjectives

ENTITY TYPES:
- PERSON: Individual people, historical figures, characters, roles with names (e.g., "Io", "Inachus", NOT "daughter" or "king" without names)
- ORGANIZATION: Named groups, institutions, civilizations, companies, teams (e.g., "Phoenicians" as a people, NOT "Egyptian" as an adjective)
- GPE: Cities, countries, regions, kingdoms, territories, political entities (e.g., "Egypt", "Argos", "Hellas")
- LOCATION: Geographic places, landmarks, natural features (e.g., "Red Sea", specific named locations)
- FACILITY: Buildings, venues, structures, stages, theaters
- ADDRESS: Street addresses, property descriptions
- PRODUCT: Objects, tools, components, materials, technical features (NOT "merchandise" or "cargo" unless specifically named)
- EVENT: Named events, ceremonies, performances, battles
- WORK_OF_ART: Books, plays, poems, manuscripts, artworks
- LANGUAGE: Languages, dialects, scripts
- LAW: Legal documents, laws, regulations
- DATE: Specific dates, periods, years, eras, reigns
- MONEY: Currencies, amounts, prices

CRITICAL RULES FOR HISTORICAL TEXTS:
- "Phoenicians", "Persians", "Greeks", "Egyptians" = ORGANIZATION only when referring to the people as a group
- "Egyptian merchandise", "Persian learned men", "Greek ships" = DO NOT tag the adjectives
- Place names like "Egypt", "Persia", "Greece", "Argos" = GPE
- Named individuals like "Io", "Inachus" = PERSON
- Unnamed roles like "the king", "the daughter" without names = DO NOT tag
- "Red Sea" or similar named bodies of water = LOCATION

{examples}

Now extract entities from this text, using context clues to determine correct interpretations:

Text: "{text}"

Remember: Only extract clear named entities. Avoid tagging adjectives, demonyms when used as modifiers, or generic terms.

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

    def get_wikipedia_candidates(self, entity_text, limit=5):
        """Get multiple Wikipedia candidates for disambiguation."""
        
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': entity_text,
            'srlimit': limit,
            'srnamespace': 0  # Only search main namespace
        }
        
        try:
            headers = {'User-Agent': 'EntityLinker/1.0'}
            response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                candidates = []
                
                for result in data.get('query', {}).get('search', []):
                    # Skip obviously bad results
                    snippet = result.get('snippet', '').lower()
                    title = result.get('title', '').lower()
                    
                    # Filter out unwanted results
                    if any(term in snippet or term in title for term in [
                        'wiktionary', 'look up', 'disambiguation page',
                        'may refer to:', 'disambiguation'
                    ]):
                        continue
                    
                    # Clean up the snippet
                    clean_snippet = re.sub(r'<[^>]+>', '', result.get('snippet', ''))
                    
                    candidates.append({
                        'title': result['title'],
                        'description': clean_snippet,
                        'url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(result['title'].replace(' ', '_'))}"
                    })
                
                return candidates
                
        except Exception as e:
            print(f"Error getting Wikipedia candidates: {e}")
            
        return []

    def llm_disambiguate_wikipedia(self, entity, candidates, full_text):
        """Use Gemini to pick the best Wikipedia match based on context."""
        
        if not candidates:
            return None
            
        if len(candidates) == 1:
            return candidates[0]
        
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return candidates[0]  # Fallback to first result
            
            # Get broader context snippet around the entity
            start = max(0, entity['start'] - 200)
            end = min(len(full_text), entity['end'] + 200)
            context = full_text[start:end]
            
            # Analyze the broader text for historical/cultural markers
            text_indicators = self._analyze_text_for_disambiguation(full_text)
            
            # Prepare candidates for LLM
            candidates_text = []
            for i, candidate in enumerate(candidates):
                description = candidate['description'][:400]  # More description for better analysis
                candidates_text.append(
                    f"{i+1}. {candidate['title']}: {description}"
                )
            
            # Enhanced prompt with specific historical context awareness
            prompt = f"""You are an expert historian helping disambiguate a Wikipedia link. You must consider the HISTORICAL CONTEXT carefully.

ENTITY: "{entity['text']}" (Type: {entity['type']})

LOCAL CONTEXT: "...{context}..."

TEXT ANALYSIS: {text_indicators}

CANDIDATES:
{chr(10).join(candidates_text)}

CRITICAL DISAMBIGUATION RULES:
1. If the text contains ancient/historical references (Persian, Phoenician, Egyptian, mythological names), ALWAYS prefer ancient/historical/mythological entities over modern ones
2. "Hellas" in ancient context = ancient Greece, NOT modern football clubs or companies
3. Names like "Inachus" in mythological context = mythological figures (gods, kings, river gods), NOT just geographical features
4. Ancient place names should link to their historical significance, not modern equivalents
5. If text mentions "Persian learned men", "Phoenicians", "Egyptian merchandise" - this is ANCIENT HISTORICAL CONTEXT

SPECIFIC ENTITY GUIDANCE:
- "Hellas" with ancient Greek context → Ancient Greece/Classical Greece
- "Inachus" with mythology/royal context → Mythological king/river god of Argos
- "Argos" with ancient context → Ancient Greek city-state
- Ancient peoples (Phoenicians, Persians) → Historical civilizations

Which candidate (1-{len(candidates)}) is the BEST match for this entity in this HISTORICAL context?
If NONE are good matches, respond with "NONE".

Response format: Just the number (1-{len(candidates)}) or "NONE"
Brief reasoning: Why this choice fits the historical context?
"""

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse the response
            if "NONE" in result.upper():
                return None
                
            # Extract number
            match = re.search(r'(\d+)', result)
            if match:
                choice = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= choice < len(candidates):
                    return candidates[choice]
            
            # Enhanced fallback heuristics for historical context
            entity_context = entity.get('context', {})
            entity_text_lower = entity['text'].lower()
            
            # Special handling for known historical entities
            if entity_text_lower == 'hellas' and text_indicators.get('ancient_context'):
                # Look for ancient Greece specifically
                for candidate in candidates:
                    title_lower = candidate['title'].lower()
                    desc_lower = candidate['description'].lower()
                    if any(term in title_lower or term in desc_lower for term in [
                        'ancient greece', 'classical greece', 'greek civilization', 'ancient greek'
                    ]):
                        return candidate
            
            elif entity_text_lower == 'inachus' and text_indicators.get('mythological_context'):
                # Look for mythological figure, not just river
                for candidate in candidates:
                    desc_lower = candidate['description'].lower()
                    if any(term in desc_lower for term in [
                        'mythology', 'mythological', 'god', 'king', 'father', 'argos', 'io'
                    ]):
                        return candidate
            
            # General ancient context preference
            if text_indicators.get('ancient_context') or entity_context.get('period') == 'ancient':
                for candidate in candidates:
                    desc_lower = candidate['description'].lower()
                    title_lower = candidate['title'].lower()
                    
                    # Strong preference for ancient/historical/mythological
                    if any(term in desc_lower or term in title_lower for term in [
                        'ancient', 'classical', 'mythology', 'mythological', 
                        'greek', 'roman', 'historical', 'antiquity', 'bc', 'bce',
                        'legendary', 'traditional', 'epic', 'homer'
                    ]):
                        # But avoid modern things wrongly tagged
                        if not any(avoid in desc_lower or avoid in title_lower for avoid in [
                            'football', 'soccer', 'club', 'team', 'company', 'corporation',
                            'modern', 'contemporary', '20th century', '21st century',
                            'founded', 'established', 'fc ', 'f.c.'
                        ]):
                            return candidate
            
            # Default fallback
            return candidates[0]
                
        except Exception as e:
            print(f"LLM disambiguation failed: {e}")
            
        # Fallback to first candidate
        return candidates[0] if candidates else None

    def _analyze_text_for_disambiguation(self, text: str) -> Dict[str, bool]:
        """Analyze the full text to provide context clues for disambiguation."""
        text_lower = text.lower()
        
        indicators = {
            'ancient_context': False,
            'mythological_context': False,
            'classical_context': False,
            'historical_narrative': False
        }
        
        # Check for ancient/historical markers
        ancient_markers = [
            'persian learned men', 'phoenicians', 'egyptian', 'assyrian',
            'ancient', 'antiquity', 'classical', 'mythology', 'mythological',
            'bc', 'bce', 'herodotus', 'homer', 'came to our seas',
            'at that time', 'daughter of the king', 'long voyages',
            'merchandise', 'cargo', 'wares', 'preeminent', 'sailed away'
        ]
        
        if any(marker in text_lower for marker in ancient_markers):
            indicators['ancient_context'] = True
        
        # Check for mythological context
        mythological_markers = [
            'according to persians and greeks alike', 'daughter of', 'king',
            'mythology', 'mythological', 'god', 'goddess', 'deity',
            'legendary', 'epic', 'traditional story'
        ]
        
        if any(marker in text_lower for marker in mythological_markers):
            indicators['mythological_context'] = True
        
        # Check for classical period
        classical_markers = [
            'hellas', 'greek', 'persian', 'phoenician', 'argos',
            'classical', 'ancient greece', 'ancient greek'
        ]
        
        if any(marker in text_lower for marker in classical_markers):
            indicators['classical_context'] = True
        
        # Check for historical narrative style
        narrative_markers = [
            'say that', 'they say', 'according to', 'at that time',
            'came to', 'sailed', 'settled', 'occupied'
        ]
        
        if any(marker in text_lower for marker in narrative_markers):
            indicators['historical_narrative'] = True
        
        return indicators

    def link_to_wikipedia_with_llm_disambiguation(self, entities, full_text):
        """Use LLM to help disambiguate Wikipedia results based on context - LAST RESORT only."""
        
        for entity in entities:
            # Skip if already has higher priority links (Wikidata, Getty AAT, or Britannica)
            if (entity.get('wikidata_url') or 
                entity.get('getty_aat_url') or 
                entity.get('britannica_url')):
                continue
                
            try:
                # Get multiple Wikipedia candidates
                candidates = self.get_wikipedia_candidates(entity['text'])
                
                if candidates:
                    # Use LLM to pick the best match
                    best_match = self.llm_disambiguate_wikipedia(entity, candidates, full_text)
                    if best_match:
                        entity['wikipedia_url'] = best_match['url']
                        entity['wikipedia_title'] = best_match['title']
                        entity['wikipedia_description'] = best_match['description']
                        entity['disambiguation_method'] = 'llm_contextual'
                        
                        # Add note about which candidates were considered
                        entity['candidates_considered'] = len(candidates)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error in LLM Wikipedia disambiguation for {entity['text']}: {e}")
        
        return entities

    def get_coordinates(self, entities, processed_text=""):
        """Enhanced coordinate lookup with LLM-powered geographical context detection."""
        # Use LLM to detect geographical context from the full text
        geographical_context = self._llm_detect_geographical_context(processed_text, entities)
        
        if geographical_context:
            print(f"LLM detected geographical context: {geographical_context}")
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try LLM-enhanced contextual geocoding first
                if self._try_llm_contextual_geocoding(entity, geographical_context, processed_text):
                    continue
                    
                # Fall back to pattern-based contextual geocoding
                context_clues = self._detect_geographical_context(processed_text, entities)
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Final fallback to OpenStreetMap
                if self._try_openstreetmap(entity):
                    continue
        
        return entities

    def _llm_detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Use LLM to intelligently detect geographical context from the full text."""
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return ""
            
            # Extract place entities for context
            place_entities = [e['text'] for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY']]
            
            prompt = f"""Analyze this text to determine the PRIMARY geographical context for geocoding purposes.

TEXT: "{text[:1000]}..."

PLACE ENTITIES FOUND: {', '.join(place_entities)}

What is the PRIMARY geographical region/country/city context for this text? 
This will be used to disambiguate place names for geocoding.

Consider:
- Are there explicit location mentions?
- Historical/cultural context clues?
- Language/terminology that suggests a region?
- Time period that suggests a location?

Respond with ONLY the most specific geographical context you can determine, in this format:
- For modern texts: "London, UK" or "New York, USA" or "Paris, France"
- For historical texts: "Ancient Greece" or "Roman Empire" or "Medieval England"
- For unclear contexts: "Europe" or "North America" or "Unknown"

Response (geographical context only):"""

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            context = response.text.strip()
            
            # Clean up the response to extract just the location
            context = context.replace('"', '').replace("'", '').strip()
            
            return context
            
        except Exception as e:
            print(f"LLM geographical context detection failed: {e}")
            return ""

    def _try_llm_contextual_geocoding(self, entity, geographical_context, full_text):
        """Try geocoding using LLM-detected geographical context."""
        if not geographical_context or geographical_context.lower() == 'unknown':
            return False
        
        # Create context-aware search terms using LLM insights
        search_variations = [entity['text']]
        
        # Add LLM-detected context
        if geographical_context:
            search_variations.append(f"{entity['text']}, {geographical_context}")
            
            # Handle historical contexts by modernizing them
            context_mappings = {
                'Ancient Greece': 'Greece',
                'Roman Empire': 'Italy',
                'Medieval England': 'England, UK',
                'Victorian London': 'London, UK',
                'Classical Athens': 'Athens, Greece',
                'Ancient Rome': 'Rome, Italy'
            }
            
            modern_context = context_mappings.get(geographical_context, geographical_context)
            if modern_context != geographical_context:
                search_variations.append(f"{entity['text']}, {modern_context}")
        
        # For FACILITY entities, be more specific about the type
        if entity['type'] == 'FACILITY':
            # Extract surrounding context to understand what kind of facility
            entity_context = self._extract_facility_context(entity, full_text)
            if entity_context:
                search_variations.append(f"{entity['text']} {entity_context}, {geographical_context}")
        
        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))
        
        # Try OpenStreetMap with LLM-enhanced context
        for search_term in search_variations[:3]:  # Try top 3 variations
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
                        entity['geocoding_source'] = 'llm_contextual'
                        entity['search_term_used'] = search_term
                        entity['llm_geographical_context'] = geographical_context
                        return True
            
                time.sleep(0.3)  # Rate limiting
            except Exception:
                continue
        
        return False

    def _extract_facility_context(self, entity, full_text):
        """Extract context around a facility entity to determine its type."""
        # Get text around the entity
        start = max(0, entity['start'] - 50)
        end = min(len(full_text), entity['end'] + 50)
        context = full_text[start:end].lower()
        
        # Look for facility type indicators
        facility_indicators = {
            'theatre': ['theatre', 'theater', 'play', 'performance', 'stage'],
            'hospital': ['hospital', 'medical', 'clinic', 'ward'],
            'school': ['school', 'university', 'college', 'academy'],
            'church': ['church', 'cathedral', 'abbey', 'monastery'],
            'hotel': ['hotel', 'inn', 'lodge', 'accommodation'],
            'museum': ['museum', 'gallery', 'exhibition'],
            'library': ['library', 'archive', 'collection']
        }
        
        for facility_type, indicators in facility_indicators.items():
            if any(indicator in context for indicator in indicators):
                return facility_type
        
        return ""

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

    def link_to_wikidata(self, entities):
        """Add context-aware Wikidata linking."""
        for entity in entities:
            try:
                # Get context from entity if available
                entity_context = entity.get('context', {})
                
                # Prepare search query with context
                search_query = entity['text']
                
                # Add context to search for better disambiguation
                if entity['type'] == 'GPE':
                    # For places, add geographical context
                    if entity_context.get('period') == 'ancient':
                        search_query = f"{entity['text']} ancient city"
                    elif entity_context.get('region') == 'mediterranean':
                        search_query = f"{entity['text']} Greece"
                    
                    # Special cases for known ancient places
                    if entity['text'].lower() == 'argos':
                        search_query = "Argos Greece ancient city"
                    elif entity['text'].lower() == 'hellas':
                        search_query = "ancient Greece Hellas"
                
                elif entity['type'] == 'PERSON':
                    # For people in ancient contexts, add mythology/history context
                    if entity_context.get('period') == 'ancient':
                        if entity['text'].lower() == 'io':
                            search_query = "Io mythology"
                        elif entity['text'].lower() == 'inachus':
                            search_query = "Inachus mythology river god"
                        else:
                            search_query = f"{entity['text']} ancient history"
                
                elif entity['type'] == 'LOCATION':
                    # For locations like "Red Sea", search more specifically
                    if 'sea' in entity['text'].lower():
                        search_query = f"{entity['text']} body of water"
                
                # Search Wikidata with enhanced query
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'search': search_query,
                    'language': 'en',
                    'limit': 5,  # Get more results to choose from
                    'type': 'item'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search') and len(data['search']) > 0:
                        # Try to find the best match based on context
                        best_match = None
                        
                        for result in data['search']:
                            description = result.get('description', '').lower()
                            label = result.get('label', '').lower()
                            
                            # Score each result based on context relevance
                            if entity['type'] == 'GPE':
                                # Prefer geographical entities
                                if any(term in description for term in ['city', 'town', 'ancient', 'greece', 'greek', 'historical', 'archaeological', 'country', 'region']):
                                    # Skip if it's clearly wrong (like video games)
                                    if not any(skip in description for skip in ['video game', 'game', 'software', 'album', 'film', 'movie', 'book']):
                                        best_match = result
                                        break
                            
                            elif entity['type'] == 'PERSON':
                                # Prefer mythological/historical figures for ancient texts
                                if entity_context.get('period') == 'ancient':
                                    if any(term in description for term in ['mythology', 'mythological', 'ancient', 'greek', 'deity', 'god', 'goddess', 'hero', 'king', 'queen']):
                                        best_match = result
                                        break
                                # Otherwise just avoid obvious non-persons
                                elif not any(skip in description for skip in ['genus', 'species', 'asteroid', 'crater', 'company']):
                                    best_match = result
                                    break
                            
                            elif entity['type'] == 'LOCATION':
                                # Prefer geographical features
                                if any(term in description for term in ['sea', 'ocean', 'river', 'mountain', 'lake', 'water', 'geographic']):
                                    best_match = result
                                    break
                            
                            elif entity['type'] == 'ORGANIZATION':
                                # Prefer historical/ethnic groups
                                if any(term in description for term in ['people', 'ethnic', 'ancient', 'historical', 'civilization']):
                                    best_match = result
                                    break
                        
                        # Use best match or fall back to first result
                        if best_match:
                            entity['wikidata_url'] = f"http://www.wikidata.org/entity/{best_match['id']}"
                            entity['wikidata_description'] = best_match.get('description', '')
                        else:
                            # If no good match found, use first result but mark it as uncertain
                            result = data['search'][0]
                            entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                            entity['wikidata_description'] = result.get('description', '')
                            # Add warning if description seems wrong
                            if entity['type'] == 'GPE' and any(term in result.get('description', '').lower() for term in ['game', 'software', 'album']):
                                entity['wikidata_description'] = f"[May be incorrect] {result.get('description', '')}"
                
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

    def link_to_getty_aat(self, entities):
        """Add Getty Art & Architecture Thesaurus linking - especially good for PRODUCT, FACILITY, WORK_OF_ART entities."""
        for entity in entities:
            # Skip if already has higher priority links
            if entity.get('wikidata_url'):
                continue
                
            # Getty AAT is particularly valuable for these entity types
            relevant_types = ['PRODUCT', 'FACILITY', 'WORK_OF_ART', 'EVENT', 'ORGANIZATION']
            if entity['type'] not in relevant_types:
                continue
                
            try:
                # Getty AAT SPARQL endpoint
                query = f"""
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                PREFIX gvp: <http://vocab.getty.edu/ontology#>
                
                SELECT ?concept ?prefLabel ?note WHERE {{
                  ?concept skos:inScheme <http://vocab.getty.edu/aat/> ;
                           skos:prefLabel ?prefLabel ;
                           gvp:prefLabelGVP/gvp:term ?term .
                  OPTIONAL {{ ?concept skos:scopeNote/rdf:value ?note }}
                  FILTER(CONTAINS(LCASE(?term), LCASE("{entity['text']}")))
                }}
                LIMIT 3
                """
                
                headers = {
                    'Accept': 'application/sparql-results+json',
                    'User-Agent': 'EntityLinker/1.0'
                }
                
                sparql_endpoint = "http://vocab.getty.edu/sparql"
                response = requests.get(sparql_endpoint, 
                                      params={'query': query}, 
                                      headers=headers, 
                                      timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', {}).get('bindings', [])
                    
                    if results:
                        # Take the first result
                        result = results[0]
                        concept_uri = result['concept']['value']
                        pref_label = result['prefLabel']['value']
                        
                        # Convert to web URL
                        getty_url = concept_uri.replace('http://vocab.getty.edu/aat/', 
                                                      'http://www.getty.edu/vow/AATFullDisplay?find=&logic=AND&note=&page=1&subjectid=')
                        
                        entity['getty_aat_url'] = getty_url
                        entity['getty_aat_label'] = pref_label
                        
                        if 'note' in result:
                            entity['getty_aat_description'] = result['note']['value'][:200]
                
                time.sleep(0.3)  # Rate limiting for Getty
                
            except Exception as e:
                # Getty AAT can be unreliable, continue silently
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add basic Britannica linking - only for entities without higher priority links.""" 
        for entity in entities:
            # Skip if already has Wikidata or Getty AAT link
            if entity.get('wikidata_url') or entity.get('getty_aat_url'):
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
    
    Provides the same interface as the NLTK version but uses LLM for entity extraction
    AND intelligent disambiguation of Wikipedia links.
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
                st.info("Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            # If there's any error loading the logo, continue without it
            st.warning(f"Could not load logo: {e}")        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using LLM")
        st.markdown("**Extract and link named entities from text using Gemini LLM with intelligent disambiguation**")
        
        # Create a simple process diagram - enhanced with LLM disambiguation
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
                <div style="background-color: #BF7B69; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>LLM-Powered Disambiguation</strong><br><small>Context-aware linking decisions</small>
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
                    <div style="background-color: #E6D7C9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
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
        """Render the sidebar with information about LLM disambiguation."""
        # Entity linking information
        st.sidebar.subheader("Smart Entity Linking")
        st.sidebar.info("Entities are linked with priority hierarchy: 1) Wikidata (authoritative structured data), 2) Getty AAT (art & architecture terminology), 3) Britannica (scholarly encyclopedia), 4) Wikipedia (general reference). The LLM provides intelligent disambiguation for the final step.")
        
        st.sidebar.subheader("Geocoding")
        st.sidebar.info("Places and addresses are geocoded using multiple services with contextual hints for accurate coordinates.")

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
        Process the input text using the LLM EntityLinker with contextual analysis and intelligent disambiguation.
        
        Args:
            text: Input text to process
            title: Analysis title
        """
        if not text.strip():
            st.warning("Please enter some text to analyse.")
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
                
                # Step 1: analyse text context for better linking
                status_text.text("Analyzing text context...")
                progress_bar.progress(10)
                text_context = self.entity_linker.analyse_text_context(text)
                
                # Step 2: Extract entities using LLM (cached)
                status_text.text("Extracting entities using Gemini LLM...")
                progress_bar.progress(25)
                entities_json = self.cached_extract_entities(text)
                entities = json.loads(entities_json)
                
                if not entities:
                    st.warning("No entities found in the text.")
                    return
                
                # Step 3: Link to Wikidata (cached) - FIRST PRIORITY
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(40)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Link to Getty AAT - SECOND PRIORITY
                status_text.text("Linking to Getty Art & Architecture Thesaurus...")
                progress_bar.progress(50)
                entities = self.entity_linker.link_to_getty_aat(entities)
                
                # Step 5: Link to Britannica - THIRD PRIORITY
                status_text.text("Linking to Britannica...")
                progress_bar.progress(60)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 6: LLM-powered Wikipedia disambiguation - LAST RESORT
                status_text.text("Smart Wikipedia linking with LLM disambiguation...")
                progress_bar.progress(70)
                entities = self.entity_linker.link_to_wikipedia_with_llm_disambiguation(entities, text)
                
                # Step 7: Get coordinates
                status_text.text("Getting coordinates...")
                progress_bar.progress(80)
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
                
                # Step 8: Link addresses to OpenStreetMap
                status_text.text("Linking addresses to OpenStreetMap...")
                progress_bar.progress(90)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 9: Generate visualisation
                status_text.text("Generating visualisation...")
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
                
                # Show context analysis results and disambiguation stats
                disambiguation_stats = self._get_disambiguation_stats(entities)
                
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
                
                # Show disambiguation info
                if disambiguation_stats['llm_disambiguated'] > 0:
                    st.info(f"LLM intelligently disambiguated {disambiguation_stats['llm_disambiguated']} Wikipedia links using context")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def _get_disambiguation_stats(self, entities):
        """Get statistics about disambiguation methods used."""
        stats = {
            'llm_disambiguated': 0,
            'single_candidate': 0,
            'fallback': 0
        }
        
        for entity in entities:
            if entity.get('disambiguation_method') == 'llm_contextual':
                stats['llm_disambiguated'] += 1
            elif entity.get('candidates_considered') == 1:
                stats['single_candidate'] += 1
        
        return stats

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities for display.
        Uses a character-by-character replacement approach to avoid position shifting.
        """
        colors = {
            'PERSON': '#BF7B69',
            'ORGANIZATION': '#9fd2cd',
            'GPE': '#C4C3A2',
            'LOCATION': '#EFCA89',
            'FACILITY': '#C3B5AC',
            'GSP': '#C4A998',
            'ADDRESS': '#CCBEAA',
            'PRODUCT': '#E6D7C9',
            'EVENT': '#D4C5B9',
            'WORK_OF_ART': '#E8E1D4',
            'LANGUAGE': '#F0EAE2',
            'LAW': '#DDD6CE',
            'DATE': '#E3DDD7',
            'MONEY': '#D6CFCA'
        }
        
        # Create a list to track which characters belong to which entity
        char_entity_map = [None] * len(text)
        
        # Map each character position to its entity (if any)
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
            
            # Skip if positions are invalid
            if start < 0 or end > len(text) or start >= end:
                continue
                
            # Mark characters as belonging to this entity
            for i in range(start, min(end, len(text))):
                if char_entity_map[i] is None:  # Don't overwrite existing entities
                    char_entity_map[i] = entity
        
        # Build the HTML output
        result = []
        i = 0
        while i < len(text):
            if char_entity_map[i] is not None:
                # Start of an entity
                entity = char_entity_map[i]
                entity_start = i
                
                # Find the end of this entity
                while i < len(text) and char_entity_map[i] == entity:
                    i += 1
                entity_end = i
                
                # Extract the entity text
                entity_text = text[entity_start:entity_end]
                escaped_text = html_module.escape(entity_text)
                
                # Build the link/span
                color = colors.get(entity['type'], '#E7E2D2')
                
                tooltip_parts = [f"Type: {entity['type']}"]
                if entity.get('wikidata_description'):
                    desc = entity['wikidata_description'][:100]  # Limit description length
                    tooltip_parts.append(f"Description: {desc}")
                elif entity.get('getty_aat_description'):
                    desc = entity['getty_aat_description'][:100]
                    tooltip_parts.append(f"Getty AAT: {desc}")
                elif entity.get('wikipedia_description'):
                    desc = entity['wikipedia_description'][:100]
                    tooltip_parts.append(f"Description: {desc}")
                if entity.get('location_name'):
                    loc = entity['location_name'][:100]  # Limit location length
                    tooltip_parts.append(f"Location: {loc}")
                if entity.get('disambiguation_method') == 'llm_contextual':
                    tooltip_parts.append("LLM disambiguated")
                
                tooltip = html_module.escape(" | ".join(tooltip_parts))
                
                url = (entity.get('wikidata_url') or
                      entity.get('getty_aat_url') or
                      entity.get('britannica_url') or
                      entity.get('wikipedia_url') or
                      entity.get('openstreetmap_url'))
                
                if url:
                    # Ensure URL is properly escaped
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
                # Regular text - escape HTML characters
                result.append(html_module.escape(text[i]))
                i += 1
        
        return ''.join(result)

    def render_results(self):
        """Render the results section with entities and visualisations - enhanced with disambiguation info."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Show disambiguation statistics
        disambiguation_stats = self._get_disambiguation_stats(entities)
        if disambiguation_stats['llm_disambiguated'] > 0:
            st.markdown(f"""
            <div style="background-color: #E8F4FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 20px;">
                <strong>Smart Disambiguation Applied</strong><br>
                The LLM intelligently selected the best Wikipedia links for <strong>{disambiguation_stats['llm_disambiguated']}</strong> entities 
                by analyzing context, eliminating ambiguous results like "Look up X in Wiktionary" messages.
            </div>
            """, unsafe_allow_html=True)
        
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
        """Render a table of entity details - enhanced with disambiguation info."""
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
            elif entity.get('getty_aat_description'):
                row['Description'] = entity['getty_aat_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description']
            
            if entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            # Add disambiguation method info
            if entity.get('disambiguation_method') == 'llm_contextual':
                row['Disambiguation'] = f"LLM (from {entity.get('candidates_considered', 1)} candidates)"
            elif entity.get('candidates_considered'):
                row['Disambiguation'] = f"Auto ({entity.get('candidates_considered', 1)} candidates)"
            else:
                row['Disambiguation'] = "Direct"
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table - updated priority order."""
        links = []
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('getty_aat_url'):
            links.append("Getty AAT")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results - enhanced with disambiguation metadata."""
        # Stack buttons vertically for mobile
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export - create JSON-LD format with disambiguation info
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "processingMethod": "LLM entity extraction with intelligent disambiguation",
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
                
                # Add disambiguation metadata
                if entity.get('disambiguation_method'):
                    entity_data['disambiguationMethod'] = entity['disambiguation_method']
                if entity.get('candidates_considered'):
                    entity_data['candidatesConsidered'] = entity['candidates_considered']
                
                if entity.get('wikidata_url'):
                    entity_data['sameAs'] = entity['wikidata_url']
                
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('getty_aat_description'):
                    entity_data['description'] = entity['getty_aat_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                
                if entity.get('getty_aat_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['getty_aat_url']]
                        else:
                            entity_data['sameAs'].append(entity['getty_aat_url'])
                    else:
                        entity_data['sameAs'] = entity['getty_aat_url']
                
                if entity.get('britannica_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['britannica_url']]
                        else:
                            entity_data['sameAs'].append(entity['britannica_url'])
                    else:
                        entity_data['sameAs'] = entity['britannica_url']
                
                if entity.get('wikipedia_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['wikipedia_url']]
                        else:
                            entity_data['sameAs'].append(entity['wikipedia_url'])
                    else:
                        entity_data['sameAs'] = entity['wikipedia_url']
                
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
    <title>Entity Analysis - {st.session_state.analysis_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .processing-info {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="processing-info">
        <strong>Generated by LLM Entity Linker</strong><br>
        Entities extracted and disambiguated using Gemini LLM with intelligent context analysis.
    </div>
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
                st.warning("Please enter some text to analyse.")
        
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
