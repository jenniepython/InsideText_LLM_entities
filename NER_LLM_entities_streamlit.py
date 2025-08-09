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
        """Analyse text to determine historical, geographical, and cultural context using LLM intelligence."""
        try:
            import google.generativeai as genai
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Fallback to basic structure if no API key
                return self._basic_context_fallback()
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Let the LLM analyze the context without any hardcoded assumptions
            prompt = f"""Analyze this text to determine its context for entity disambiguation purposes.
    
    TEXT: "{text[:1500]}..."
    
    Provide a comprehensive analysis covering:
    
    1. HISTORICAL PERIOD: What time period does this text reference? (ancient, medieval, renaissance, industrial, modern, contemporary, or specific periods like "Tang Dynasty", "Meiji era", etc.)
    
    2. GEOGRAPHICAL REGION: What geographical/cultural region is the primary focus? (Mediterranean, Europe, Asia, Africa, Americas, Middle East, or specific regions)
    
    3. CULTURAL CONTEXT: What specific culture/civilization is discussed? (Greek, Roman, Chinese, Islamic, Byzantine, Maya, etc.)
    
    4. SUBJECT MATTER: What is the main topic? (history, literature, architecture, theater, military, politics, religion, science, art, etc.)
    
    5. LANGUAGE STYLE: What writing style is used? (academic, narrative, archaic, contemporary, journalistic, literary, etc.)
    
    6. SPECIFIC INDICATORS: What specific words, phrases, or concepts indicate this context?
    
    Respond in this EXACT JSON format:
    {{
        "period": "detected_period_or_null",
        "region": "detected_region_or_null", 
        "culture": "detected_culture_or_null",
        "subject_matter": "detected_subject_or_null",
        "language_style": "detected_style_or_modern",
        "time_indicators": ["list", "of", "temporal", "clues"],
        "place_indicators": ["list", "of", "geographical", "clues"],
        "subject_indicators": ["list", "of", "topical", "clues"],
        "confidence": "high|medium|low",
        "reasoning": "brief explanation of analysis"
    }}
    
    Focus on being accurate and unbiased. If uncertain about any aspect, use null or indicate low confidence.
    """
            
            response = model.generate_content(prompt)
            llm_response = response.text.strip()
            
            # Parse the LLM's context analysis
            context_data = self.extract_json_from_response(llm_response)
            
            if context_data and isinstance(context_data, list) and len(context_data) > 0:
                context = context_data[0]
            elif context_data and isinstance(context_data, dict):
                context = context_data
            else:
                # If LLM response parsing fails, return basic structure
                return self._basic_context_fallback()
            
            # Ensure all required keys exist with defaults
            return {
                'period': context.get('period'),
                'region': context.get('region'),
                'culture': context.get('culture'),
                'subject_matter': context.get('subject_matter'),
                'language_style': context.get('language_style', 'modern'),
                'time_indicators': context.get('time_indicators', []),
                'place_indicators': context.get('place_indicators', []),
                'subject_indicators': context.get('subject_indicators', []),
                'confidence': context.get('confidence', 'medium'),
                'reasoning': context.get('reasoning', 'LLM-based analysis'),
                'analysis_method': 'llm_driven'
            }
            
        except Exception as e:
            print(f"LLM context analysis failed: {e}")
            return self._basic_context_fallback()
    
    def _basic_context_fallback(self):
        """Minimal fallback context structure when LLM analysis fails."""
        return {
            'period': None,
            'region': None,
            'culture': None,
            'subject_matter': None,
            'language_style': 'modern',
            'time_indicators': [],
            'place_indicators': [],
            'subject_indicators': [],
            'confidence': 'low',
            'reasoning': 'Fallback - LLM analysis unavailable',
            'analysis_method': 'fallback'
        }

    def get_wikipedia_candidates(self, entity_text, limit=5):
        """Get multiple Wikipedia candidates for disambiguation - let LLM decide what's relevant."""
        
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': entity_text,
            'srlimit': limit,
            'srnamespace': 0
        }
        
        try:
            headers = {'User-Agent': 'EntityLinker/1.0'}
            response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                candidates = []
                
                for result in data.get('query', {}).get('search', []):
                    # NO FILTERING - let the LLM decide what's relevant
                    clean_snippet = re.sub(r'<[^>]+>', '', result.get('snippet', ''))
                    
                    candidates.append({
                        'title': result['title'],
                        'description': clean_snippet,
                        'url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(result['title'].replace(' ', '_'))}",
                        'type': self._classify_result_type(result['title'], clean_snippet)  # Optional: help LLM with metadata
                    })
                
                return candidates
                
        except Exception as e:
            print(f"Error getting Wikipedia candidates: {e}")
            
        return []
    
    def _classify_result_type(self, title, snippet):
        """Optional: Add metadata to help LLM make better decisions."""
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        if 'disambiguation' in title_lower:
            return 'disambiguation_page'
        elif any(term in snippet_lower for term in ['wiktionary', 'look up']):
            return 'dictionary_reference'  
        else:
            return 'standard_article'

def llm_disambiguate_wikipedia(self, entity, candidates, full_text):
    """Use Gemini to pick the best Wikipedia match based on context - fully LLM-driven."""
    
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
        
        # Get entity context from the LLM-driven analysis
        entity_context = entity.get('context', {})
        
        # Get broader context snippet around the entity
        start = max(0, entity['start'] - 300)
        end = min(len(full_text), entity['end'] + 300)
        local_context = full_text[start:end]
        
        # Prepare candidates for LLM with clean formatting
        candidates_text = []
        for i, candidate in enumerate(candidates):
            candidate_type = candidate.get('type', 'standard_article')
            description = candidate['description'][:500]  # More context for better analysis
            
            candidates_text.append(
                f"{i+1}. Title: {candidate['title']}\n"
                f"   Type: {candidate_type}\n" 
                f"   Description: {description}\n"
            )
        
        # Clean, unbiased prompt that trusts LLM intelligence
        prompt = f"""You are an expert at disambiguating Wikipedia links using contextual analysis.

ENTITY TO DISAMBIGUATE: "{entity['text']}" (Entity Type: {entity['type']})

FULL TEXT CONTEXT: "{full_text[:2000]}..."

LOCAL CONTEXT AROUND ENTITY: "...{local_context}..."

DETECTED CONTEXT (from previous analysis):
- Period: {entity_context.get('period', 'unknown')}
- Region: {entity_context.get('region', 'unknown')}
- Culture: {entity_context.get('culture', 'unknown')}
- Subject Matter: {entity_context.get('subject_matter', 'unknown')}
- Confidence: {entity_context.get('confidence', 'unknown')}

WIKIPEDIA CANDIDATES:
{chr(10).join(candidates_text)}

TASK: Analyze the full context to determine which candidate is the best match for this specific entity in this specific text.

CONSIDER:
1. Historical period and cultural context of the text
2. Subject matter and domain
3. Geographical and temporal alignment
4. Relationship to other entities in the text
5. Semantic coherence with the overall narrative
6. Entity type appropriateness

IMPORTANT: Base your decision on the FULL CONTEXT, not on preconceived notions. Some texts may reference:
- Historical vs. modern entities with the same name
- Mythological vs. geographical features
- Cultural works vs. literal objects
- Academic vs. popular references

Which candidate (1-{len(candidates)}) best matches this entity in this specific context?
If none are appropriate matches, respond with "NONE".

Respond with:
1. Your choice number (1-{len(candidates)}) or "NONE"
2. Confidence level (high/medium/low)
3. Brief reasoning based on contextual analysis

Format:
Choice: [number or NONE]
Confidence: [high/medium/low]
Reasoning: [your analysis]
"""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Parse the LLM response
        choice_match = re.search(r'Choice:\s*(\d+|NONE)', result, re.IGNORECASE)
        confidence_match = re.search(r'Confidence:\s*(high|medium|low)', result, re.IGNORECASE)
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', result, re.IGNORECASE | re.DOTALL)
        
        if choice_match:
            choice_str = choice_match.group(1).upper()
            
            if choice_str == "NONE":
                return None
            
            try:
                choice = int(choice_str) - 1  # Convert to 0-based index
                if 0 <= choice < len(candidates):
                    selected_candidate = candidates[choice]
                    
                    # Add disambiguation metadata
                    selected_candidate['disambiguation_confidence'] = confidence_match.group(1) if confidence_match else 'medium'
                    selected_candidate['disambiguation_reasoning'] = reasoning_match.group(1).strip() if reasoning_match else 'LLM selection'
                    selected_candidate['candidates_available'] = len(candidates)
                    
                    return selected_candidate
            except ValueError:
                pass
        
        # If parsing fails, let LLM try simpler format
        return self._fallback_simple_disambiguation(entity, candidates, full_text)
                
    except Exception as e:
        print(f"LLM disambiguation failed: {e}")
        return self._fallback_simple_disambiguation(entity, candidates, full_text)

def _fallback_simple_disambiguation(self, entity, candidates, full_text):
    """Simplified LLM disambiguation if main method fails."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return candidates[0]
        
        # Super simple prompt as backup
        candidates_simple = [f"{i+1}. {c['title']}: {c['description'][:200]}" 
                           for i, c in enumerate(candidates)]
        
        simple_prompt = f"""Text context: "{full_text[:1000]}..."
Entity: "{entity['text']}"
Options: {chr(10).join(candidates_simple)}

Which option number (1-{len(candidates)}) best fits this entity in this context? Just respond with the number."""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(simple_prompt)
        result = response.text.strip()
        
        # Extract number
        match = re.search(r'(\d+)', result)
        if match:
            choice = int(match.group(1)) - 1
            if 0 <= choice < len(candidates):
                candidates[choice]['disambiguation_method'] = 'llm_simple_fallback'
                return candidates[choice]
        
    except Exception:
        pass
    
    # Ultimate fallback
    if candidates:
        candidates[0]['disambiguation_method'] = 'first_result_fallback'
        return candidates[0]
    
    return None

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
            # Skip if already has higher priority links (Getty AAT, Wikidata, or Britannica)
            if (entity.get('getty_aat_url') or 
                entity.get('wikidata_url') or 
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
                    
                # Final fallback to OpenStreetMap with context
                if self._try_openstreetmap(entity, geographical_context):
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
        """Try geocoding using LLM to create appropriate search terms."""
        if not geographical_context or geographical_context.lower() == 'unknown':
            return False
        
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._basic_contextual_geocoding(entity, geographical_context)
            
            # Let LLM decide how to adapt historical context for modern geocoding
            prompt = f"""You need to help geocode a location for mapping purposes.
    
    ENTITY: "{entity['text']}" (Type: {entity['type']})
    DETECTED CONTEXT: "{geographical_context}"
    SURROUNDING TEXT: "{full_text[max(0, entity['start']-200):entity['end']+200]}"
    
    TASK: Create 2-3 search terms that would help find the correct modern location for mapping this historical/cultural entity.
    
    CONSIDERATIONS:
    - If it's a historical place, what modern location best represents it?
    - If it's an ancient city, where are the ruins/archaeological site?
    - If it's a cultural region, what modern political entity covers it?
    - If it's already modern, keep as-is
    
    EXAMPLES:
    - "Ancient Rome" → ["Rome, Italy", "Roman Forum, Rome"]
    - "Medieval London" → ["London, UK", "City of London"]  
    - "Classical Athens" → ["Athens, Greece", "Ancient Agora, Athens"]
    - "Argos" in ancient Greek context → ["Argos, Greece", "Ancient Argos, Peloponnese"]
    
    Respond with 2-3 search terms in JSON format:
    ["search term 1", "search term 2", "search term 3"]
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            search_terms = self.extract_json_from_response(response.text)
            
            if search_terms and isinstance(search_terms, list):
                return self._geocode_with_terms(entity, search_terms, geographical_context)
            
        except Exception as e:
            print(f"LLM geocoding context failed: {e}")
        
        # Fallback to basic approach
        return self._basic_contextual_geocoding(entity, geographical_context)
    
    def _geocode_with_terms(self, entity, search_terms, geographical_context):
        """Geocode using LLM-generated search terms."""
        for search_term in search_terms:
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
                        entity['geocoding_source'] = 'llm_intelligent_contextual'
                        entity['search_term_used'] = search_term
                        entity['llm_geographical_context'] = geographical_context
                        return True
            
                time.sleep(0.3)
            except Exception:
                continue
        
        return False

    def _extract_facility_context(self, entity, full_text):
        """Use LLM to determine facility type from context."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return ""
            
            # Get context around the entity
            start = max(0, entity['start'] - 100)
            end = min(len(full_text), entity['end'] + 100)
            context = full_text[start:end]
            
            prompt = f"""Analyze this text context to determine what type of facility "{entity['text']}" is.
    
    FACILITY ENTITY: "{entity['text']}"
    CONTEXT: "{context}"
    
    What type of facility is this? Consider:
    - Religious buildings (church, mosque, temple, synagogue, etc.)
    - Performance venues (theatre, concert hall, opera house, etc.) 
    - Educational institutions (school, university, academy, etc.)
    - Healthcare facilities (hospital, clinic, dispensary, etc.)
    - Cultural institutions (museum, library, gallery, etc.)
    - Commercial venues (hotel, restaurant, market, etc.)
    - Sports/recreational facilities (stadium, gym, pool, etc.)
    - Government buildings (courthouse, parliament, city hall, etc.)
    - Any other facility type
    
    Respond with just the facility type (e.g., "theatre", "mosque", "university", "hospital") or "unknown" if unclear.
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            facility_type = response.text.strip().lower()
            
            # Clean up response
            facility_type = facility_type.replace('"', '').replace("'", '')
            
            return facility_type if facility_type != "unknown" else ""
            
        except Exception as e:
            print(f"LLM facility context extraction failed: {e}")
            return ""

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with LLM-enhanced geographical context."""
        if not context_clues:
            return False
        
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._basic_contextual_fallback(entity, context_clues)
            
            # Let LLM create intelligent search variations
            prompt = f"""Create geocoding search terms for this entity using the detected geographical context.
    
    ENTITY: "{entity['text']}" (Type: {entity['type']})
    DETECTED CONTEXT CLUES: {context_clues}
    
    TASK: Generate 2-3 search terms that would help geocode this entity accurately using the geographical context.
    
    CONSIDERATIONS:
    - Use the context clues to disambiguate the location
    - Consider common alternative names/spellings
    - Include country/region qualifiers where helpful
    - Prioritize the most specific and likely to succeed
    
    EXAMPLES:
    - Entity: "Manchester", Context: ["uk", "england"] → ["Manchester, UK", "Manchester, England"]
    - Entity: "Alexandria", Context: ["egypt", "ancient"] → ["Alexandria, Egypt", "Alexandria, Ancient Egypt"]
    - Entity: "Paris", Context: ["france", "european"] → ["Paris, France", "Paris, Île-de-France"]
    
    Respond with a JSON array of 2-3 search terms:
    ["search term 1", "search term 2", "search term 3"]
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            search_terms = self.extract_json_from_response(response.text)
            
            if search_terms and isinstance(search_terms, list):
                return self._geocode_with_search_terms(entity, search_terms, 'llm_contextual')
            
        except Exception as e:
            print(f"LLM contextual geocoding failed: {e}")
        
        # Fallback to basic approach
        return self._basic_contextual_fallback(entity, context_clues)
    
    def _basic_contextual_fallback(self, entity, context_clues):
        """Simple fallback that just appends context without hardcoded mappings."""
        search_variations = [entity['text']]
        
        # Just append context clues directly - let geocoding service handle variations
        for context in context_clues[:2]:  # Limit to top 2 context clues
            search_variations.append(f"{entity['text']}, {context}")
        
        return self._geocode_with_search_terms(entity, search_variations, 'basic_contextual')
    
    def _geocode_with_search_terms(self, entity, search_terms, method):
        """Common geocoding logic using provided search terms."""
        for search_term in search_terms[:3]:
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
                        entity['geocoding_source'] = method
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.3)
            except Exception:
                continue
        
        return False

    def _try_openstreetmap(self, entity, geographical_context=None):
        """Fall back to direct OpenStreetMap Nominatim API with geographical context."""
        try:
            # Create search variations using geographical context
            search_variations = [entity['text']]
            
            if geographical_context:
                # Add the LLM-detected context
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
            
            # Remove duplicates while preserving order
            search_variations = list(dict.fromkeys(search_variations))
            
            # Try each variation
            for search_term in search_variations:
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
                        entity['geocoding_source'] = 'openstreetmap_contextual' if geographical_context else 'openstreetmap'
                        entity['search_term_used'] = search_term
                        if geographical_context:
                            entity['llm_geographical_context'] = geographical_context
                        return True
            
                time.sleep(0.3)  # Rate limiting
        
        except Exception as e:
            pass
        
        return False

    def link_to_wikidata(self, entities):
        """Add context-aware Wikidata linking using LLM intelligence."""
        for entity in entities:
            # Skip if already has Getty AAT link (higher priority)
            if entity.get('getty_aat_url'):
                continue
                
            try:
                # Use LLM to create intelligent Wikidata search queries
                search_queries = self._create_wikidata_search_queries(entity)
                
                # Try each LLM-generated query
                for search_query in search_queries:
                    wikidata_result = self._search_wikidata(search_query, entity)
                    if wikidata_result:
                        entity.update(wikidata_result)
                        break
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Wikidata linking failed for {entity['text']}: {e}")
            
        return entities
    
    def _create_wikidata_search_queries(self, entity):
        """Use LLM to create intelligent Wikidata search queries."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return [entity['text']]  # Fallback to basic search
            
            entity_context = entity.get('context', {})
            
            prompt = f"""Create optimized Wikidata search queries for this entity.
    
    ENTITY: "{entity['text']}" (Type: {entity['type']})
    
    CONTEXT:
    - Period: {entity_context.get('period', 'unknown')}
    - Region: {entity_context.get('region', 'unknown')}
    - Culture: {entity_context.get('culture', 'unknown')}
    - Subject: {entity_context.get('subject_matter', 'unknown')}
    
    TASK: Generate 2-3 search queries that would find the correct Wikidata entry for this entity in this context.
    
    CONSIDERATIONS:
    - Add contextual qualifiers to disambiguate
    - Consider alternative names/spellings
    - Include cultural/temporal context where relevant
    - Make queries specific enough to avoid wrong matches
    
    EXAMPLES:
    - "Argos" in ancient Greek context → ["Argos Greece ancient city", "Argos Peloponnese", "Argos archaeological site"]
    - "Io" in mythology context → ["Io Greek mythology", "Io daughter Inachus", "Io mythological figure"]
    - "Paris" in French context → ["Paris France capital", "Paris city France"]
    - "Mercury" in astronomy context → ["Mercury planet", "Mercury solar system"]
    
    Respond with JSON array of 2-3 search queries:
    ["query 1", "query 2", "query 3"]
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            search_queries = self.extract_json_from_response(response.text)
            
            if search_queries and isinstance(search_queries, list):
                return search_queries
            
        except Exception as e:
            print(f"LLM search query generation failed: {e}")
        
        # Fallback to basic query
        return [entity['text']]
    
    def _search_wikidata(self, search_query, entity):
        """Search Wikidata and use LLM to select best match."""
        try:
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
                results = data.get('search', [])
                
                if not results:
                    return None
                
                if len(results) == 1:
                    # Only one result, use it
                    result = results[0]
                    return {
                        'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                        'wikidata_description': result.get('description', ''),
                        'wikidata_search_query': search_query
                    }
                
                # Multiple results - use LLM to choose best match
                return self._llm_select_wikidata_result(entity, results, search_query)
            
        except Exception as e:
            print(f"Wikidata search failed for '{search_query}': {e}")
        
        return None
    
    def _llm_select_wikidata_result(self, entity, results, search_query):
        """Use LLM to select the best Wikidata result."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Fallback to first result
                result = results[0]
                return {
                    'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                    'wikidata_description': result.get('description', ''),
                    'wikidata_search_query': search_query
                }
            
            entity_context = entity.get('context', {})
            
            candidates_text = []
            for i, result in enumerate(results):
                candidates_text.append(
                    f"{i+1}. {result.get('label', 'No label')}: {result.get('description', 'No description')}"
                )
            
            prompt = f"""Select the best Wikidata match for this entity.
    
    ENTITY: "{entity['text']}" (Type: {entity['type']})
    SEARCH QUERY USED: "{search_query}"
    
    CONTEXT:
    - Period: {entity_context.get('period', 'unknown')}
    - Region: {entity_context.get('region', 'unknown')}
    - Culture: {entity_context.get('culture', 'unknown')}
    
    WIKIDATA CANDIDATES:
    {chr(10).join(candidates_text)}
    
    Which candidate (1-{len(results)}) best matches this entity in this context?
    If none are appropriate, respond with "NONE".
    
    Response format: Just the number (1-{len(results)}) or "NONE"
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse LLM response
            match = re.search(r'(\d+)', result_text)
            if match:
                choice = int(match.group(1)) - 1
                if 0 <= choice < len(results):
                    result = results[choice]
                    return {
                        'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                        'wikidata_description': result.get('description', ''),
                        'wikidata_search_query': search_query,
                        'wikidata_selection_method': 'llm_intelligent'
                    }
            
            # Fallback to first result
            result = results[0]
            return {
                'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                'wikidata_description': result.get('description', ''),
                'wikidata_search_query': search_query,
                'wikidata_selection_method': 'first_result_fallback'
            }
            
        except Exception as e:
            print(f"LLM Wikidata selection failed: {e}")
            # Fallback to first result
            result = results[0]
            return {
                'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                'wikidata_description': result.get('description', ''),
                'wikidata_search_query': search_query,
                'wikidata_selection_method': 'error_fallback'
            }

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Use LLM to intelligently detect geographical context."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._pycountry_fallback(text, entities)
            
            # Extract geographical entities for context
            geo_entities = [e['text'] for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY']]
            
            prompt = f"""Analyze this text to identify geographical context clues for geocoding purposes.
    
    TEXT: "{text[:1500]}..."
    
    IDENTIFIED GEOGRAPHICAL ENTITIES: {geo_entities}
    
    TASK: Extract geographical context that would help disambiguate place names for geocoding.
    
    Consider:
    - Countries, regions, states, provinces mentioned
    - Historical geographical references
    - Cultural/linguistic geographic indicators
    - Alternative names for places (e.g., "Britain" = "UK")
    - Postal codes or address patterns
    - Implicit geographical context from cultural references
    
    Respond with a JSON array of 2-4 geographical context clues, ordered by relevance:
    ["most specific context", "broader context", "alternative context"]
    
    EXAMPLES:
    - Text about "London theatre district" → ["London, UK", "England", "United Kingdom"]
    - Text with "ZIP code 90210" → ["California, USA", "United States", "Beverly Hills"]
    - Text about "ancient Athens" → ["Greece", "Ancient Greece", "Mediterranean"]
    - Text mentioning "Sydney Opera House" → ["Sydney, Australia", "Australia", "New South Wales"]
    
    Focus on context that would help geocoding services find the right places.
    """
    
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            context_clues = self.extract_json_from_response(response.text)
            
            if context_clues and isinstance(context_clues, list):
                return context_clues[:3]  # Limit to top 3
            
        except Exception as e:
            print(f"LLM geographical context detection failed: {e}")
        
        # Fallback to pycountry approach
        return self._pycountry_fallback(text, entities)
    
    def _pycountry_fallback(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Fallback using pycountry when LLM is unavailable."""
        try:
            import pycountry
            
            context_clues = []
            text_lower = text.lower()
            
            # Extract from entities
            geo_entities = [e['text'].lower() for e in entities 
                           if e['type'] in ['GPE', 'LOCATION', 'FACILITY']]
            
            # Check against pycountry
            for country in pycountry.countries:
                if country.name.lower() in text_lower:
                    context_clues.append(country.name)
            
            # Add geo entities
            context_clues.extend(geo_entities)
            
            return list(dict.fromkeys(context_clues))[:3]
            
        except ImportError:
            # Ultimate fallback - just use entities
            return [e['text'] for e in entities if e['type'] in ['GPE', 'LOCATION']][:3]

    def link_to_getty_aat(self, entities):
        """Add Getty Art & Architecture Thesaurus linking - especially good for PRODUCT, FACILITY, WORK_OF_ART entities."""
        for entity in entities:
            # Getty AAT is particularly valuable for these entity types
            relevant_types = ['PRODUCT', 'FACILITY', 'WORK_OF_ART', 'EVENT', 'ORGANIZATION']
            if entity['type'] not in relevant_types:
                continue
                
            try:
                # Try multiple approaches to find Getty AAT entries
                getty_result = None
                
                # Method 1: Direct search via Getty's search API (more reliable than SPARQL)
                getty_result = self._search_getty_api(entity['text'])
                
                if not getty_result:
                    # Method 2: Try variations of the search term
                    variations = self._create_getty_search_variations(entity['text'], entity['type'])
                    for variation in variations:
                        getty_result = self._search_getty_api(variation)
                        if getty_result:
                            break
                
                if getty_result:
                    entity['getty_aat_url'] = getty_result['url']
                    entity['getty_aat_label'] = getty_result['label']
                    entity['getty_aat_description'] = getty_result['description']
                    entity['getty_search_term'] = getty_result['search_term']
                
                time.sleep(0.5)  # More conservative rate limiting for Getty
                
            except Exception as e:
                print(f"Getty AAT search failed for {entity['text']}: {e}")
                # Continue silently - Getty can be unreliable
                pass
        
        return entities

    def _search_getty_api(self, search_term):
        """Search Getty AAT using their REST API instead of SPARQL."""
        try:
            # Getty's Vocabulary Program API
            search_url = "http://vocab.getty.edu/sparql.json"
            
            # Simplified SPARQL query that's more likely to work
            query = f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX gvp: <http://vocab.getty.edu/ontology#>
            PREFIX xl: <http://www.w3.org/2008/05/skos-xl#>
            
            SELECT ?subject ?term ?note WHERE {{
              ?subject skos:inScheme <http://vocab.getty.edu/aat/> ;
                       gvp:prefLabelGVP/xl:literalForm ?term ;
                       skos:scopeNote ?noteObj .
              ?noteObj rdf:value ?note .
              FILTER(CONTAINS(LCASE(?term), LCASE("{search_term}")))
            }}
            LIMIT 5
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'EntityLinker/1.0'
            }
            
            response = requests.get(search_url, 
                                  params={'query': query}, 
                                  headers=headers, 
                                  timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    result = results[0]  # Take first result
                    subject_uri = result['subject']['value']
                    term = result['term']['value']
                    note = result.get('note', {}).get('value', '')
                    
                    # Convert URI to web URL
                    aat_id = subject_uri.split('/')[-1]
                    web_url = f"http://www.getty.edu/vow/AATFullDisplay?find=&logic=AND&note=&page=1&subjectid={aat_id}"
                    
                    return {
                        'url': web_url,
                        'label': term,
                        'description': note[:200] if note else f"Getty AAT entry for {term}",
                        'search_term': search_term
                    }
            
            # Fallback: Try alternative Getty search approach
            return self._search_getty_alternative(search_term)
            
        except Exception as e:
            print(f"Getty SPARQL search failed for '{search_term}': {e}")
            return self._search_getty_alternative(search_term)

    def _search_getty_alternative(self, search_term):
        """Alternative Getty search using their web interface."""
        try:
            # Search Getty's web interface directly
            search_url = "http://www.getty.edu/vow/AATServlet"
            params = {
                'english': 'N',
                'find': search_term,
                'logic': 'AND',
                'note': '',
                'page': '1'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Look for AAT entries in the response
                import re
                # Find AAT IDs in the response
                aat_pattern = r'AATFullDisplay\?[^"]*subjectid=(\d+)[^"]*"[^>]*>([^<]+)</a>'
                matches = re.findall(aat_pattern, response.text)
                
                if matches:
                    aat_id, label = matches[0]
                    web_url = f"http://www.getty.edu/vow/AATFullDisplay?find=&logic=AND&note=&page=1&subjectid={aat_id}"
                    
                    return {
                        'url': web_url,
                        'label': label.strip(),
                        'description': f"Getty AAT architectural/cultural term: {label.strip()}",
                        'search_term': search_term
                    }
            
        except Exception as e:
            print(f"Getty alternative search failed for '{search_term}': {e}")
        
        return None

    def _create_getty_search_variations(self, text, entity_type):
        """Create search variations optimized for Getty AAT."""
        variations = [text]
        text_lower = text.lower()
        
        # Add plurals/singulars
        if text_lower.endswith('s') and len(text) > 3:
            variations.append(text[:-1])  # Remove 's'
        elif not text_lower.endswith('s'):
            variations.append(text + 's')  # Add 's'
        
        # Add architectural/cultural terms for different entity types
        if entity_type == 'FACILITY':
            facility_terms = ['building', 'structure', 'architecture']
            variations.extend([f"{text} {term}" for term in facility_terms])
            
        elif entity_type == 'PRODUCT':
            product_terms = ['architectural element', 'component', 'feature']
            variations.extend([f"{text} {term}" for term in product_terms])
            
        elif entity_type == 'WORK_OF_ART':
            art_terms = ['art', 'artwork', 'creative work']
            variations.extend([f"{text} {term}" for term in art_terms])
        
        # Remove duplicates
        return list(dict.fromkeys(variations))[:5]  # Limit to 5 variations

    def link_to_britannica(self, entities):
        """Add basic Britannica linking - only for entities without Getty AAT or Wikidata links.""" 
        for entity in entities:
            # Skip if already has Getty AAT or Wikidata link
            if entity.get('getty_aat_url') or entity.get('wikidata_url'):
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
        """Add OpenStreetMap links to addresses with geographical context."""
        for entity in entities:
            # Only process ADDRESS entities
            if entity['type'] != 'ADDRESS':
                continue
                
            try:
                # Use geographical context if available
                geographical_context = entity.get('llm_geographical_context', '')
                
                # Create context-aware search terms
                search_variations = [entity['text']]
                if geographical_context:
                    search_variations.append(f"{entity['text']}, {geographical_context}")
                
                # Remove duplicates
                search_variations = list(dict.fromkeys(search_variations))
                
                # Try each variation
                for search_term in search_variations:
                    # Search OpenStreetMap Nominatim for the address
                    url = "https://nominatim.openstreetmap.org/search"
                    params = {
                        'q': search_term,
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
                            entity['search_term_used'] = search_term
                            if geographical_context:
                                entity['llm_geographical_context'] = geographical_context
                            break  # Success, stop trying variations
                    
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
        st.sidebar.info("Entities are linked with priority hierarchy: 1) Getty AAT (authoritative art & architecture terminology), 2) Wikidata (structured knowledge), 3) Britannica (scholarly encyclopedia), 4) Wikipedia (general reference). The LLM provides intelligent disambiguation for the final step.")
        
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
                
                # Step 3: Link to Getty AAT - FIRST PRIORITY for cultural/architectural terms
                status_text.text("Linking to Getty Art & Architecture Thesaurus...")
                progress_bar.progress(40)
                entities = self.entity_linker.link_to_getty_aat(entities)
                
                # Step 4: Link to Wikidata - SECOND PRIORITY for general structured data
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
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
                if entity.get('getty_aat_description'):
                    desc = entity['getty_aat_description'][:100]
                    tooltip_parts.append(f"Getty AAT: {desc}")
                elif entity.get('wikidata_description'):
                    desc = entity['wikidata_description'][:100]  # Limit description length
                    tooltip_parts.append(f"Description: {desc}")
                elif entity.get('wikipedia_description'):
                    desc = entity['wikipedia_description'][:100]
                    tooltip_parts.append(f"Description: {desc}")
                if entity.get('location_name'):
                    loc = entity['location_name'][:100]  # Limit location length
                    tooltip_parts.append(f"Location: {loc}")
                if entity.get('disambiguation_method') == 'llm_contextual':
                    tooltip_parts.append("LLM disambiguated")
                
                tooltip = html_module.escape(" | ".join(tooltip_parts))
                
                url = (entity.get('getty_aat_url') or
                      entity.get('wikidata_url') or
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
            
            if entity.get('getty_aat_description'):
                row['Description'] = entity['getty_aat_description']
            elif entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
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
        if entity.get('getty_aat_url'):
            links.append("Getty AAT")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
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
                
                # Add Getty AAT link first
                if entity.get('getty_aat_url'):
                    entity_data['sameAs'] = entity['getty_aat_url']
                
                # Add Wikidata link
                if entity.get('wikidata_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['wikidata_url']]
                        else:
                            entity_data['sameAs'].append(entity['wikidata_url'])
                    else:
                        entity_data['sameAs'] = entity['wikidata_url']
                
                if entity.get('getty_aat_description'):
                    entity_data['description'] = entity['getty_aat_description']
                elif entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
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
