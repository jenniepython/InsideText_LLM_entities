#!/usr/bin/env python3
"""
Streamlit LLM Entity Linker Application - Fixed Version with Proper Disambiguation

FIXED:
1. Removed premature deduplication - each occurrence processed individually
2. Generic context-based disambiguation (no hardcoded entity names)
3. Proper case sensitivity handling
4. Algorithmic similarity scoring between context and candidates

Author: Enhanced from NER_LLM_entities_streamlit.py
Version: 2.2 (Fixed Disambiguation)
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
    Main class for LLM-based entity linking with PROPER generic disambiguation.
    
    FIXED:
    - No premature deduplication
    - Generic context-based disambiguation 
    - Proper case sensitivity handling
    - Algorithmic similarity scoring
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
        """Extract context window around an entity for better disambiguation."""
        context_start = max(0, entity_start - window_size)
        context_end = min(len(text), entity_end + window_size)
        
        before_text = text[context_start:entity_start].strip()
        after_text = text[entity_end:context_end].strip()
        entity_text = text[entity_start:entity_end]
        
        stop_words = self._get_stop_words()
        
        before_words = [w.strip('.,;:!?"()[]{}') for w in before_text.split() if w.lower().strip('.,;:!?"()[]{}') not in stop_words and len(w) > 2]
        after_words = [w.strip('.,;:!?"()[]{}') for w in after_text.split() if w.lower().strip('.,;:!?"()[]{}') not in stop_words and len(w) > 2]
        
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
        """Construct a simplified NER prompt that works consistently."""
        prompt = f"""Extract named entities from the following text. Only extract proper nouns and named things.

ENTITY TYPES:
- PERSON: Named individuals (e.g., "John Smith", "Caesar")
- ORGANIZATION: Named groups, companies, civilizations (e.g., "Apple Inc", "the Phoenicians") 
- GPE: Countries, cities, regions (e.g., "France", "London", "ancient Egypt")
- LOCATION: Geographic features (e.g., "Red Sea", "Mount Everest")
- FACILITY: Named buildings, venues (e.g., "Empire State Building")
- PRODUCT: Named objects, brands (e.g., "iPhone", "merchandise" only if specifically named)
- EVENT: Named events (e.g., "World War II", "the Olympics")
- WORK_OF_ART: Titles of books, movies, songs, etc.
- DATE: Specific dates, years, periods
- MONEY: Specific amounts with currency

RULES:
1. Only extract proper nouns - things with specific names
2. Don't extract adjectives or descriptive words (e.g., skip "Egyptian" in "Egyptian goods")
3. Don't extract common job titles or roles unless they're part of a proper name
4. Don't extract generic terms like "king", "merchants", "women" unless they're part of a proper name
5. IMPORTANT: Only list each unique entity ONCE in your response, even if it appears multiple times in the text

Now extract entities from this text:

"{text}"

Output only a JSON array with the entities found:"""
        
        return prompt

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response with improved parsing."""
        response_text = response_text.strip()
        
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
        """FIXED: Extract entities with proper individual occurrence processing."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found!")
                return []
            
            context = self.analyse_text_context(text)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = self.construct_ner_prompt(text, context)
            gemini_response = model.generate_content(prompt)
            llm_response = gemini_response.text
            
            entities_raw = self.extract_json_from_response(llm_response)
            if not entities_raw:
                st.warning("Could not parse JSON from Gemini response.")
                return []
            
            # FIXED: NO premature deduplication - process each occurrence individually
            entities = []
            
            for entity_raw in entities_raw:
                if 'text' in entity_raw and 'type' in entity_raw:
                    entity_text = entity_raw['text'].strip()
                    entity_type = entity_raw['type']
                    
                    # Find ALL occurrences, preserving case sensitivity
                    pattern = r'\b' + re.escape(entity_text) + r'\b'
                    
                    # Try exact case match first, then case-insensitive as fallback
                    matches = list(re.finditer(pattern, text))
                    if not matches:
                        matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    
                    if matches:
                        # Create separate entity for EACH occurrence with its own context
                        for match_idx, match in enumerate(matches):
                            context_window = self.extract_context_window(
                                text, match.start(), match.end()
                            )
                            
                            entity = {
                                'text': match.group(),  # Preserves original case from text
                                'type': entity_type,
                                'start': match.start(),
                                'end': match.end(),
                                'context': context,
                                'context_window': context_window,
                                'occurrence_id': f"{entity_text}_{match.start()}_{entity_type}_{match_idx}"
                            }
                            entities.append(entity)
                    else:
                        # Fallback: try case-sensitive exact search
                        start_pos = 0
                        match_idx = 0
                        while True:
                            pos = text.find(entity_text, start_pos)
                            if pos == -1:
                                break
                            
                            context_window = self.extract_context_window(
                                text, pos, pos + len(entity_text)
                            )
                            
                            entity = {
                                'text': entity_text,
                                'type': entity_type,
                                'start': pos,
                                'end': pos + len(entity_text),
                                'context': context,
                                'context_window': context_window,
                                'occurrence_id': f"{entity_text}_{pos}_{entity_type}_{match_idx}"
                            }
                            entities.append(entity)
                            
                            start_pos = pos + len(entity_text)
                            match_idx += 1
            
            return entities
            
        except Exception as e:
            st.error(f"Error in LLM entity extraction: {e}")
            return []

    def analyse_text_context(self, text: str) -> Dict[str, Any]:
        """Analyse text to determine historical, geographical, and cultural context."""
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
        
        if any(indicator in text_lower for indicator in ['europe', 'european', 'britain', 'france', 'germany', 'italy', 'spain']):
            context['region'] = 'european'
        elif any(indicator in text_lower for indicator in ['mediterranean', 'greece', 'greek', 'rome', 'roman', 'egypt', 'phoenician']):
            context['region'] = 'mediterranean'
            if any(indicator in text_lower for indicator in ['greece', 'greek', 'hellas', 'athens', 'sparta']):
                context['culture'] = 'greek'
            elif any(indicator in text_lower for indicator in ['rome', 'roman', 'latin', 'caesar']):
                context['culture'] = 'roman'
        elif any(indicator in text_lower for indicator in ['asia', 'china', 'japan', 'india', 'persia', 'persian']):
            context['region'] = 'asian'
        
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
        
        if any(pattern in text_lower for pattern in ['thee', 'thou', 'thy', 'hath', 'doth', 'forsooth', 'wherefore']):
            context['language_style'] = 'archaic'
        elif any(pattern in text_lower for pattern in ['it is said', 'according to', 'the learned men', 'historians say']):
            context['language_style'] = 'historical_narrative'
        elif any(pattern in text_lower for pattern in ['recording', 'survey', 'analysis', 'study', 'research']):
            context['language_style'] = 'academic'
        
        return context

    def _get_stop_words(self):
        """Generic stop words list for context analysis."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
            'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
            'us', 'them', 'my', 'your', 'his', 'our', 'their', 'some', 'any', 'all', 
            'each', 'every', 'no', 'not', 'only', 'just', 'about', 'up', 'out', 'so', 
            'if', 'what', 'who', 'when', 'where', 'why', 'how', 'then', 'than', 'now', 
            'here', 'there', 'back', 'away', 'said', 'says', 'very', 'more', 'most'
        }

    def _build_generic_context_queries(self, entity, context_window):
        """Build search queries using generic contextual patterns - TRULY GENERIC."""
        entity_type = entity.get('type', '')
        entity_text = entity['text']
        queries = []
        
        # Extract context words
        context_words = []
        if context_window.get('before'):
            words = [w.strip('.,;:!?"()[]{}') for w in context_window['before'].split()]
            context_words.extend([w for w in words if len(w) > 2 and 
                                w.lower() not in self._get_stop_words()][-3:])
        
        if context_window.get('after'):
            words = [w.strip('.,;:!?"()[]{}') for w in context_window['after'].split()]
            context_words.extend([w for w in words if len(w) > 2 and 
                                w.lower() not in self._get_stop_words()][:3])
        
        # Strategy: Always lead with context-specific queries (most disambiguating)
        for word in context_words:
            queries.append(f"{entity_text} {word}")
        
        # Always add type-specific queries (strong disambiguation)
        if entity_type == 'PERSON':
            queries.extend([
                f"{entity_text} person",
                f"{entity_text} biography", 
                f"{entity_text} individual"
            ])
        elif entity_type == 'GPE':
            queries.extend([
                f"{entity_text} place location",
                f"{entity_text} city",
                f"{entity_text} country"
            ])
        elif entity_type == 'ORGANIZATION':
            queries.extend([
                f"{entity_text} organization",
                f"{entity_text} group company"
            ])
        elif entity_type in ['LOCATION', 'FACILITY']:
            queries.extend([
                f"{entity_text} location",
                f"{entity_text} place"
            ])
        elif entity_type == 'WORK_OF_ART':
            queries.extend([
                f"{entity_text} work art",
                f"{entity_text} book play film"
            ])
        
        # Generic ambiguity detection: only add bare name if we have strong context
        # This is algorithmic - no hardcoded names
        has_strong_context = len(context_words) >= 2  # At least 2 meaningful context words
        entity_is_common_word = len(entity_text.split()) == 1 and entity_text.islower()  # Single lowercase word
        
        if has_strong_context or not entity_is_common_word:
            # Either we have good context OR it's a proper noun - safe to try bare name
            queries.append(entity_text)
        # Otherwise skip bare name search to avoid ambiguous results
        
        # Additional context hints
        if entity_text.istitle() and entity_type == 'PERSON':
            queries.append(f"{entity_text} name")
        elif entity_text.islower():
            queries.append(f"{entity_text} common noun")
        
        return list(dict.fromkeys(queries))

    def _calculate_context_similarity_score(self, candidate, entity, context_window):
        """Generic algorithm to score candidate relevance based on context with STRONG type enforcement."""
        score = 0
        description = candidate.get('description', '').lower()
        label = candidate.get('label', '').lower()
        entity_type = entity.get('type', '')
        
        # Exact label match bonus
        if label == entity['text'].lower():
            score += 15
        
        # Context word matching
        context_words = []
        if context_window.get('before'):
            context_words.extend(context_window['before'].lower().split())
        if context_window.get('after'):
            context_words.extend(context_window['after'].lower().split())
        
        meaningful_words = [w for w in context_words if len(w) > 3 and 
                           w not in self._get_stop_words()]
        
        for word in meaningful_words:
            if word in description or word in label:
                score += 3
        
        # STRONGER type enforcement - hard rejections for wrong types
        if entity_type == 'PERSON':
            # Strong positive signals for person
            if any(term in description for term in ['person', 'human', 'individual', 'people', 'born', 'died', 'biography', 'life', 'career', 'actor', 'writer', 'politician', 'scientist', 'artist', 'musician', 'athlete', 'student', 'teacher', 'professor', 'doctor', 'engineer']):
                score += 15
            
            # HARD REJECTION for geographical/political entities
            geographical_terms = ['country', 'nation', 'state', 'city', 'kingdom', 'republic', 'province', 'region', 'territory', 'capital', 'government', 'border', 'located', 'area', 'population', 'geography', 'continent', 'island', 'peninsula', 'landlocked']
            if any(term in description for term in geographical_terms):
                return -100  # Hard rejection
                
        elif entity_type == 'GPE':
            # Strong positive for geographical/political
            if any(term in description for term in ['country', 'city', 'state', 'nation', 'place', 'capital', 'located', 'government', 'region', 'territory', 'province', 'municipality', 'district', 'area']):
                score += 15
            
            # Hard rejection for persons
            person_terms = ['person', 'individual', 'human', 'born', 'died', 'biography', 'life', 'career', 'actor', 'writer', 'politician', 'scientist', 'artist']
            if any(term in description for term in person_terms):
                return -100
                
        elif entity_type == 'ORGANIZATION':
            if any(term in description for term in ['organization', 'company', 'group', 'institution', 'association', 'corporation', 'foundation', 'society', 'club', 'team', 'business', 'firm']):
                score += 15
            
            # Reject individual persons or places
            if any(term in description for term in ['person', 'individual', 'born', 'died']) or any(term in description for term in ['country', 'city', 'located in']):
                return -100
                
        elif entity_type in ['LOCATION', 'FACILITY']:
            if any(term in description for term in ['location', 'place', 'building', 'structure', 'site', 'facility', 'venue', 'landmark', 'monument', 'museum', 'theater', 'stadium', 'hospital', 'school', 'university', 'church', 'temple']):
                score += 15
            
            # Reject persons or abstract concepts
            if any(term in description for term in ['person', 'individual', 'born', 'died']):
                return -100
                
        elif entity_type == 'WORK_OF_ART':
            if any(term in description for term in ['work', 'art', 'book', 'play', 'film', 'movie', 'song', 'album', 'painting', 'novel', 'poem', 'story', 'drama', 'comedy', 'musical', 'opera', 'ballet', 'sculpture', 'artwork']):
                score += 15
            
            # Reject persons or places (unless they're about the work)
            if any(term in description for term in ['person', 'individual', 'born', 'died']) and not any(term in description for term in ['character', 'protagonist', 'story', 'novel', 'book']):
                score -= 15  # Softer rejection as works can be about people
        
        # Capitalization patterns
        if entity['text'].istitle() and not entity['text'].islower():
            if any(term in description for term in ['specific', 'named', 'particular']):
                score += 5
            if any(term in description for term in ['generic', 'common', 'type of']):
                score -= 3
        elif entity['text'].islower():
            if any(term in description for term in ['type of', 'kind of', 'category']):
                score += 5
            if any(term in description for term in ['specific', 'particular instance']):
                score -= 3
        
        # Always reject these categories
        wrong_categories = ['disambiguation page', 'list of', 'category:', 'template:', 'wikimedia', 'redirect']
        for wrong in wrong_categories:
            if wrong in description or wrong in label:
                return -100  # Hard rejection
        
        return score

    def link_to_wikidata(self, entities):
        """Generic context-based linking for each individual occurrence."""
        for entity in entities:
            try:
                context_window = entity.get('context_window', {})
                search_queries = self._build_generic_context_queries(entity, context_window)
                
                best_match = None
                best_score = 0
                
                for search_query in search_queries[:5]:
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
                        if data.get('search'):
                            for candidate in data['search']:
                                score = self._calculate_context_similarity_score(
                                    candidate, entity, context_window
                                )
                                if score > best_score:
                                    best_score = score
                                    best_match = candidate
                                    best_match['search_query_used'] = search_query
                    
                    time.sleep(0.1)
                
                if best_match and best_score > 10:
                    entity['wikidata_url'] = f"http://www.wikidata.org/entity/{best_match['id']}"
                    entity['wikidata_description'] = best_match.get('description', '')
                    entity['search_query_used'] = best_match.get('search_query_used', '')
                    entity['confidence_score'] = best_score
                
            except Exception:
                pass
        
        return entities

    def link_to_wikipedia_contextual(self, entities, text_context):
        """Add contextual Wikipedia linking using surrounding words for better disambiguation."""
        for entity in entities:
            if entity.get('wikidata_url'):
                continue
                
            try:
                entity_context = entity.get('context', text_context)
                context_window = entity.get('context_window', {})
                
                search_terms = [entity['text']]
                
                contextual_keywords = []
                if context_window.get('before'):
                    words = [w for w in context_window['before'].split() if len(w) > 2]
                    contextual_keywords.extend(words[-2:])
                if context_window.get('after'):
                    words = [w for w in context_window['after'].split() if len(w) > 2]
                    contextual_keywords.extend(words[:2])
                
                for keyword in contextual_keywords:
                    keyword_clean = keyword.strip('.,;:!?"()[]{}').lower()
                    if keyword_clean not in ['the', 'and', 'with', 'from', 'near', 'this', 'that', 'was', 'were', 'are']:
                        search_terms.append(f"{entity['text']} {keyword}")
                
                context_modifiers = []
                if entity_context.get('time_indicators'):
                    context_modifiers.extend(entity_context['time_indicators'])
                if entity_context.get('subject_indicators'):
                    context_modifiers.extend(entity_context['subject_indicators'])
                if entity_context.get('place_indicators'):
                    context_modifiers.extend(entity_context['place_indicators'])
                
                if entity_context.get('period') and entity_context['period'] != 'modern':
                    context_modifiers.extend(['historical', entity_context['period']])
                
                if entity['type'] in ['GPE', 'LOCATION'] and context_modifiers:
                    for modifier in context_modifiers[:2]:
                        search_terms.append(f"{entity['text']} {modifier}")
                        if modifier != 'historical':
                            search_terms.append(f"{modifier} {entity['text']}")
                
                elif entity['type'] == 'PERSON' and context_modifiers:
                    for modifier in context_modifiers[:2]:
                        search_terms.append(f"{entity['text']} {modifier}")
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
                
                search_terms = list(dict.fromkeys(search_terms))[:5]
                
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
                            
                            snippet = result.get('snippet', '').lower()
                            title_lower = page_title.lower()
                            
                            skip_terms = ['video game', 'software', 'app', 'company', 'corporation', 'brand']
                            if entity_context.get('period') and entity_context['period'] != 'modern':
                                if any(term in snippet or term in title_lower for term in skip_terms):
                                    continue
                            
                            relevant = False
                            
                            if contextual_keywords:
                                for keyword in contextual_keywords:
                                    if keyword.lower() in snippet or keyword.lower() in title_lower:
                                        relevant = True
                                        break
                            
                            if not relevant and search_term == entity['text']:
                                relevant = True
                            
                            if relevant:
                                encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                                entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                                entity['wikipedia_title'] = page_title
                                
                                if result.get('snippet'):
                                    snippet_clean = re.sub(r'<[^>]+>', '', result['snippet'])
                                    entity['wikipedia_description'] = snippet_clean[:200] + "..." if len(snippet_clean) > 200 else snippet_clean
                                
                                entity['search_context'] = search_term
                                break
                    
                    time.sleep(0.2)
                
            except Exception as e:
                pass
        
        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        for entity in entities:
            if entity.get('wikidata_url'):
                continue
                
            try:
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
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                        entity['wikipedia_title'] = page_title
                        
                        if result.get('snippet'):
                            snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                            entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                
                time.sleep(0.2)
            except Exception as e:
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add basic Britannica linking with contextual search.""" 
        for entity in entities:
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_terms = [entity['text']]
                
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
                
                for search_term in search_terms[:3]:
                    search_url = "https://www.britannica.com/search"
                    params = {'query': search_term}
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(search_url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
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
                            break
                
                    time.sleep(0.3)
            except Exception:
                pass
        
        return entities

    def get_coordinates(self, entities, processed_text=""):
        """Enhanced coordinate lookup ONLY for place entities."""
        place_entities = [e for e in entities if e['type'] in self.geocodable_types]
        
        if not place_entities:
            st.info("No place entities found for geocoding.")
            return entities
        
        context_clues = self._detect_geographical_context(processed_text, place_entities)
        
        geocoded_count = 0
        for entity in place_entities:
            if entity.get('latitude') is not None:
                continue
            
            if self._try_contextual_geocoding(entity, context_clues):
                geocoded_count += 1
                continue
                
            if self._try_openstreetmap(entity):
                geocoded_count += 1
                continue
        
        st.info(f"Geocoded {geocoded_count}/{len(place_entities)} place entities")
        return entities

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text."""
        context_clues = []
        text_lower = text.lower()
        
        geographical_entities = []
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION', 'FACILITY']:
                geographical_entities.append(entity['text'].lower())
        
        all_countries = self._get_all_countries()
        
        for country in all_countries:
            if country in text_lower:
                context_clues.append(country)
                if len(context_clues) >= 10:
                    break
        
        for geo_entity in geographical_entities:
            if geo_entity not in context_clues:
                context_clues.append(geo_entity)
        
        return context_clues[:5]

    def _get_all_countries(self) -> List[str]:
        """Get comprehensive list of all countries."""
        if not PYCOUNTRY_AVAILABLE:
            return ['usa', 'united states', 'uk', 'united kingdom', 'france', 'germany', 'china', 'japan', 'india', 'australia', 'canada', 'brazil', 'russia']
        
        countries = []
        for country in pycountry.countries:
            countries.append(country.name.lower())
            if hasattr(country, 'common_name') and country.common_name:
                countries.append(country.common_name.lower())
            countries.append(country.alpha_2.lower())
            countries.append(country.alpha_3.lower())
        
        return list(set(countries))

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context."""
        if not context_clues:
            return False
        
        search_variations = [entity['text']]
        
        context_window = entity.get('context_window', {})
        if context_window.get('before') or context_window.get('after'):
            contextual_keywords = []
            if context_window.get('before'):
                contextual_keywords.extend(context_window['before'].split()[-2:])
            if context_window.get('after'):
                contextual_keywords.extend(context_window['after'].split()[:2])
            
            for keyword in contextual_keywords:
                if len(keyword) > 2 and keyword.lower() not in ['the', 'and', 'with', 'from', 'near']:
                    search_variations.append(f"{entity['text']} {keyword}")
                    search_variations.append(f"{keyword} {entity['text']}")
        
        for context in context_clues:
            search_variations.append(f"{entity['text']}, {context}")
        
        search_variations = list(dict.fromkeys(search_variations))
        
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
            
                time.sleep(0.3)
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
        
            time.sleep(0.3)
        except Exception as e:
            pass
        
        return False

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses ONLY."""
        for entity in entities:
            if entity['type'] != 'ADDRESS':
                continue
                
            try:
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
                        lat = result['lat']
                        lon = result['lon']
                        entity['openstreetmap_url'] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
                        entity['openstreetmap_display_name'] = result['display_name']
                        
                        entity['latitude'] = float(lat)
                        entity['longitude'] = float(lon)
                        entity['location_name'] = result['display_name']
                
                time.sleep(0.2)
            except Exception:
                pass
        
        return entities

    def get_disambiguation_report(self, entities):
        """Generate generic disambiguation report."""
        report = {}
        
        grouped = {}
        for entity in entities:
            key = entity['text'].lower()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entity)
        
        for text_key, occurrences in grouped.items():
            if len(occurrences) > 1 or any(occ.get('confidence_score', 0) < 15 for occ in occurrences):
                report[text_key] = []
                for occ in occurrences:
                    report[text_key].append({
                        'original_text': occ['text'],
                        'position': f"{occ['start']}-{occ['end']}",
                        'context_snippet': occ.get('context_window', {}).get('context_snippet', '')[:100],
                        'linked_to': occ.get('wikidata_description', 'Not linked'),
                        'confidence': occ.get('confidence_score', 0),
                        'search_used': occ.get('search_query_used', 'Basic')
                    })
        
        return report

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities for display."""
        colors = self.colors
        
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


class StreamlitLLMEntityLinker:
    """Streamlit wrapper for the fixed LLM Entity Linker class."""
    
    def __init__(self):
        """Initialize the Streamlit LLM Entity Linker."""
        self.entity_linker = LLMEntityLinker()
        
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
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("💡 Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.header("From Text to Linked Data using LLM")
        st.markdown("**Extract and link named entities from text using Gemini LLM with proper disambiguation**")
        
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>FIXED: Individual Occurrence Processing</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Generic Context-Based Disambiguation:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata</strong><br><small>Contextual similarity scoring</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikipedia/Britannica</strong><br><small>Context-aware search</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Global Geocoding</strong><br><small>Places only</small>
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
        """Render the sidebar with enhanced information."""
        st.sidebar.subheader("FIXED: Proper Disambiguation")
        st.sidebar.success("✅ Each occurrence processed individually\n✅ Generic context-based linking\n✅ Case sensitivity preserved\n✅ No hardcoded entity rules")
        
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
        
        text_input = st.text_area(
            "Enter your text here:",
            value="",
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
        """Process the input text using the fixed LLM EntityLinker."""
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text with FIXED disambiguation..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Analyzing text context...")
                progress_bar.progress(10)
                text_context = self.entity_linker.analyse_text_context(text)
                
                status_text.text("FIXED: Extracting entities individually...")
                progress_bar.progress(25)
                entities_json = self.cached_extract_entities(text)
                entities = json.loads(entities_json)
                
                if not entities:
                    st.warning("No entities found in the text.")
                    return
                
                place_entities = [e for e in entities if e['type'] in self.entity_linker.geocodable_types]
                
                status_text.text("FIXED: Context-based Wikidata linking...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                status_text.text("FIXED: Individual Wikipedia linking...")
                progress_bar.progress(60)
                entities = self.entity_linker.link_to_wikipedia_contextual(entities, text_context)
                
                status_text.text("FIXED: Contextual Britannica linking...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                if place_entities:
                    status_text.text(f"Geocoding {len(place_entities)} place entities...")
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
                
                status_text.text("Linking addresses to OpenStreetMap...")
                progress_bar.progress(90)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.entity_linker.create_highlighted_html(text, entities)
                
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                st.session_state.text_context = text_context
                
                progress_bar.empty()
                status_text.empty()
                
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
                
                success_message = f"FIXED processing complete! Found {len(linked_entities)} properly disambiguated entities"
                if total_places > 0:
                    success_message += f" ({geocoded_places}/{total_places} places geocoded)"
                
                unlinked_count = len(entities) - len(linked_entities)
                if unlinked_count > 0:
                    success_message += f" ({unlinked_count} entities found but not linked)"
                
                st.success(success_message)
                
                # Show disambiguation report if there are ambiguous entities
                disambiguation_report = self.entity_linker.get_disambiguation_report(entities)
                if disambiguation_report:
                    st.info(f"Disambiguation applied to {len(disambiguation_report)} ambiguous entities")
                    with st.expander("View Disambiguation Decisions"):
                        for entity_text, decisions in disambiguation_report.items():
                            st.write(f"**{entity_text}** ({len(decisions)} occurrences):")
                            for i, decision in enumerate(decisions):
                                st.write(f"  {i+1}. Position {decision['position']}: \"{decision['context_snippet']}\" → {decision['linked_to']} (confidence: {decision['confidence']})")
                
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

    def render_results(self):
        """Render the results section."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        st.subheader("Highlighted Text with Proper Disambiguation")
        if st.session_state.html_content:
            st.markdown(st.session_state.html_content, unsafe_allow_html=True)
        else:
            st.info("No highlighted text available. Process some text first.")
        
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details."""
        if not entities:
            st.info("No entities found.")
            return
        
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
        
        table_data = []
        for entity in sorted_entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Position': f"{entity['start']}-{entity['end']}",
                'Context': entity.get('context_window', {}).get('context_snippet', '')[:100] + "..." if entity.get('context_window', {}).get('context_snippet', '') else 'N/A',
                'Links': self.format_entity_links(entity),
                'Occurrence ID': entity.get('occurrence_id', 'N/A')
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
            
            if entity.get('confidence_score'):
                row['Confidence'] = entity['confidence_score']
            
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
        """Render export options for the results."""
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "entities": [],
                "processingInfo": {
                    "entityExtraction": "FIXED: Gemini LLM with proper individual occurrence processing",
                    "disambiguation": "FIXED: Generic context-based disambiguation (no hardcoded rules)",
                    "geocodingMethod": "Global coverage (places only)",
                    "linkingStrategy": "FIXED: Contextual similarity scoring using surrounding words",
                    "globalCoverage": PYCOUNTRY_AVAILABLE,
                    "version": "2.2_Fixed"
                }
            }
            
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end'],
                    "occurrenceId": entity.get('occurrence_id', '')
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
                
                if entity.get('confidence_score'):
                    entity_data['confidenceScore'] = entity['confidence_score']
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download FIXED JSON-LD",
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
    <title>FIXED Entity Analysis with Proper Disambiguation</title>
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
        <h3>FIXED Processing Information</h3>
        <p><strong>Method:</strong> FIXED - Individual occurrence processing with proper disambiguation</p>
        <p><strong>Disambiguation:</strong> FIXED - Generic context-based (no hardcoded entity rules)</p>
        <p><strong>Geocoding:</strong> Global coverage (places only) - {"195+ countries" if PYCOUNTRY_AVAILABLE else "Limited coverage"}</p>
        <p><strong>Version:</strong> 2.2 Fixed</p>
    </div>
    {st.session_state.html_content}
</body>
</html>"""
                
                st.download_button(
                    label="Download FIXED HTML",
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
        
        if st.button("Process Text with FIXED Disambiguation", type="primary", use_container_width=True):
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
