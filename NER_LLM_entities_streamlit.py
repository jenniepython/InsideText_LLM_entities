def link_to_britannica(self, entities, full_text=""):
        """Add Britannica linking for entities with enhanced context-aware search.""" 
        for entity in entities:
            # Skip if already has Getty AAT or Wikidata link
            if entity.get('getty_aat_url') or entity.get('wikidata_url'):
                continue
                
            try:
                # Generate context-aware search terms
                search_terms = self._generate_britannica_search_terms(entity, full_text)
                
                for search_term in search_terms[:3]:  # Try up to 3 terms
                    search_url = "https://www.britannica.com/search"
                    params = {'query': search_term}
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }

    def _get_local_context(self, entity, full_text):
        """Get standardized local context around an entity."""
        if not full_text or entity.get('start') is None:
            return ""
        
        start = max(0, entity.get('start', 0) - self.CONTEXT_RADIUS)
        end = min(len(full_text), entity.get('end', 0) + self.CONTEXT_RADIUS)
        return full_text[start:end]
                    
                    response = requests.get(search_url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        # Look for article links
                        pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                        matches = re.findall(pattern, response.text)
                        
                        # Use LLM to select best match if multiple found
                        if matches:
                            best_match = self._select_best_britannica_match(entity, matches, search_term, full_text)
                            if best_match:
                                url_path, link_text = best_match
                                entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                                entity['britannica_title'] = link_text.strip()
                                entity['britannica_search_term'] = search_term
                                entity['britannica_context_used'] = True
                                break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def _generate_britannica_search_terms(self, entity, full_text):
        """Generate context-aware search terms for Britannica using LLM."""
        # Default fallback
        base_terms = [entity['text']]
        
        # Try LLM enhancement if available
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return base_terms
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return base_terms
            
            entity_context = entity.get('context', {})
            
            # Get local context around the entity
            local_context = self._get_local_context(entity, full_text)
            
            if not entity_context and not local_context:
                return base_terms
            
            prompt = f"""Generate Britannica search terms for this entity using context.

ENTITY: "{entity['text']}" (Type: {entity['type']})

GLOBAL CONTEXT:
- Period: {entity_context.get('period', 'unknown')}
- Region: {entity_context.get('region', 'unknown')}
- Culture: {entity_context.get('culture', 'unknown')}
- Subject: {entity_context.get('subject_matter', 'unknown')}

LOCAL CONTEXT: "...{local_context[:500]}..."

TASK: Create 2-3 search terms for Britannica that would find scholarly articles about this entity.

CONSIDERATIONS:
- Britannica focuses on scholarly, educational content
- Add historical/cultural qualifiers to disambiguate
- Consider academic terminology and formal names
- Include geographical, temporal, or cultural context
- Think about how scholars would refer to this entity

EXAMPLES:
- "Paris" in mythology context → ["Paris Troy mythology", "Paris Greek mythology", "Paris Trojan War"]
- "theater" in Roman context → ["Roman theater", "ancient Roman drama", "Roman theatrical architecture"]
- "Argos" in ancient context → ["Argos ancient Greece", "Argos Peloponnese", "ancient Greek city Argos"]

Respond with JSON array of 2-3 search terms:
["term 1", "term 2", "term 3"]
"""
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            enhanced_terms = self.extract_json_from_response(response.text)
            if enhanced_terms and isinstance(enhanced_terms, list):
                # Combine enhanced terms with base terms
                all_terms = enhanced_terms + base_terms
                return list(dict.fromkeys(all_terms))  # Remove duplicates while preserving order
                
        except Exception as e:
            print(f"LLM Britannica search term generation failed: {e}")
        
        return base_terms

    def _select_best_britannica_match(self, entity, matches, search_term, full_text):
        """Use LLM to select the best Britannica match from multiple results."""
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                # Fallback to first match
                return matches[0]
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return matches[0]
            
            entity_context = entity.get('context', {})
            
            # Get local context
            local_context = self._get_local_context(entity, full_text)
            
            # Format matches for LLM
            candidates_text = []
            for i, (url_path, link_text) in enumerate(matches[:5]):  # Limit to 5 matches
                candidates_text.append(f"{i+1}. {link_text.strip()}")
            
            prompt = f"""Select the best Britannica article for this entity.

ENTITY: "{entity['text']}" (Type: {entity['type']})
SEARCH TERM USED: "{search_term}"

GLOBAL CONTEXT:
- Period: {entity_context.get('period', 'unknown')}
- Region: {entity_context.get('region', 'unknown')}
- Culture: {entity_context.get('culture', 'unknown')}
- Subject: {entity_context.get('subject_matter', 'unknown')}

LOCAL CONTEXT: "...{local_context[:400]}..."

BRITANNICA CANDIDATES:
{chr(10).join(candidates_text)}

Which candidate (1-{len(candidates_text)}) best matches this entity in this context?
Consider both the immediate surrounding text and the overall document context.
If none are appropriate, respond with "NONE".

Response format: Just the number (1-{len(candidates_text)}) or "NONE"
"""

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse LLM response
            match = re.search(r'(\d+)', result_text)
            if match:
                choice = int(match.group(1)) - 1
                if 0 <= choice < len(matches):
                    return matches[choice]
            
            # Fallback to first match
            return matches[0]
            
        except Exception as e:
            print(f"LLM Britannica selection failed: {e}")
            return matches[0] if matches else None#!/usr/bin/env python3
"""
Streamlit LLM Entity Linker Application

A web interface for entity extraction using LLM (Gemini) with intelligent linking and geocoding.
This application uses LLM for both entity extraction AND intelligent disambiguation throughout.

Author: Enhanced LLM-driven version
Version: 4.1 - Extended text processing with 5000 char context limit
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="From Text to Linked Data using LLM",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Add custom CSS for Farrow & Ball Slipper Satin background
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

class LLMEntityLinker:
    """
    Main class for LLM-based entity linking functionality.
    
    This class uses Gemini for entity extraction AND intelligent disambiguation
    throughout the entire pipeline - no hardcoded bias.
    """
    
    def __init__(self):
        """Initialize the LLM Entity Linker."""
        # Context configuration - standardized across all services
        self.CONTEXT_RADIUS = 400  # Standard context radius for all linking services
        
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

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response with improved parsing."""
        if not response_text or not isinstance(response_text, str):
            return None
            
        response_text = response_text.strip()
        
        # Try to find JSON array patterns first
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
        
        # Try to parse the entire response as JSON
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
        
        # If all else fails, try to extract key information manually
        print(f"Could not parse JSON from response: {response_text[:200]}...")
        return None

    def analyse_text_context(self, text: str) -> Dict[str, Any]:
        """Analyse text to determine historical, geographical, and cultural context using LLM intelligence."""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                st.error("Google Generative AI library not found. Please install it with: pip install google-generativeai")
                return {}
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found! Please set your API key.")
                return {}
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Use first 5000 characters for context analysis but note full text length
            context_sample = text[:5000]
            full_text_length = len(text)
            
            # Let the LLM analyze the context without any hardcoded assumptions
            prompt = f"""Analyze this text to determine its context for entity disambiguation purposes.

TEXT SAMPLE (first 5000 chars of {full_text_length} total): "{context_sample}{'...' if full_text_length > 5000 else ''}"

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
    "reasoning": "brief explanation of analysis",
    "text_length_analyzed": {len(context_sample)},
    "total_text_length": {full_text_length}
}}

Focus on being accurate and unbiased. If uncertain about any aspect, use null or indicate low confidence.
"""
            
            response = model.generate_content(prompt)
            llm_response = response.text.strip()
            
            # Debug: Show what we got from Gemini
            print(f"Gemini context response (first 500 chars): {llm_response[:500]}")
            
            # Parse the LLM's context analysis
            context_data = self.extract_json_from_response(llm_response)
            
            # Handle different response formats
            context = None
            if context_data and isinstance(context_data, list) and len(context_data) > 0:
                context = context_data[0]
            elif context_data and isinstance(context_data, dict):
                context = context_data
            else:
                # If JSON parsing failed, try to extract basic info from text response
                st.warning("Could not parse structured context response. Attempting manual extraction.")
                context = self._extract_context_manually(llm_response)
                if not context:
                    context = {
                        'period': None,
                        'region': None,
                        'culture': None,
                        'subject_matter': None,
                        'language_style': 'modern',
                        'time_indicators': [],
                        'place_indicators': [],
                        'subject_indicators': [],
                        'confidence': 'low',
                        'reasoning': 'Fallback - JSON parsing failed'
                    }
            
            # Ensure context is a dictionary
            if not isinstance(context, dict):
                st.error(f"Context analysis returned unexpected format: {type(context)}")
                context = {
                    'period': None,
                    'region': None,
                    'culture': None,
                    'subject_matter': None,
                    'language_style': 'modern',
                    'time_indicators': [],
                    'place_indicators': [],
                    'subject_indicators': [],
                    'confidence': 'low',
                    'reasoning': 'Error in context analysis'
                }
            
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
                'analysis_method': 'llm_driven',
                'text_length_analyzed': len(context_sample),
                'total_text_length': full_text_length
            }
            
        except Exception as e:
            st.error(f"LLM context analysis failed: {e}")
            return {}

    def _extract_context_manually(self, response_text: str) -> Dict[str, Any]:
        """Manually extract context information from LLM response when JSON parsing fails."""
        try:
            context = {
                'period': None,
                'region': None,
                'culture': None,
                'subject_matter': None,
                'language_style': 'modern',
                'time_indicators': [],
                'place_indicators': [],
                'subject_indicators': [],
                'confidence': 'low',
                'reasoning': 'Manual extraction from non-JSON response'
            }
            
            response_lower = response_text.lower()
            
            # Try to extract period information
            period_keywords = {
                'ancient': ['ancient', 'antiquity', 'classical'],
                'medieval': ['medieval', 'middle ages', 'feudal'],
                'renaissance': ['renaissance', 'early modern'],
                'modern': ['modern', 'industrial', 'contemporary'],
                'roman': ['roman', 'rome'],
                'greek': ['greek', 'greece', 'hellenic'],
                'byzantine': ['byzantine', 'byzantium']
            }
            
            for period, keywords in period_keywords.items():
                if any(keyword in response_lower for keyword in keywords):
                    context['period'] = period
                    break
            
            # Try to extract regional information
            region_keywords = {
                'europe': ['europe', 'european'],
                'mediterranean': ['mediterranean', 'aegean'],
                'asia': ['asia', 'asian'],
                'greece': ['greece', 'greek', 'hellenic'],
                'italy': ['italy', 'italian', 'rome'],
                'britain': ['britain', 'british', 'england']
            }
            
            for region, keywords in region_keywords.items():
                if any(keyword in response_lower for keyword in keywords):
                    context['region'] = region
                    break
            
            # Extract confidence if mentioned
            if 'high confidence' in response_lower:
                context['confidence'] = 'high'
            elif 'medium confidence' in response_lower:
                context['confidence'] = 'medium'
            elif 'low confidence' in response_lower:
                context['confidence'] = 'low'
            
            return context
            
        except Exception as e:
            print(f"Manual context extraction failed: {e}")
            return None

    def extract_entities(self, text: str):
        """Extract named entities from text using Gemini LLM with context awareness - processes text directly."""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                st.error("Google Generative AI library not found. Please install it with: pip install google-generativeai")
                return []
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY environment variable not found!")
                return []
            
            # First, analyse the text context using 5000 char sample
            context = self.analyse_text_context(text)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Process text directly
            return self._extract_entities_single_pass(text, context, model)
            
        except Exception as e:
            st.error(f"Error in LLM entity extraction: {e}")
            st.exception(e)
            return []

    def _extract_entities_single_pass(self, text: str, context: Dict[str, Any], model):
        """Extract entities from a single text passage."""
        # SIMPLIFIED PROMPT - much more direct
        prompt = f"""Extract ALL named entities from this text. Return a JSON array only.

Text: "{text}"

Find these entity types:
- PERSON: People's names (e.g., "John Smith", "Dr. Johnson")
- ORGANIZATION: Companies, groups, institutions 
- GPE: Cities, countries, regions (e.g., "London", "England")
- LOCATION: Geographic places, landmarks
- FACILITY: Buildings, venues, theaters, institutions
- ADDRESS: Street addresses
- DATE: Years, dates, time periods
- PRODUCT: Objects, tools, equipment
- WORK_OF_ART: Books, plays, artworks
- EVENT: Named events

For each entity found, return:
{{"text": "entity name", "type": "ENTITY_TYPE", "start_pos": 0}}

Extract ALL entities you can find. Return ONLY a JSON array, no other text:
"""
        
        # Use simpler prompt
        gemini_response = model.generate_content(prompt)
        llm_response = gemini_response.text
        
        entities_raw = self.extract_json_from_response(llm_response)
        
        if not entities_raw:
            st.error("Could not parse JSON from Gemini response. Please try again.")
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

    def link_to_getty_aat(self, entities, full_text=""):
        """Add Getty Art & Architecture Thesaurus linking using enhanced search with context."""
        for entity in entities:
            # Getty AAT is particularly valuable for these entity types
            relevant_types = ['PRODUCT', 'FACILITY', 'WORK_OF_ART', 'EVENT', 'ORGANIZATION']
            if entity['type'] not in relevant_types:
                continue
                
            try:
                # Try enhanced Getty AAT search with context
                getty_result = self._search_getty_enhanced_with_context(entity, full_text)
                
                if getty_result:
                    entity['getty_aat_url'] = getty_result['url']
                    entity['getty_aat_label'] = getty_result['label']
                    entity['getty_aat_description'] = getty_result['description']
                    entity['getty_search_term'] = getty_result['search_term']
                    entity['getty_context_used'] = getty_result.get('context_used', False)
                
                time.sleep(0.5)  # Conservative rate limiting for Getty
                
            except Exception as e:
                print(f"Getty AAT search failed for {entity['text']}: {e}")
                pass
        
        return entities

    def _search_getty_enhanced_with_context(self, entity, full_text):
        """Enhanced Getty AAT search using LLM to generate context-aware search terms."""
        # Get local context around the entity
        local_context = self._get_local_context(entity, full_text)
        
        # Generate context-aware search terms using LLM
        search_terms = self._generate_getty_search_terms(entity, local_context)
        
        # Try each search term
        for search_term in search_terms[:4]:  # Try up to 4 terms
            result = self._try_getty_sparql(search_term)
            if result:
                result['context_used'] = True
                return result
            
            result = self._try_getty_web_search(search_term)
            if result:
                result['context_used'] = True
                return result
        
        return None

    def _generate_getty_search_terms(self, entity, local_context):
        """Generate contextual search terms for Getty AAT using LLM."""
        # Default fallback
        base_terms = [entity['text']]
        
        # Add basic variations
        text_lower = entity['text'].lower()
        if text_lower.endswith('s') and len(entity['text']) > 3:
            base_terms.append(entity['text'][:-1])  # Remove 's'
        elif not text_lower.endswith('s'):
            base_terms.append(entity['text'] + 's')  # Add 's'
        
        # Try LLM enhancement if available
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return base_terms
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return base_terms
            
            entity_context = entity.get('context', {})
            if not entity_context and not local_context:
                return base_terms
            
            prompt = f"""Generate Getty Art & Architecture Thesaurus search terms for this entity.

ENTITY: "{entity['text']}" (Type: {entity['type']})

GLOBAL CONTEXT:
- Period: {entity_context.get('period', 'unknown')}
- Region: {entity_context.get('region', 'unknown')}  
- Culture: {entity_context.get('culture', 'unknown')}
- Subject: {entity_context.get('subject_matter', 'unknown')}

LOCAL CONTEXT: "...{local_context[:500]}..."

TASK: Create 2-3 search terms for Getty AAT that would find the correct architectural/cultural term.

CONSIDERATIONS:
- Getty AAT focuses on art, architecture, cultural objects, materials, techniques
- Add historical/cultural qualifiers to disambiguate
- Consider alternative terminology (ancient vs classical, theater vs theatre)
- Include material, style, or functional descriptors

EXAMPLES:
- "theater" in ancient context → ["theater architecture", "ancient theater", "classical theater"]
- "column" in Roman context → ["Roman column", "classical column", "architectural column"]

Respond with JSON array of 2-3 search terms:
["term 1", "term 2", "term 3"]
"""
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            enhanced_terms = self.extract_json_from_response(response.text)
            if enhanced_terms and isinstance(enhanced_terms, list):
                # Combine enhanced terms with base terms
                all_terms = enhanced_terms + base_terms
                return list(dict.fromkeys(all_terms))  # Remove duplicates while preserving order
                
        except Exception as e:
            print(f"LLM Getty search term generation failed: {e}")
        
        return base_terms

    def _try_getty_sparql(self, search_term):
        """Try Getty AAT SPARQL endpoint."""
        try:
            search_url = "http://vocab.getty.edu/sparql.json"
            
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
            LIMIT 3
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'EntityLinker/1.0'
            }
            
            response = requests.get(search_url, params={'query': query}, 
                                  headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    result = results[0]
                    subject_uri = result['subject']['value']
                    term = result['term']['value']
                    note = result.get('note', {}).get('value', '')
                    
                    aat_id = subject_uri.split('/')[-1]
                    web_url = f"http://www.getty.edu/vow/AATFullDisplay?find=&logic=AND&note=&page=1&subjectid={aat_id}"
                    
                    return {
                        'url': web_url,
                        'label': term,
                        'description': note[:200] if note else f"Getty AAT entry for {term}",
                        'search_term': search_term
                    }
                    
        except Exception as e:
            print(f"Getty SPARQL failed for '{search_term}': {e}")
            
        return None

    def _try_getty_web_search(self, search_term):
        """Try Getty AAT web interface search."""
        try:
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
                aat_pattern = r'AATFullDisplay\?[^"]*subjectid=(\d+)[^"]*"[^>]*>([^<]+)</a>'
                matches = re.findall(aat_pattern, response.text)
                
                if matches:
                    aat_id, label = matches[0]
                    web_url = f"http://www.getty.edu/vow/AATFullDisplay?find=&logic=AND&note=&page=1&subjectid={aat_id}"
                    
                    return {
                        'url': web_url,
                        'label': label.strip(),
                        'description': f"Getty AAT cultural/architectural term: {label.strip()}",
                        'search_term': search_term
                    }
                    
        except Exception as e:
            print(f"Getty web search failed for '{search_term}': {e}")
        
        return None

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
        # Safe default
        base = [entity['text']]
    
        # Try Gemini only if installed + key set
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return base
    
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return base
    
            entity_context = entity.get('context', {})
            if not entity_context:
                return base
    
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
    
    Respond with JSON array of 2-3 search queries:
    ["query 1", "query 2", "query 3"]
    """
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
    
            # Prefer your robust extractor if present
            if hasattr(self, "extract_json_from_response"):
                queries = self.extract_json_from_response(text)
            else:
                import json
                queries = json.loads(text)
    
            # Normalize and validate
            if isinstance(queries, list):
                queries = [q for q in (q.strip() for q in queries) if q]
                if 1 <= len(queries) <= 5 and all(isinstance(q, str) for q in queries):
                    return queries
        except Exception as e:
            print(f"LLM search query generation failed: {e}")
    
        return base

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
            try:
                import google.generativeai as genai
            except ImportError:
                # Fallback to first result
                result = results[0]
                return {
                    'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                    'wikidata_description': result.get('description', ''),
                    'wikidata_search_query': search_query
                }
            
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
            
            # Skip LLM if context is empty (no API key case)
            if not entity_context:
                result = results[0]
                return {
                    'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                    'wikidata_description': result.get('description', ''),
                    'wikidata_search_query': search_query
                }
            
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

    def link_to_britannica(self, entities):
        """Add Britannica linking for entities without higher priority links.""" 
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
                        'type': self._classify_result_type(result['title'], clean_snippet)
                    })
                
                return candidates
                
        except Exception as e:
            print(f"Error getting Wikipedia candidates: {e}")
            
        return []

    def _classify_result_type(self, title, snippet):
        """Add metadata to help LLM make better decisions."""
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        if 'disambiguation' in title_lower:
            return 'disambiguation_page'
        elif any(term in snippet_lower for term in ['wiktionary', 'look up']):
            return 'dictionary_reference'  
        else:
            return 'standard_article'

    def llm_disambiguate_wikipedia(self, entity, candidates, full_text):
        """Use Gemini to pick the best Wikipedia match based on context."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
    
        # Try Gemini; otherwise fall back gracefully
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                # No LLM available
                return self._fallback_simple_disambiguation(entity, candidates, full_text)
    
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._fallback_simple_disambiguation(entity, candidates, full_text)
    
            entity_context = entity.get('context', {})
            if not entity_context:
                return self._fallback_simple_disambiguation(entity, candidates, full_text)
    
            # local and global context
            local_context = self._get_local_context(entity, full_text)
            context_sample = full_text[:5000]
    
            # format candidates for prompt
            lines = []
            for i, c in enumerate(candidates):
                ctype = c.get('type', 'standard_article')
                desc = (c.get('description') or '')[:500]
                lines.append(f"{i+1}. Title: {c.get('title','')}\n   Type: {ctype}\n   Description: {desc}\n")
    
            prompt = f"""You are an expert at disambiguating Wikipedia links using contextual analysis.
    
    ENTITY TO DISAMBIGUATE: "{entity['text']}" (Entity Type: {entity['type']})
    
    FULL TEXT CONTEXT (first 5000 chars): "{context_sample}{'...' if len(full_text) > 5000 else ''}"
    
    LOCAL CONTEXT AROUND ENTITY: "...{local_context}..."
    
    DETECTED CONTEXT (from previous analysis):
    - Period: {entity_context.get('period','unknown')}
    - Region: {entity_context.get('region','unknown')}
    - Culture: {entity_context.get('culture','unknown')}
    - Subject Matter: {entity_context.get('subject_matter','unknown')}
    - Confidence: {entity_context.get('confidence','unknown')}
    
    WIKIPEDIA CANDIDATES:
    {chr(10).join(lines)}
    
    TASK: Analyze the context and select the best candidate.
    
    Respond with:
    Choice: [number or NONE]
    Confidence: [high/medium/low]
    Reasoning: [your analysis]
    """
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            result = (response.text or "").strip()
    
            import re
            choice_match = re.search(r'Choice:\s*(\d+|NONE)', result, re.IGNORECASE)
            conf_match = re.search(r'Confidence:\s*(high|medium|low)', result, re.IGNORECASE)
            reason_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', result, re.IGNORECASE | re.DOTALL)
    
            if not choice_match:
                return self._fallback_simple_disambiguation(entity, candidates, full_text)
    
            choice_str = choice_match.group(1).upper()
            if choice_str == "NONE":
                return None
    
            try:
                idx = int(choice_str) - 1
                if 0 <= idx < len(candidates):
                    chosen = dict(candidates[idx])  # copy to annotate
                    chosen['disambiguation_method'] = 'llm_contextual'
                    if conf_match:
                        chosen['disambiguation_confidence'] = conf_match.group(1).lower()
                    if reason_match:
                        chosen['disambiguation_reasoning'] = reason_match.group(1).strip()
                    chosen['candidates_available'] = len(candidates)
                    return chosen
            except ValueError:
                pass
    
            return self._fallback_simple_disambiguation(entity, candidates, full_text)
    
        except Exception as e:
            print(f"LLM disambiguation failed: {e}")
            return self._fallback_simple_disambiguation(entity, candidates, full_text)

    def _fallback_simple_disambiguation(self, entity, candidates, full_text):
        """Simplified LLM disambiguation if main method fails."""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return candidates[0]
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return candidates[0]
            
            # Skip LLM if context is empty (no API key case)
            entity_context = entity.get('context', {})
            if not entity_context:
                return candidates[0]
            
            # Super simple prompt as backup - use first 2000 chars
            candidates_simple = [f"{i+1}. {c['title']}: {c['description'][:200]}" 
                               for i, c in enumerate(candidates)]
            
            simple_prompt = f"""Text context: "{full_text[:2000]}..."
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

    def link_to_wikipedia_with_llm_disambiguation(self, entities, full_text):
        """Use LLM to help disambiguate Wikipedia results based on context - LAST RESORT only."""
        
        for entity in entities:
            # Skip if already has higher priority links
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
        # Use LLM to detect geographical context from the full text (first 5000 chars)
        geographical_context = self._llm_detect_geographical_context(processed_text, entities)
        
        if geographical_context:
            print(f"LLM detected geographical context: {geographical_context}")
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try LLM-enhanced contextual geocoding
                if self._try_llm_contextual_geocoding(entity, geographical_context, processed_text):
                    continue
                    
                # Fallback to basic geocoding
                if self._try_basic_geocoding(entity):
                    continue
        
        return entities

    def _llm_detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Use LLM to intelligently detect geographical context from the full text."""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return ""
            
            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return ""
            
            # Extract place entities for context
            place_entities = [e['text'] for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY']]
            
            # Use first 2000 chars for geographical context detection
            context_sample = text[:2000]
            
            prompt = f"""Analyze this text to determine the PRIMARY geographical context for geocoding purposes.

TEXT SAMPLE: "{context_sample}{'...' if len(text) > 2000 else ''}"

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
            try:
                import google.generativeai as genai
            except ImportError:
                return self._try_basic_geocoding(entity)
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._try_basic_geocoding(entity)
            
            # Get local context around the entity
            local_context = self._get_local_context(entity, full_text)
            
            # Let LLM decide how to adapt context for modern geocoding
            prompt = f"""You need to help geocode a location for mapping purposes.

ENTITY: "{entity['text']}" (Type: {entity['type']})
DETECTED CONTEXT: "{geographical_context}"
SURROUNDING TEXT: "{local_context}"

TASK: Create 2-3 search terms that would help find the correct modern location for mapping this entity.

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
        return self._try_basic_geocoding(entity)

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

    def _try_basic_geocoding(self, entity):
        """Basic geocoding fallback using just the entity name."""
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
                    entity['geocoding_source'] = 'basic_geocoding'
                    return True
        
            time.sleep(0.3)
        except Exception:
            pass
        
        return False

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
    
    Provides clean interface for LLM-driven entity extraction and intelligent disambiguation.
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

    def render_header(self):
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using LLM")
        st.markdown("**Extract and link named entities from text using Gemini LLM with intelligent disambiguation**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">&#8595;</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Gemini LLM Context Analysis</strong><br><small>First 5000 chars for context</small>
                </div>
                <div style="margin: 10px 0;">&#8595;</div>
                <div style="background-color: #BF7B69; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>LLM Entity Recognition</strong><br><small>Processes full text</small>
                </div>
                <div style="margin: 10px 0;">&#8595;</div>
                <div style="text-align: center;">
                    <strong>Intelligent Linking Priority:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Getty AAT</strong><br><small>Cultural/architectural terms</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikidata</strong><br><small>Structured knowledge</small>
                    </div>
                    <div style="background-color: #E6D7C9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Britannica</strong><br><small>Scholarly articles</small>
                    </div>
                    <div style="background-color: #D4C5B9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikipedia</strong><br><small>LLM disambiguated</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">&#8595;</div>
                <div style="background-color: #F0EAE2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>LLM-Enhanced Geocoding</strong><br><small>Context-aware coordinates</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with information about LLM approach."""
        st.sidebar.subheader("Enhanced LLM Processing")
        st.sidebar.info("""Context Analysis: First 5,000 characters
Entity Extraction: Full text processing
Disambiguation: Extended context analysis""")
        
        st.sidebar.subheader("Linking Priority")
        st.sidebar.info("""1) Getty AAT (cultural/architectural)
2) Wikidata (structured)
3) Britannica (scholarly)
4) Wikipedia (with LLM disambiguation)""")

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Text input area
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content."
        )
        
        # Show character count
        if text_input:
            char_count = len(text_input)
            st.caption(f"Text length: {char_count:,} characters")
        
        # File upload option in expander
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
                    char_count = len(uploaded_text)
                    st.success(f"File uploaded successfully! ({char_count:,} characters)")
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
        """Process the input text using the LLM EntityLinker with full intelligence."""
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return
        
        # Check for API key first
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY environment variable not found! Please set your API key to use this application.")
            st.info("This application requires a Gemini API key to function. Please set the GEMINI_API_KEY environment variable.")
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
                
                # Step 1: LLM context analysis (5000 char sample)
                status_text.text("LLM analyzing text context (first 5000 chars)...")
                progress_bar.progress(10)
                text_context = self.entity_linker.analyse_text_context(text)
                
                # Check if context analysis failed (empty dict means API key issue)
                if not text_context:
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Show context analysis results early
                if text_context.get('period') or text_context.get('region') or text_context.get('subject_matter'):
                    context_info = []
                    if text_context.get('period'):
                        context_info.append(f"Period: {text_context['period']}")
                    if text_context.get('region'):
                        context_info.append(f"Region: {text_context['region']}")
                    if text_context.get('subject_matter'):
                        context_info.append(f"Subject: {text_context['subject_matter']}")
                    
                    analyzed_chars = text_context.get('text_length_analyzed', 0)
                    total_chars = text_context.get('total_text_length', len(text))
                    
                    st.info(f"LLM detected context: {' | '.join(context_info)} (analyzed {analyzed_chars:,}/{total_chars:,} chars)")
                
                # Store context for later use
                st.session_state.text_context = text_context
                
                # Step 2: Extract entities using LLM (FULL TEXT)
                status_text.text("LLM extracting entities from full text...")
                progress_bar.progress(25)
                entities_json = self.cached_extract_entities(text)
                entities = json.loads(entities_json)
                
                if not entities:
                    st.warning("No entities found in the text.")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                st.success(f"Found {len(entities)} entities in text")
                
                # Step 3: Link to Getty AAT - FIRST PRIORITY
                status_text.text("Linking to Getty Art & Architecture Thesaurus...")
                progress_bar.progress(40)
                entities = self.entity_linker.link_to_getty_aat(entities, text)
                
                # Step 4: Link to Wikidata - SECOND PRIORITY
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(55)
                entities = self.entity_linker.link_to_wikidata(entities, text)
                
                # Step 5: Link to Britannica - THIRD PRIORITY
                status_text.text("Linking to Britannica...")
                progress_bar.progress(70)
                entities = self.entity_linker.link_to_britannica(entities, text)
                
                # Step 6: LLM Wikipedia disambiguation - LAST RESORT
                status_text.text("LLM Wikipedia disambiguation...")
                progress_bar.progress(80)
                entities = self.entity_linker.link_to_wikipedia_with_llm_disambiguation(entities, text)
                
                # Step 7: LLM-enhanced geocoding
                status_text.text("LLM-enhanced geocoding...")
                progress_bar.progress(90)
                place_entities = [e for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']]
                
                if place_entities:
                    try:
                        geocoded_entities = self.entity_linker.get_coordinates(place_entities, text)
                        
                        # Update the entities list with geocoded results
                        for geocoded_entity in geocoded_entities:
                            for idx, entity in enumerate(entities):
                                if (entity['text'] == geocoded_entity['text'] and 
                                    entity['type'] == geocoded_entity['type'] and
                                    entity['start'] == geocoded_entity['start']):
                                    entities[idx] = geocoded_entity
                                    break
                    except Exception as e:
                        st.warning(f"Some geocoding failed: {e}")
                
                # Step 8: Link addresses to OpenStreetMap
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 9: Generate visualization
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                # text_context already stored above
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show results summary
                disambiguation_stats = self._get_disambiguation_stats(entities)
                linking_stats = self._get_linking_stats(entities)
                
                st.success(f"Processing complete! Found {len(entities)} entities with {linking_stats['total_linked']} linked.")
                
                # Show detailed stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Entities", len(entities))
                
                with col2:
                    st.metric("Linked Entities", linking_stats['total_linked'])
                
                with col3:
                    geocoded_count = len([e for e in entities if e.get('latitude')])
                    st.metric("Geocoded", geocoded_count)
                
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

    def _get_linking_stats(self, entities):
        """Get statistics about linking success."""
        stats = {
            'total_linked': 0,
            'getty_aat': 0,
            'wikidata': 0,
            'britannica': 0,
            'wikipedia': 0,
            'openstreetmap': 0
        }
        
        for entity in entities:
            has_link = False
            
            if entity.get('getty_aat_url'):
                stats['getty_aat'] += 1
                has_link = True
            if entity.get('wikidata_url'):
                stats['wikidata'] += 1
                has_link = True
            if entity.get('britannica_url'):
                stats['britannica'] += 1
                has_link = True
            if entity.get('wikipedia_url'):
                stats['wikipedia'] += 1
                has_link = True
            if entity.get('openstreetmap_url'):
                stats['openstreetmap'] += 1
                has_link = True
            
            if has_link:
                stats['total_linked'] += 1
        
        return stats

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities for display - HIGHLIGHT ALL ENTITIES."""
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
        
        # Map each character position to its entity (if any) - HIGHLIGHT ALL ENTITIES, NOT JUST LINKED ONES
        for entity in entities:
            start = entity.get('start', -1)
            end = entity.get('end', -1)
            
            # Skip if positions are invalid
            if start < 0 or end > len(text) or start >= end:
                continue
                
            # Mark characters as belonging to this entity - HIGHLIGHT ALL
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
                    desc = entity['wikidata_description'][:100]
                    tooltip_parts.append(f"Description: {desc}")
                elif entity.get('wikipedia_description'):
                    desc = entity['wikipedia_description'][:100]
                    tooltip_parts.append(f"Description: {desc}")
                if entity.get('location_name'):
                    loc = entity['location_name'][:100]
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
                    # HIGHLIGHT ALL ENTITIES, even without links
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
        """Render the results section with entities and visualizations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Show enhanced statistics
        disambiguation_stats = self._get_disambiguation_stats(entities)
        linking_stats = self._get_linking_stats(entities)
        
        # Enhanced statistics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            st.metric("Linked", linking_stats['total_linked'])
        
        with col3:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded", geocoded_count)
        
        with col4:
            st.metric("LLM Disambiguated", disambiguation_stats['llm_disambiguated'])
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            # Show text length info
            text_length = len(st.session_state.processed_text)
            st.caption(f"Text length: {text_length:,} characters - All entities highlighted regardless of links")
            
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Entity details in collapsible section
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        # Export options in collapsible section
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details."""
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
                row['Description'] = entity['getty_aat_description'][:100] + '...' if len(entity['getty_aat_description']) > 100 else entity['getty_aat_description']
            elif entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description'][:100] + '...' if len(entity['wikidata_description']) > 100 else entity['wikidata_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description'][:100] + '...' if len(entity['wikipedia_description']) > 100 else entity['wikipedia_description']
            
            if entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')[:50] + '...' if len(entity.get('location_name', '')) > 50 else entity.get('location_name', '')
            
            # Add disambiguation method info
            if entity.get('disambiguation_method') == 'llm_contextual':
                row['Method'] = f"LLM ({entity.get('candidates_considered', 1)} candidates)"
            elif entity.get('candidates_considered'):
                row['Method'] = f"Auto ({entity.get('candidates_considered', 1)} candidates)"
            else:
                row['Method'] = "Direct"
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
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
        """Render export options for the results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export - create JSON-LD format
            text_length = len(st.session_state.processed_text)
            context_info = getattr(st.session_state, 'text_context', {})
            
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "processingMethod": "LLM entity extraction with intelligent disambiguation",
                "textLength": text_length,
                "contextAnalysis": {
                    "period": context_info.get('period'),
                    "region": context_info.get('region'),
                    "culture": context_info.get('culture'),
                    "subjectMatter": context_info.get('subject_matter'),
                    "confidence": context_info.get('confidence'),
                    "charactersAnalyzed": context_info.get('text_length_analyzed', 0),
                    "totalCharacters": context_info.get('total_text_length', text_length)
                },
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
                
                # Add links in priority order
                if entity.get('getty_aat_url'):
                    entity_data['sameAs'] = entity['getty_aat_url']
                
                if entity.get('wikidata_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['wikidata_url']]
                        else:
                            entity_data['sameAs'].append(entity['wikidata_url'])
                    else:
                        entity_data['sameAs'] = entity['wikidata_url']
                
                # Add other links
                for url_key in ['britannica_url', 'wikipedia_url', 'openstreetmap_url']:
                    if entity.get(url_key):
                        if 'sameAs' in entity_data:
                            if isinstance(entity_data['sameAs'], str):
                                entity_data['sameAs'] = [entity_data['sameAs'], entity[url_key]]
                            else:
                                entity_data['sameAs'].append(entity[url_key])
                        else:
                            entity_data['sameAs'] = entity[url_key]
                
                # Add description
                if entity.get('getty_aat_description'):
                    entity_data['description'] = entity['getty_aat_description']
                elif entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                
                # Add geo data
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                    if entity.get('geocoding_source'):
                        entity_data['geo']['source'] = entity['geocoding_source']
                
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
            # HTML export
            if st.session_state.html_content:
                context_info = getattr(st.session_state, 'text_context', {})
                text_length = len(st.session_state.processed_text)
                
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
        .context-info {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
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
    <div class="processing-info">
        <strong>Generated by LLM Entity Linker</strong><br>
        Entities extracted and disambiguated using Gemini LLM with intelligent context analysis.<br>
        No hardcoded bias - fully LLM-driven approach processing {text_length:,} characters.
    </div>
    <div class="context-info">
        <strong>Context Analysis:</strong> 
        Period: {context_info.get('period', 'Unknown')} | 
        Region: {context_info.get('region', 'Unknown')} | 
        Subject: {context_info.get('subject_matter', 'Unknown')}
        <br>
        <strong>Processing:</strong> {len(st.session_state.entities)} entities found
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
        """Main application runner."""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
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
