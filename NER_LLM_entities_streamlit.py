#!/usr/bin/env python3
"""
Streamlit Entity Linker Application with Open Source LLM-Based Entity Extraction

A web interface for entity extraction and linking using open-source LLMs.
This application provides contextual entity extraction and comprehensive linking
to external knowledge bases using free, local models.

Author: Based on NER_spaCy_streamlit.py - Modified to use open-source LLMs
Version: 2.0 - Updated to use open-source LLMs instead of spaCy
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="From Text to Linked Data using Open Source LLM",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication is REQUIRED - do not run app without proper login
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    # Load configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Setup authentication
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any, Optional
import sys
import os
import re
import time
import requests
import urllib.parse

class OpenSourceEntityLinker:
    """
    Main class for open-source LLM-based entity linking functionality.
    
    This class handles the complete pipeline from text processing to entity
    extraction using open-source LLMs, validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialize the OpenSourceEntityLinker."""
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground. 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'EVENT': '#C4A998',           # F&B Dead salmon
            'PRODUCT': '#CCBEAA',         # F&B Oxford stone
            'WORK_OF_ART': '#D4C5B9',     # F&B String
            'ADDRESS': '#E8E1D4',         # F&B Clunch
            'MONEY': '#F0E6D2',           # F&B Lime White
            'DATE': '#E6D7C3',            # F&B Shaded White
            'MISC': '#DDD3C0'             # F&B Old White
        }
        
        # Initialize models
        self.ner_model = None
        self.text_generator = None
        self._load_models()

    def _load_models(self):
        """Load open-source models for entity extraction and text generation."""
        try:
            from transformers import (
                AutoTokenizer, AutoModelForTokenClassification,
                AutoModelForCausalLM, pipeline
            )
            
            # Load NER model (BERT-based NER model)
            with st.spinner("Loading NER model..."):
                try:
                    ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=ner_model_name,
                        tokenizer=ner_model_name,
                        aggregation_strategy="simple"
                    )
                    st.success("✅ NER model loaded successfully")
                except Exception as e:
                    st.warning(f"Primary NER model failed, trying alternative: {e}")
                    # Fallback to a lighter model
                    try:
                        fallback_model = "dslim/bert-base-NER"
                        self.ner_pipeline = pipeline(
                            "ner",
                            model=fallback_model,
                            tokenizer=fallback_model,
                            aggregation_strategy="simple"
                        )
                        st.success("✅ Fallback NER model loaded successfully")
                    except Exception as e2:
                        st.error(f"Failed to load NER model: {e2}")
                        st.stop()
            
            # Load text generation model for contextual analysis
            with st.spinner("Loading text generation model..."):
                try:
                    # Use DistilGPT2 as it's lightweight and free
                    gen_model_name = "distilgpt2"
                    self.text_generator = pipeline(
                        "text-generation",
                        model=gen_model_name,
                        tokenizer=gen_model_name,
                        max_length=512,
                        temperature=0.7,
                        pad_token_id=50256
                    )
                    st.success("✅ Text generation model loaded successfully")
                except Exception as e:
                    st.warning(f"Text generation model failed: {e}")
                    # Continue without text generation
                    self.text_generator = None
                    
        except ImportError:
            st.error("Required packages not installed. Please install:")
            st.code("pip install transformers torch")
            st.stop()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()

    def _generate_contextual_prompt(self, text: str, entity_text: str, entity_type: str) -> str:
        """Generate contextual information about an entity using the text generator."""
        if not self.text_generator:
            return ""
        
        try:
            # Create a prompt for contextual analysis
            prompt = f"Context: {text[:200]}...\n\nThe entity '{entity_text}' is a {entity_type}. Additional context:"
            
            # Generate contextual information
            generated = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True
            )
            
            if generated and len(generated) > 0:
                full_text = generated[0]['generated_text']
                # Extract only the generated part
                context_part = full_text[len(prompt):].strip()
                return context_part[:200]  # Limit length
            
        except Exception as e:
            print(f"Context generation failed: {e}")
            
        return ""

    def extract_entities(self, text: str):
        """Extract named entities from text using open-source NER model with contextual enhancement."""
        if not self.ner_pipeline:
            st.error("NER model not loaded")
            return []
        
        try:
            # Get entities from the NER pipeline
            raw_entities = self.ner_pipeline(text)
            
            entities = []
            
            # Process and enhance each entity
            for ent in raw_entities:
                # Map entity types to our format
                entity_type = self._map_entity_type(ent['entity_group'])
                
                # Filter out low-confidence entities
                if ent['score'] < 0.7:
                    continue
                
                # Create entity dictionary
                entity = {
                    'text': ent['word'],
                    'type': entity_type,
                    'start': ent['start'],
                    'end': ent['end'],
                    'confidence': ent['score'],
                    'original_label': ent['entity_group']
                }
                
                # Add contextual information
                context = self._generate_contextual_prompt(text, ent['word'], entity_type)
                if context:
                    entity['context'] = context
                
                # Additional validation
                if self._is_valid_entity(entity, text):
                    entities.append(entity)
            
            # Extract additional patterns
            additional_entities = self._extract_additional_patterns(text)
            entities.extend(additional_entities)
            
            # Remove overlapping entities
            entities = self._remove_overlapping_entities(entities)
            
            return entities
            
        except Exception as e:
            st.error(f"Entity extraction failed: {e}")
            return []

    def _map_entity_type(self, ner_label: str) -> str:
        """Map NER model labels to our standardized types."""
        mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'GPE',
            'MISC': 'MISC',
            'MONEY': 'MONEY',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'PERCENT': 'MISC',
            'FACILITY': 'FACILITY',
            'EVENT': 'EVENT',
            'PRODUCT': 'PRODUCT',
            'WORK_OF_ART': 'WORK_OF_ART'
        }
        return mapping.get(ner_label, 'MISC')

    def _is_valid_entity(self, entity: Dict[str, Any], text: str) -> bool:
        """Validate an entity using heuristics and context."""
        entity_text = entity['text'].strip()
        
        # Skip very short entities
        if len(entity_text) <= 1:
            return False
        
        # Skip common false positives
        false_positives = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if entity_text.lower() in false_positives:
            return False
        
        # Skip entities that are mostly punctuation
        if len(re.sub(r'[^\w\s]', '', entity_text)) <= 1:
            return False
        
        # Additional validation based on entity type
        if entity['type'] == 'PERSON':
            return self._validate_person_entity(entity_text)
        elif entity['type'] in ['ORGANIZATION', 'LOCATION', 'GPE']:
            return self._validate_place_or_org_entity(entity_text)
        
        return True

    def _validate_person_entity(self, entity_text: str) -> bool:
        """Validate person entities."""
        # Should contain at least one capital letter
        if not any(c.isupper() for c in entity_text):
            return False
        
        # Should not be all caps (likely not a person name)
        if entity_text.isupper() and len(entity_text) > 3:
            return False
        
        # Should not contain numbers
        if any(c.isdigit() for c in entity_text):
            return False
        
        return True

    def _validate_place_or_org_entity(self, entity_text: str) -> bool:
        """Validate place or organization entities."""
        # Should contain at least one capital letter
        if not any(c.isupper() for c in entity_text):
            return False
        
        # Should not be just punctuation
        if not any(c.isalnum() for c in entity_text):
            return False
        
        return True

    def _extract_additional_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract additional entities using pattern matching."""
        additional_entities = []
        
        # Extract addresses
        address_patterns = [
            r'\b\d{1,4}[-–]\d{1,4}\s+[A-Z][a-zA-Z\s]+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens|Court|Close|Crescent)\b',
            r'\b\d{1,4}\s+[A-Z][a-zA-Z\s]+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens|Court|Close|Crescent)\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                additional_entities.append({
                    'text': match.group().strip(),
                    'type': 'ADDRESS',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'original_label': 'ADDRESS_PATTERN'
                })
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            additional_entities.append({
                'text': match.group(),
                'type': 'CONTACT',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'original_label': 'EMAIL_PATTERN'
            })
        
        # Extract phone numbers
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',
            r'\b\+\d{1,3}\s?\d{3}\s?\d{3}\s?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                additional_entities.append({
                    'text': match.group(),
                    'type': 'CONTACT',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'original_label': 'PHONE_PATTERN'
                })
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, text):
            additional_entities.append({
                'text': match.group(),
                'type': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'original_label': 'URL_PATTERN'
            })
        
        return additional_entities

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the highest confidence ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the higher confidence entity
                    if entity.get('confidence', 0) > existing.get('confidence', 0):
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Detect geographical context from the text to improve geocoding accuracy.
        
        Args:
            text: The full input text
            entities: List of extracted entities
        
        Returns:
            List of context strings to use for geocoding
        """
        context_clues = []
        text_lower = text.lower()
        
        # Extract major cities/countries mentioned in the text
        major_locations = {
            # Countries
            'uk': ['uk', 'united kingdom', 'britain', 'great britain', 'england', 'scotland', 'wales'],
            'usa': ['usa', 'united states', 'america', 'us ', 'united states of america'],
            'canada': ['canada'],
            'australia': ['australia'],
            'france': ['france'],
            'germany': ['germany'],
            'italy': ['italy'],
            'spain': ['spain'],
            'japan': ['japan'],
            'china': ['china'],
            'india': ['india'],
            'brazil': ['brazil'],
            'russia': ['russia'],
            
            # Major cities
            'london': ['london'],
            'new york': ['new york', 'nyc', 'manhattan'],
            'paris': ['paris'],
            'tokyo': ['tokyo'],
            'sydney': ['sydney'],
            'toronto': ['toronto'],
            'berlin': ['berlin'],
            'rome': ['rome'],
            'madrid': ['madrid'],
            'beijing': ['beijing'],
            'mumbai': ['mumbai'],
            'los angeles': ['los angeles', 'la ', ' la,'],
            'chicago': ['chicago'],
            'boston': ['boston'],
            'edinburgh': ['edinburgh'],
            'glasgow': ['glasgow'],
            'manchester': ['manchester'],
            'birmingham': ['birmingham'],
            'liverpool': ['liverpool'],
            'bristol': ['bristol'],
            'leeds': ['leeds'],
            'cardiff': ['cardiff'],
            'belfast': ['belfast'],
            'dublin': ['dublin'],
        }
        
        # Check for explicit mentions
        for location, patterns in major_locations.items():
            for pattern in patterns:
                if pattern in text_lower:
                    context_clues.append(location)
                    break
        
        # Extract from entities that are already identified as places
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION']:
                entity_lower = entity['text'].lower()
                # Add major locations found in entities
                for location, patterns in major_locations.items():
                    if entity_lower in patterns or any(p in entity_lower for p in patterns):
                        if location not in context_clues:
                            context_clues.append(location)
        
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
        
        # Prioritize context (more specific first)
        priority_order = ['london', 'new york', 'paris', 'tokyo', 'sydney', 'uk', 'usa', 'canada', 'australia', 'france', 'germany']
        prioritized_context = []
        
        for priority_location in priority_order:
            if priority_location in context_clues:
                prioritized_context.append(priority_location)
        
        # Add remaining context clues
        for clue in context_clues:
            if clue not in prioritized_context:
                prioritized_context.append(clue)
        
        return prioritized_context[:3]  # Return top 3 context clues

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with geographical context detection."""
        # Detect geographical context from the full text
        context_clues = self._detect_geographical_context(
            st.session_state.get('processed_text', ''), 
            entities
        )
        
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
                    
                # Fall back to original methods
                if self._try_python_geocoding(entity):
                    continue
                
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
        
        # Try geopy first with context
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoder = Nominatim(user_agent="EntityLinker/1.0", timeout=10)
            
            for search_term in search_variations[:5]:  # Try top 5 variations
                try:
                    location = geocoder.geocode(search_term, timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_contextual'
                        entity['search_term_used'] = search_term
                        return True
                    
                    time.sleep(0.2)  # Rate limiting
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                    
        except ImportError:
            pass
        
        # Fall back to OpenStreetMap with context
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
    
    def _try_python_geocoding(self, entity):
        """Try Python geocoding libraries (geopy)."""
        try:
            from geopy.geocoders import Nominatim, ArcGIS
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoders = [
                ('nominatim', Nominatim(user_agent="EntityLinker/1.0", timeout=10)),
                ('arcgis', ArcGIS(timeout=10)),
            ]
            
            for name, geocoder in geocoders:
                try:
                    location = geocoder.geocode(entity['text'], timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_{name}'
                        return True
                        
                    time.sleep(0.3)
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                except Exception as e:
                    continue
                        
        except ImportError:
            pass
        except Exception as e:
            pass
        
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
            print(f"OpenStreetMap geocoding failed for {entity['text']}: {e}")
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
        """Add Wikidata linking with enhanced context."""
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                
                # Create enhanced search terms
                search_terms = [entity['text']]
                
                # Add type-specific search enhancements
                if entity['type'] == 'PERSON':
                    search_terms.append(f"{entity['text']} person")
                elif entity['type'] == 'ORGANIZATION':
                    search_terms.append(f"{entity['text']} organization")
                    search_terms.append(f"{entity['text']} company")
                elif entity['type'] in ['LOCATION', 'GPE']:
                    search_terms.append(f"{entity['text']} place")
                    search_terms.append(f"{entity['text']} location")
                
                # Try each search term
                for search_term in search_terms[:3]:  # Limit to top 3 to avoid rate limits
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'search': search_term,
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
                            entity['wikidata_label'] = result.get('label', '')
                            break  # Found a match, stop searching
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
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
                
                # Create enhanced search terms based on entity type and context
                search_terms = [entity['text']]
                
                # Add contextual information if available
                if entity.get('context'):
                    # Extract relevant keywords from context
                    context_keywords = re.findall(r'\b[A-Z][a-z]+\b', entity['context'])
                    for keyword in context_keywords[:2]:  # Use top 2 context keywords
                        search_terms.append(f"{entity['text']} {keyword}")
                
                # Add type-specific enhancements
                if entity['type'] == 'PERSON':
                    search_terms.extend([
                        f"{entity['text']} biography",
                        f"{entity['text']} person"
                    ])
                elif entity['type'] == 'ORGANIZATION':
                    search_terms.extend([
                        f"{entity['text']} company",
                        f"{entity['text']} organization"
                    ])
                elif entity['type'] in ['LOCATION', 'GPE']:
                    search_terms.extend([
                        f"{entity['text']} geography",
                        f"{entity['text']} place"
                    ])
                
                # Try each search term
                for search_term in search_terms[:3]:
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
                            
                            break  # Found a match, stop searching
                
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Wikipedia linking failed for {entity['text']}: {e}")
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add Britannica linking for entities without existing links.""" 
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                
                # Create enhanced search terms
                search_terms = [entity['text']]
                
                # Add contextual search terms
                if entity.get('context'):
                    context_keywords = re.findall(r'\b[A-Z][a-z]+\b', entity['context'])
                    for keyword in context_keywords[:1]:  # Use top context keyword
                        search_terms.append(f"{entity['text']} {keyword}")
                
                # Try each search term
                for search_term in search_terms[:2]:
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
                                break
                        
                        if entity.get('britannica_url'):
                            break  # Found a match, stop searching
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses and places."""
        for entity in entities:
            # Process ADDRESS entities and places with coordinates
            if entity['type'] in ['ADDRESS', 'LOCATION', 'GPE', 'FACILITY']:
                try:
                    # Search OpenStreetMap Nominatim
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
                            
                            # Also add coordinates if not already present
                            if not entity.get('latitude'):
                                entity['latitude'] = float(lat)
                                entity['longitude'] = float(lon)
                                entity['location_name'] = result['display_name']
                    
                    time.sleep(0.2)  # Rate limiting
                except Exception:
                    pass
        
        return entities

    def enhance_entities_with_context(self, entities, text):
        """Enhance entities with additional contextual information."""
        enhanced_entities = []
        
        for entity in entities:
            # Create a copy of the entity
            enhanced_entity = entity.copy()
            
            # Add sentence context
            enhanced_entity['sentence_context'] = self._extract_sentence_context(entity, text)
            
            # Add nearby entities for relationship context
            enhanced_entity['nearby_entities'] = self._find_nearby_entities(entity, entities)
            
            # Add semantic category based on context
            enhanced_entity['semantic_category'] = self._determine_semantic_category(entity, text)
            
            # Add confidence score enhancement based on context
            enhanced_entity['context_confidence'] = self._calculate_context_confidence(entity, text)
            
            enhanced_entities.append(enhanced_entity)
        
        return enhanced_entities

    def _extract_sentence_context(self, entity, text):
        """Extract the sentence containing the entity."""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if entity['text'] in sentence:
                return sentence.strip()
        
        # Fallback: extract context window around entity
        start = max(0, entity['start'] - 100)
        end = min(len(text), entity['end'] + 100)
        return text[start:end].strip()

    def _find_nearby_entities(self, target_entity, all_entities):
        """Find entities that appear near the target entity."""
        nearby = []
        target_start = target_entity['start']
        target_end = target_entity['end']
        
        for entity in all_entities:
            if entity == target_entity:
                continue
            
            # Check if entity is within 200 characters
            if (abs(entity['start'] - target_end) <= 200 or 
                abs(target_start - entity['end']) <= 200):
                nearby.append({
                    'text': entity['text'],
                    'type': entity['type'],
                    'distance': min(abs(entity['start'] - target_end), abs(target_start - entity['end']))
                })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        return nearby[:3]  # Return top 3 nearest entities

    def _determine_semantic_category(self, entity, text):
        """Determine semantic category based on context."""
        context_window = self._extract_sentence_context(entity, text).lower()
        
        # Define keyword patterns for different semantic categories
        categories = {
            'business': ['company', 'corporation', 'business', 'industry', 'market', 'sales', 'revenue'],
            'politics': ['government', 'political', 'election', 'policy', 'minister', 'president', 'parliament'],
            'entertainment': ['movie', 'film', 'actor', 'music', 'band', 'concert', 'show', 'entertainment'],
            'sports': ['team', 'player', 'game', 'match', 'sport', 'league', 'championship', 'tournament'],
            'academic': ['university', 'research', 'study', 'professor', 'academic', 'education', 'school'],
            'technology': ['technology', 'software', 'computer', 'digital', 'internet', 'tech', 'innovation'],
            'healthcare': ['hospital', 'medical', 'health', 'doctor', 'patient', 'treatment', 'medicine'],
            'travel': ['travel', 'tourism', 'hotel', 'airport', 'flight', 'vacation', 'destination']
        }
        
        for category, keywords in categories.items():
            if any(keyword in context_window for keyword in keywords):
                return category
        
        return 'general'

    def _calculate_context_confidence(self, entity, text):
        """Calculate confidence score based on contextual factors."""
        base_confidence = entity.get('confidence', 0.5)
        
        # Factors that increase confidence
        confidence_boost = 0
        
        # Check if entity appears multiple times
        occurrences = text.lower().count(entity['text'].lower())
        if occurrences > 1:
            confidence_boost += 0.1
        
        # Check if entity is properly capitalized
        if entity['text'][0].isupper() and entity['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
            confidence_boost += 0.1
        
        # Check if entity has links (external validation)
        if any(entity.get(key) for key in ['wikidata_url', 'wikipedia_url', 'britannica_url']):
            confidence_boost += 0.2
        
        # Check if entity has coordinates (for places)
        if entity.get('latitude') and entity['type'] in ['LOCATION', 'GPE', 'ADDRESS']:
            confidence_boost += 0.1
        
        # Check context quality
        if entity.get('context') and len(entity['context']) > 20:
            confidence_boost += 0.1
        
        return min(1.0, base_confidence + confidence_boost)


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the OpenSourceEntityLinker class.
    
    Provides a web interface with additional visualization and
    export capabilities for entity analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = OpenSourceEntityLinker()
        
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
    def cached_extract_entities(_self, text: str) -> List[Dict[str, Any]]:
        """Cached entity extraction to avoid reprocessing same text."""
        return _self.entity_linker.extract_entities(text)
    
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
            # Try to load and display the logo
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("💡 Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using Open Source LLM")
        st.markdown("**Extract and link named entities from text using open-source models**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Open Source NER + LLM Context</strong>
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
                    <strong>Enhanced with Context:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #D4C5B9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Semantic Categories</strong><br><small>Business, Politics, etc.</small>
                    </div>
                    <div style="background-color: #CCBEAA; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Entity Relationships</strong><br><small>Nearby entities</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Output Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #EFCA89;">
                         <strong>Enhanced JSON-LD</strong><br><small>Rich structured data</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #C3B5AC;">
                         <strong>Interactive HTML</strong><br><small>Contextual visualization</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with model information."""
        st.sidebar.subheader("Model Information")
        st.sidebar.info("""
        **NER Model**: BERT-based NER (dbmdz/bert-large-cased-finetuned-conll03-english)
        
        **Context Model**: DistilGPT2 for contextual enhancement
        
        **Linking**: Wikidata, Wikipedia, Britannica, OpenStreetMap
        
        **Features**: 
        - Semantic categorization
        - Entity relationships
        - Contextual confidence scoring
        - Enhanced geocoding
        """)
        
        st.sidebar.subheader("Entity Types")
        st.sidebar.info("""
        - **PERSON**: People and names
        - **ORGANIZATION**: Companies, institutions
        - **LOCATION**: Places and geographical entities
        - **GPE**: Countries, cities, states
        - **FACILITY**: Buildings, landmarks
        - **ADDRESS**: Street addresses
        - **CONTACT**: Emails, phone numbers
        - **URL**: Web addresses
        - **EVENT**: Events and occurrences
        - **PRODUCT**: Products and services
        - **WORK_OF_ART**: Creative works
        """)

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Sample text for demonstration
        sample_text = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
        
Tim Cook serves as the CEO, while Steve Jobs was the co-founder. The company was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.

Apple's main campus is located at 1 Apple Park Way, Cupertino, CA 95014. The company is known for products like the iPhone, iPad, and Mac computers.

In 2023, Apple reported revenue of $394.3 billion. The company has retail stores worldwide, including flagship stores in New York, London, and Tokyo.

Contact information: For media inquiries, email press@apple.com or call 408-996-1010. More information is available at https://www.apple.com."""
        
        # Text input area
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,
            height=250,
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        # File upload option
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md)"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    # Set default title from filename if no title provided
                    if not analysis_title:
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Use suggested title if available
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title:
            analysis_title = "text_analysis"
        
        return text_input, analysis_title

    def process_text(self, text: str, title: str):
        """Process the input text using the OpenSourceEntityLinker."""
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Check if we've already processed this exact text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text and extracting entities..."):
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract entities
                status_text.text("Extracting entities with open-source NER model...")
                progress_bar.progress(20)
                entities = self.cached_extract_entities(text)
                
                # Step 2: Enhance with context
                status_text.text("Enhancing entities with contextual information...")
                progress_bar.progress(35)
                entities = self.entity_linker.enhance_entities_with_context(entities, text)
                
                # Step 3: Link to Wikidata
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Link to Wikipedia
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(65)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikipedia(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 5: Link to Britannica
                status_text.text("Linking to Britannica...")
                progress_bar.progress(75)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 6: Get coordinates
                status_text.text("Getting coordinates and geocoding...")
                progress_bar.progress(85)
                entities = self.entity_linker.get_coordinates(entities)
                
                # Step 7: Link to OpenStreetMap
                status_text.text("Linking to OpenStreetMap...")
                progress_bar.progress(95)
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
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Processing complete! Found {len(entities)} entities with enhanced context.")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities and enhanced tooltips."""
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Only highlight entities that have links, coordinates, or high confidence
            has_links = any(entity.get(key) for key in ['britannica_url', 'wikidata_url', 'wikipedia_url', 'openstreetmap_url'])
            has_coordinates = entity.get('latitude') is not None
            high_confidence = entity.get('context_confidence', 0) > 0.7
            
            if not (has_links or has_coordinates or high_confidence):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = self.entity_linker.colors.get(entity['type'], '#E7E2D2')
            
            # Create enhanced tooltip
            tooltip_parts = [f"Type: {entity['type']}"]
            
            if entity.get('semantic_category'):
                tooltip_parts.append(f"Category: {entity['semantic_category']}")
            
            if entity.get('context_confidence'):
                tooltip_parts.append(f"Confidence: {entity['context_confidence']:.2f}")
            
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            elif entity.get('wikipedia_description'):
                tooltip_parts.append(f"Description: {entity['wikipedia_description']}")
            
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            if entity.get('nearby_entities'):
                nearby_texts = [e['text'] for e in entity['nearby_entities'][:2]]
                tooltip_parts.append(f"Nearby: {', '.join(nearby_texts)}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link (priority: Wikipedia > Wikidata > Britannica > OpenStreetMap)
            if entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #1976D2;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #D32F2F;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('openstreetmap_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #388E3C;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just highlight with enhanced styling
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border-left: 3px solid #757575;" title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section with entities and visualizations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Statistics
        self.render_statistics(entities)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Entity analysis tabs
        tab1, tab2, tab3 = st.tabs(["Entity Summary", "🔍 Detailed Analysis", "📤 Export Options"])
        
        with tab1:
            self.render_entity_summary(entities)
        
        with tab2:
            self.render_detailed_analysis(entities)
        
        with tab3:
            self.render_export_section(entities)

    def render_statistics(self, entities: List[Dict[str, Any]]):
        """Render enhanced statistics about the extracted entities."""
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        
        with col3:
            linked_count = len([e for e in entities if any(e.get(key) for key in ['wikidata_url', 'wikipedia_url', 'britannica_url'])])
            st.metric("Linked Entities", linked_count)
        
        with col4:
            avg_confidence = sum(e.get('context_confidence', 0) for e in entities) / len(entities) if entities else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Entity type distribution
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        if type_counts:
            st.subheader("Entity Type Distribution")
            # Create a simple bar chart
            type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
            fig = px.bar(type_df, x='Type', y='Count', color='Type')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Semantic categories
        category_counts = {}
        for entity in entities:
            category = entity.get('semantic_category', 'general')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if len(category_counts) > 1:
            st.subheader("Semantic Categories")
            category_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
            fig = px.pie(category_df, values='Count', names='Category', title="Semantic Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    def render_entity_summary(self, entities: List[Dict[str, Any]]):
        """Render a summary table of entities."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare data for table
        summary_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Category': entity.get('semantic_category', 'general'),
                'Confidence': f"{entity.get('context_confidence', 0):.2f}",
                'Links': self.format_entity_links(entity),
                'Context': entity.get('sentence_context', '')[:100] + "..." if entity.get('sentence_context', '') else ""
            }
            summary_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

    def render_detailed_analysis(self, entities: List[Dict[str, Any]]):
        """Render detailed analysis of entities."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + list(set(e['type'] for e in entities))
            )
        
        with col2:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All"] + list(set(e.get('semantic_category', 'general') for e in entities))
            )
        
        # Apply filters
        filtered_entities = entities
        if type_filter != "All":
            filtered_entities = [e for e in filtered_entities if e['type'] == type_filter]
        if category_filter != "All":
            filtered_entities = [e for e in filtered_entities if e.get('semantic_category') == category_filter]
        
        # Display detailed information for each entity
        for i, entity in enumerate(filtered_entities):
            with st.expander(f"{entity['text']} ({entity['type']})", expanded=i < 5):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- Type: {entity['type']}")
                    st.write(f"- Category: {entity.get('semantic_category', 'general')}")
                    st.write(f"- Confidence: {entity.get('context_confidence', 0):.2f}")
                    st.write(f"- Position: {entity['start']}-{entity['end']}")
                    
                    if entity.get('context'):
                        st.write("**Generated Context:**")
                        st.write(entity['context'])
                
                with col2:
                    st.write("**External Links:**")
                    if entity.get('wikipedia_url'):
                        st.write(f"- [Wikipedia]({entity['wikipedia_url']})")
                    if entity.get('wikidata_url'):
                        st.write(f"- [Wikidata]({entity['wikidata_url']})")
                    if entity.get('britannica_url'):
                        st.write(f"- [Britannica]({entity['britannica_url']})")
                    if entity.get('openstreetmap_url'):
                        st.write(f"- [OpenStreetMap]({entity['openstreetmap_url']})")
                    
                    if entity.get('latitude'):
                        st.write("**Location:**")
                        st.write(f"- Coordinates: {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                        st.write(f"- Address: {entity.get('location_name', 'N/A')}")
                
                if entity.get('sentence_context'):
                    st.write("**Sentence Context:**")
                    st.write(f"_{entity['sentence_context']}_")
                
                if entity.get('nearby_entities'):
                    st.write("**Nearby Entities:**")
                    for nearby in entity['nearby_entities']:
                        st.write(f"- {nearby['text']} ({nearby['type']}) - {nearby['distance']} chars away")

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
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render enhanced export options for the results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced JSON-LD export
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "processingMethod": "Open Source NER + LLM Context Enhancement",
                "entities": []
            }
            
            # Format entities for enhanced JSON-LD
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end'],
                    "confidence": entity.get('context_confidence', 0),
                    "semanticCategory": entity.get('semantic_category', 'general'),
                    "contextualInformation": {
                        "sentenceContext": entity.get('sentence_context', ''),
                        "generatedContext": entity.get('context', ''),
                        "nearbyEntities": entity.get('nearby_entities', [])
                    }
                }
                
                # Add all available links
                same_as_links = []
                if entity.get('wikidata_url'):
                    same_as_links.append(entity['wikidata_url'])
                if entity.get('wikipedia_url'):
                    same_as_links.append(entity['wikipedia_url'])
                if entity.get('britannica_url'):
                    same_as_links.append(entity['britannica_url'])
                if entity.get('openstreetmap_url'):
                    same_as_links.append(entity['openstreetmap_url'])
                
                if same_as_links:
                    entity_data['sameAs'] = same_as_links
                
                # Add descriptions
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                
                # Add geographical information
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
                label="Download Enhanced JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_enhanced_entities.jsonld",
                mime="application/ld+json",
                use_container_width=True
            )
        
        with col2:
            # Enhanced HTML export
            if st.session_state.html_content:
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Entity Analysis: {st.session_state.analysis_title}</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #F5F0DC; }}
                        .header {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #dee2e6; }}
                        .content {{ background: white; padding: 25px; border-radius: 10px; line-height: 1.8; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                        .stat-box {{ text-align: center; padding: 15px; background: #e9ecef; border-radius: 8px; }}
                        .legend {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                        .legend-item {{ display: inline-block; margin: 5px 10px; }}
                        .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px; }}
                        @media (max-width: 768px) {{
                            body {{ padding: 10px; }}
                            .content {{ padding: 15px; }}
                            .stats {{ flex-direction: column; }}
                            .stat-box {{ margin: 5px 0; }}
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Enhanced Entity Analysis: {st.session_state.analysis_title}</h1>
                        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <div class="stats">
                            <div class="stat-box">
                                <strong>{len(entities)}</strong><br>Total Entities
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('latitude')])}</strong><br>Geocoded Places
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if any(e.get(key) for key in ['wikidata_url', 'wikipedia_url', 'britannica_url'])])}</strong><br>Linked Entities
                            </div>
                            <div class="stat-box">
                                <strong>{len(set(e['type'] for e in entities))}</strong><br>Entity Types
                            </div>
                        </div>
                    </div>
                    <div class="legend">
                        <h3>Entity Types:</h3>
                        {''.join([f'<div class="legend-item"><span class="legend-color" style="background-color: {self.entity_linker.colors.get(entity_type, "#E7E2D2")};"></span>{entity_type}</div>' for entity_type in sorted(set(e["type"] for e in entities))])}
                    </div>
                    <div class="content">
                        {st.session_state.html_content}
                    </div>
                    <div class="header">
                        <h3>Processing Details:</h3>
                        <p>• NER Model: BERT-based NER with contextual enhancement</p>
                        <p>• Context Model: DistilGPT2 for semantic analysis</p>
                        <p>• Linked to: Wikidata, Wikipedia, Britannica, OpenStreetMap</p>
                        <p>• Enhanced with: Semantic categories, entity relationships, confidence scoring</p>
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download Enhanced HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_enhanced_entities.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        # Additional export options
        st.subheader("Additional Export Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = []
            for entity in entities:
                csv_data.append({
                    'Entity': entity['text'],
                    'Type': entity['type'],
                    'Confidence': entity.get('context_confidence', 0),
                    'Category': entity.get('semantic_category', 'general'),
                    'Start': entity['start'],
                    'End': entity['end'],
                    'Wikipedia': entity.get('wikipedia_url', ''),
                    'Wikidata': entity.get('wikidata_url', ''),
                    'Britannica': entity.get('britannica_url', ''),
                    'Latitude': entity.get('latitude', ''),
                    'Longitude': entity.get('longitude', ''),
                    'Location': entity.get('location_name', ''),
                    'Context': entity.get('sentence_context', '')
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"{st.session_state.analysis_title}_entities.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # GeoJSON export for entities with coordinates
            geo_entities = [e for e in entities if e.get('latitude')]
            if geo_entities:
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                for entity in geo_entities:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [entity['longitude'], entity['latitude']]
                        },
                        "properties": {
                            "name": entity['text'],
                            "type": entity['type'],
                            "category": entity.get('semantic_category', 'general'),
                            "confidence": entity.get('context_confidence', 0),
                            "description": entity.get('wikidata_description') or entity.get('wikipedia_description', ''),
                            "context": entity.get('sentence_context', '')
                        }
                    }
                    geojson_data["features"].append(feature)
                
                geojson_string = json.dumps(geojson_data, indent=2)
                
                st.download_button(
                    label="Download GeoJSON",
                    data=geojson_string,
                    file_name=f"{st.session_state.analysis_title}_entities.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
            else:
                st.info("No geocoded entities available for GeoJSON export.")

    def run(self):
        """Main application runner."""
        # Add custom CSS for Farrow & Ball styling
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
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #E8E1D4 !important;
            border-radius: 4px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #C4A998 !important;
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
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Process button
        if st.button("Process Text with Open Source Models", type="primary", use_container_width=True):
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
    try:
        app = StreamlitEntityLinker()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please ensure all required packages are installed:")
        st.code("""
        pip install streamlit streamlit-authenticator
        pip install transformers torch
        pip install pandas plotly
        pip install requests geopy
        pip install PyYAML
        """)


if __name__ == "__main__":
    main()decoration: none; color: black; border-left: 3px solid #2E7D32;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-
