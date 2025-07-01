#!/usr/bin/env python3
"""
Sustainable Generic Prompt Engineering Entity Extraction

A sustainable approach using:
1. Efficient prompt caching
2. Batch processing when possible
3. Smart model selection
4. Generic prompting (no hardcoded patterns)

Author: Sustainable Generic AI Version
Version: 3.0
"""

import streamlit as st

# Configure Streamlit page FIRST
st.set_page_config(
    page_title="Sustainable Entity Extraction",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication section (keeping your existing auth code)
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
    else:
        try:
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            if login_result is None:
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()
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
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.stop()

import pandas as pd
import json
import hashlib
from typing import List, Dict, Any, Optional
import requests
import time
import re


class SustainablePromptExtractor:
    """
    Sustainable entity extraction using intelligent prompt engineering.
    Focuses on efficiency, caching, and minimal energy usage.
    """
    
    def __init__(self):
        """Initialize the sustainable prompt extractor."""
        self.colors = {
            'PERSON': '#BF7B69',
            'ORGANIZATION': '#9fd2cd',
            'GPE': '#C4C3A2',
            'LOCATION': '#EFCA89',
            'FACILITY': '#C3B5AC',
            'ADDRESS': '#CCBEAA'
        }
        
        # Sustainability tracking
        self.energy_metrics = {
            'api_calls_made': 0,
            'cache_hits': 0,
            'total_tokens_processed': 0
        }
        
        # Initialize cache
        if 'prompt_cache' not in st.session_state:
            st.session_state.prompt_cache = {}

    def extract_entities(self, text: str, domain_hint: str = "") -> List[Dict[str, Any]]:
        """
        Extract entities using sustainable prompt engineering.
        
        Args:
            text: Input text to analyze
            domain_hint: Optional domain context for better extraction
        """
        # Check cache first (zero energy cost)
        cache_key = self._get_cache_key(text, domain_hint)
        if cache_key in st.session_state.prompt_cache:
            self.energy_metrics['cache_hits'] += 1
            print(f"Cache hit - zero energy used")
            return st.session_state.prompt_cache[cache_key]
        
        # Use efficient prompting strategy
        entities = self._extract_with_efficient_prompting(text, domain_hint)
        
        # Cache the result
        st.session_state.prompt_cache[cache_key] = entities
        
        return entities

    def _get_cache_key(self, text: str, domain_hint: str) -> str:
        """Generate cache key for text and domain combination."""
        combined = f"{text}_{domain_hint}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _extract_with_efficient_prompting(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Extract entities using efficient, generic prompting."""
        
        # Strategy 1: Single optimized prompt (most efficient)
        entities = self._try_optimized_single_prompt(text, domain_hint)
        if entities:
            return entities
        
        # Strategy 2: Fallback to structured prompt
        entities = self._try_structured_prompt(text, domain_hint)
        if entities:
            return entities
        
        # Strategy 3: Final fallback to simple prompt
        entities = self._try_simple_prompt(text)
        return entities

    def _try_optimized_single_prompt(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Most efficient single prompt approach."""
        
        # Create domain-aware prompt
        domain_context = self._get_domain_context(domain_hint)
        
        prompt = f"""Extract named entities from this text. Focus on {domain_context} if relevant.

Return a JSON array with entities in this format:
[{{"text": "entity", "type": "TYPE", "start": 0, "end": 5}}]

Entity types: PERSON, ORGANIZATION, LOCATION, FACILITY, GPE, ADDRESS

Text: {text}

JSON:"""

        response = self._make_efficient_api_call(prompt, max_tokens=300)
        if response:
            return self._parse_json_response(response, text)
        
        return []

    def _try_structured_prompt(self, text: str, domain_hint: str) -> List[Dict[str, Any]]:
        """Structured prompt as fallback."""
        
        domain_context = self._get_domain_context(domain_hint)
        
        prompt = f"""Analyze this text and extract named entities. Context: {domain_context}

Text: {text}

List entities in this format:
ENTITY_TYPE: entity name
ENTITY_TYPE: another entity

Types to find: PERSON, ORGANIZATION, LOCATION, FACILITY, GPE, ADDRESS

Entities:"""

        response = self._make_efficient_api_call(prompt, max_tokens=200)
        if response:
            return self._parse_structured_response(response, text)
        
        return []

    def _try_simple_prompt(self, text: str) -> List[Dict[str, Any]]:
        """Simple fallback prompt."""
        
        prompt = f"""Find all named entities in this text:

{text}

List them as: EntityName (TYPE)
Types: PERSON, ORGANIZATION, LOCATION, FACILITY, GPE, ADDRESS

Entities:"""

        response = self._make_efficient_api_call(prompt, max_tokens=150)
        if response:
            return self._parse_simple_response(response, text)
        
        return []

    def _get_domain_context(self, domain_hint: str) -> str:
        """Generate generic domain context from hint."""
        if not domain_hint:
            return "general content"
        
        # Generic domain mapping
        domain_contexts = {
            'theatre': 'theatrical and performance venues, actors, directors',
            'academic': 'researchers, universities, institutions',
            'business': 'companies, executives, organizations',
            'historical': 'historical figures, places, events',
            'technical': 'technical terms, organizations, locations',
            'medical': 'medical professionals, institutions, locations',
            'legal': 'legal professionals, courts, organizations'
        }
        
        # Find best match or use hint directly
        for key, context in domain_contexts.items():
            if key.lower() in domain_hint.lower():
                return context
        
        return f"{domain_hint} related content"

    def _make_efficient_api_call(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Make efficient API call with minimal resource usage."""
        
        # Use smaller, more efficient model first
        models_to_try = [
            "google/flan-t5-base",  # Smaller, more efficient
            "google/flan-t5-large", # Fallback
        ]
        
        for model in models_to_try:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": max_tokens,
                        "temperature": 0.1,
                        "do_sample": False  # More deterministic, less computation
                    }
                }
                
                response = requests.post(url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        self.energy_metrics['api_calls_made'] += 1
                        self.energy_metrics['total_tokens_processed'] += len(prompt.split())
                        
                        response_text = result[0].get("generated_text", "")
                        print(f"API call successful with {model}")
                        return response_text
                
                time.sleep(1)  # Brief pause between attempts
                
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        return None

    def _parse_json_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response efficiently."""
        entities = []
        debug_mode = st.session_state.get('debug_mode', False)
        
        if debug_mode:
            st.write("**Debug: Parsing JSON Response**")
        
        # Find JSON in response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            if debug_mode:
                st.write("Found JSON pattern in response")
                st.code(json_match.group())
            
            try:
                parsed = json.loads(json_match.group())
                if debug_mode:
                    st.write(f"Successfully parsed JSON with {len(parsed)} items")
                
                for item in parsed:
                    if isinstance(item, dict) and 'text' in item and 'type' in item:
                        entity_text = item['text'].strip()
                        start_pos = original_text.find(entity_text)
                        if start_pos != -1:
                            entities.append({
                                'text': entity_text,
                                'type': item['type'].upper(),
                                'start': start_pos,
                                'end': start_pos + len(entity_text)
                            })
                        if debug_mode:
                            st.write(f"Found: {entity_text} ({item['type']})")
                        elif debug_mode:
                            st.write(f"Could not locate: {entity_text}")
            except json.JSONDecodeError as e:
                if debug_mode:
                    st.error(f"JSON parsing error: {e}")
        elif debug_mode:
            st.warning("No JSON pattern found in response")
        
        return entities

    def _parse_structured_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse structured response efficiently."""
        entities = []
        debug_mode = st.session_state.get('debug_mode', False)
        
        if debug_mode:
            st.write("**Debug: Parsing Structured Response**")
        
        # Parse "TYPE: entity" format
        pattern = r'([A-Z]+):\s*([^\n]+)'
        matches = re.findall(pattern, response)
        
        if debug_mode:
            st.write(f"Found {len(matches)} structured matches")
        
        for entity_type, entity_text in matches:
            entity_text = entity_text.strip()
            start_pos = original_text.find(entity_text)
            if start_pos != -1:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'start': start_pos,
                    'end': start_pos + len(entity_text)
                })
                if debug_mode:
                    st.write(f"Found: {entity_text} ({entity_type})")
            elif debug_mode:
                st.write(f"Could not locate: {entity_text}")
        
        return entities

    def _parse_simple_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse simple response efficiently."""
        entities = []
        debug_mode = st.session_state.get('debug_mode', False)
        
        if debug_mode:
            st.write("**Debug: Parsing Simple Response**")
        
        # Parse "Entity (TYPE)" format
        pattern = r'([^(]+)\s*\(([^)]+)\)'
        matches = re.findall(pattern, response)
        
        if debug_mode:
            st.write(f"Found {len(matches)} simple matches")
        
        for entity_text, entity_type in matches:
            entity_text = entity_text.strip()
            entity_type = entity_type.strip().upper()
            start_pos = original_text.find(entity_text)
            if start_pos != -1:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'start': start_pos,
                    'end': start_pos + len(entity_text)
                })
                if debug_mode:
                    st.write(f"Found: {entity_text} ({entity_type})")
            elif debug_mode:
                st.write(f"Could not locate: {entity_text}")
        
        return entities

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with Pelagios priority and geographical context detection."""
        import requests
        import time
        
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
                
                # Priority 1: Try Pelagios coordinates first
                if self._try_pelagios_coordinates(entity):
                    continue
                    
                # Priority 2: Try geocoding with context
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Priority 3: Fall back to standard geocoding
                if self._try_python_geocoding(entity):
                    continue
                
                # Priority 4: Try OpenStreetMap
                if self._try_openstreetmap(entity):
                    continue
                    
                # Priority 5: Try aggressive geocoding
                self._try_aggressive_geocoding(entity)
        
        return entities

    def _try_pelagios_coordinates(self, entity):
        """Priority 1: Extract coordinates from Pelagios data if available."""
        
        # Check if entity has Pelagios coordinates
        if entity.get('pelagios_coordinates'):
            coords = entity['pelagios_coordinates']
            if isinstance(coords, list) and len(coords) >= 2:
                # Pelagios typically uses [longitude, latitude] format (GeoJSON standard)
                try:
                    longitude, latitude = coords[0], coords[1]
                    entity['latitude'] = float(latitude)
                    entity['longitude'] = float(longitude)
                    entity['geocoding_source'] = 'pelagios_api'
                    if entity.get('pelagios_title'):
                        entity['location_name'] = entity['pelagios_title']
                    print(f"Using Pelagios coordinates for {entity['text']}: {latitude}, {longitude}")
                    return True
                except (ValueError, IndexError) as e:
                    print(f"Error parsing Pelagios coordinates for {entity['text']}: {e}")
        
        # If no coordinates in Pelagios data, but entity has Pelagios links, 
        # we could try to fetch coordinates from Pelagios API
        if entity.get('pelagios_api_url') and not entity.get('pelagios_coordinates'):
            try:
                # Try to get more detailed info from Pelagios API
                response = requests.get(entity['pelagios_api_url'], timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('geometry', {}).get('coordinates'):
                        coords = data['geometry']['coordinates']
                        if isinstance(coords, list) and len(coords) >= 2:
                            longitude, latitude = coords[0], coords[1]
                            entity['latitude'] = float(latitude)
                            entity['longitude'] = float(longitude)
                            entity['geocoding_source'] = 'pelagios_api_detailed'
                            entity['location_name'] = data.get('title', entity['text'])
                            print(f"Fetched detailed Pelagios coordinates for {entity['text']}: {latitude}, {longitude}")
                            return True
                
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"Failed to fetch detailed Pelagios coordinates for {entity['text']}: {e}")
        
        return False

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text to improve geocoding accuracy."""
        import re
        
        context_clues = []
        text_lower = text.lower()
        
        # Extract major cities/countries mentioned in the text
        major_locations = {
            # Countries
            'uk': ['uk', 'united kingdom', 'britain', 'great britain'],
            'usa': ['usa', 'united states', 'america', 'us '],
            'canada': ['canada'],
            'australia': ['australia'],
            'france': ['france'],
            'germany': ['germany'],
            'italy': ['italy'],
            'spain': ['spain'],
            'japan': ['japan'],
            'china': ['china'],
            'india': ['india'],
            
            # Major cities that provide strong context
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

    def get_sustainability_metrics(self) -> Dict[str, Any]:
        """Get current sustainability metrics."""
        return {
            'api_calls': self.energy_metrics['api_calls_made'],
            'cache_hits': self.energy_metrics['cache_hits'],
            'cache_efficiency': self.energy_metrics['cache_hits'] / max(1, self.energy_metrics['api_calls_made'] + self.energy_metrics['cache_hits']) * 100,
            'tokens_processed': self.energy_metrics['total_tokens_processed']
        }

    def _try_contextual_geocoding(self, entity, context_clues):
        """Priority 2: Try geocoding with geographical context."""
        import requests
        import time
        
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
        """Priority 3: Try Python geocoding libraries (geopy)."""
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
        """Priority 4: Fall back to direct OpenStreetMap Nominatim API."""
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
        """Priority 5: Try more aggressive geocoding with different search terms."""
        import requests
        import time
        
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",  # Add country for UK places
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
        """Efficient Wikidata linking with caching."""
        for entity in entities:
            cache_key = f"wikidata_{entity['text']}"
            if cache_key in st.session_state.prompt_cache:
                entity.update(st.session_state.prompt_cache[cache_key])
                continue
                
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
                        link_data = {
                            'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                            'wikidata_description': result.get('description', '')
                        }
                        entity.update(link_data)
                        st.session_state.prompt_cache[cache_key] = link_data
                
                time.sleep(0.1)
            except Exception:
                pass
        
        return entities

    def link_to_wikipedia(self, entities):
        """Efficient Wikipedia linking with caching."""
        for entity in entities:
            if entity.get('wikidata_url'):
                continue
                
            cache_key = f"wikipedia_{entity['text']}"
            if cache_key in st.session_state.prompt_cache:
                entity.update(st.session_state.prompt_cache[cache_key])
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
                
                response = requests.get(search_url, params=search_params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        import urllib.parse
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        link_data = {
                            'wikipedia_url': f"https://en.wikipedia.org/wiki/{encoded_title}",
                            'wikipedia_title': page_title
                        }
                        entity.update(link_data)
                        st.session_state.prompt_cache[cache_key] = link_data
                
                time.sleep(0.2)
            except Exception:
                pass
        
        return entities

    def link_entity_to_pelagios(self, entity_text):
        """Generate Pleiades search URL for entity."""
        import urllib.parse
        encoded_text = urllib.parse.quote(entity_text)
        url = f"https://pleiades.stoa.org/places/search?SearchableText={encoded_text}"
        return url

    def link_to_pelagios(self, entities):
        """Link entities to Pelagios network using both Pleiades and new API."""
        for entity in entities:
            # Only link geographical/place entities to Pelagios
            if entity['type'] not in ['GPE', 'LOCATION', 'FACILITY', 'ADDRESS']:
                continue
                
            cache_key = f"pelagios_{entity['text']}"
            if cache_key in st.session_state.prompt_cache:
                entity.update(st.session_state.prompt_cache[cache_key])
                continue
            
            # Always add Pleiades search URL
            entity['pleiades_search_url'] = self.link_entity_to_pelagios(entity['text'])
            
            # Try the new Pelagios API
            try:
                api_url = "http://pelagios.dme.ait.ac.at/api"
                search_endpoint = f"{api_url}/search"
                
                params = {
                    'q': entity['text'],
                    'format': 'json',
                    'limit': 1
                }
                
                response = requests.get(search_endpoint, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        result = data[0]
                        link_data = {
                            'pelagios_api_url': result.get('uri', ''),
                            'pelagios_title': result.get('title', ''),
                            'pelagios_description': result.get('description', ''),
                            'pelagios_coordinates': result.get('geometry', {}).get('coordinates', [])
                        }
                        entity.update(link_data)
                        st.session_state.prompt_cache[cache_key] = link_data
                        print(f"Pelagios API link found for {entity['text']}")
                    else:
                        print(f"No Pelagios API results for {entity['text']}")
                else:
                    print(f"Pelagios API request failed with status {response.status_code}")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"Pelagios API linking failed for {entity['text']}: {e}")
                # Still keep the Pleiades search URL even if API fails
                pass
        
        return entities


class StreamlitSustainableApp:
    """Streamlit app for sustainable entity extraction."""
    
    def __init__(self):
        """Initialize the app."""
        self.extractor = SustainablePromptExtractor()
        
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

    def render_header(self):
        """Render the application header with logo."""
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
        
        st.header("Sustainable Entity Extraction")
        st.markdown("**Generic prompt engineering with intelligent caching and minimal energy usage**")
        
        # Show sustainability metrics
        metrics = self.extractor.get_sustainability_metrics()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Calls", metrics['api_calls'])
        with col2:
            st.metric("Cache Hits", metrics['cache_hits'])
        with col3:
            st.metric("Cache Efficiency", f"{metrics['cache_efficiency']:.1f}%")
        with col4:
            st.metric("Tokens Used", metrics['tokens_processed'])

    def render_sidebar(self):
        """Render the sidebar with sustainability info."""
        st.sidebar.subheader("Sustainability Features")
        st.sidebar.info("• Intelligent caching reduces API calls\n• Smaller models for efficiency\n• Generic prompts (no hardcoding)\n• Batch processing when possible")
        
        st.sidebar.subheader("Knowledge Bases")
        st.sidebar.info("Priority linking order:\n• Pelagios API (historical geography)\n• Pleiades Search (ancient places)\n• Wikidata (structured data)\n• Wikipedia (last resort)")
        
        st.sidebar.subheader("Domain Context")
        domain_hint = st.sidebar.selectbox(
            "Optional domain hint for better extraction:",
            ["", "theatre", "academic", "business", "historical", "technical", "medical", "legal"],
            help="Helps the AI understand context without hardcoded patterns"
        )
        
        if 'domain_hint' not in st.session_state:
            st.session_state.domain_hint = ""
        st.session_state.domain_hint = domain_hint
        
        # Add debugging section
        st.sidebar.subheader("Debug Mode")
        debug_mode = st.sidebar.checkbox("Enable Debug Output", help="Shows detailed extraction process")
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        st.session_state.debug_mode = debug_mode

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Sample text
        sample_text = """Recording the Whitechapel Pavilion in 1961. 191-193 Whitechapel Road. theatre. Richard Southern's explanations enabled me to allocate names to the various pieces of apparatus. Since then, we have learned of complete surviving complexes at, for example, Her Majesty's theatre in London, the Citizens in Glasgow and, most importantly, the Tyne theatre in Newcastle, which has been restored to full working order twice by Dr David Wilmore."""
        
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,
            height=200,
            placeholder="Paste your text here for sustainable entity extraction...",
            help="The AI will use efficient prompts to extract entities generically"
        )
        
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
                    if not analysis_title:
                        import os
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title:
            analysis_title = "text_analysis"
        
        return text_input, analysis_title

    def process_text(self, text: str, title: str):
        """Process text with sustainable extraction."""
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Check cache
        import hashlib
        text_hash = hashlib.md5(f"{text}_{st.session_state.domain_hint}".encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text with sustainable AI..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract entities
                status_text.text("Extracting entities with efficient prompts...")
                progress_bar.progress(30)
                
                entities = self.extractor.extract_entities(text, st.session_state.domain_hint)
                
                if not entities:
                    st.warning("No entities were extracted.")
                    return
                
                # Link to knowledge bases
                status_text.text("Linking entities to knowledge bases...")
                progress_bar.progress(60)
                
                entities = self.extractor.link_to_wikidata(entities)
                
                progress_bar.progress(70)
                entities = self.extractor.link_to_wikipedia(entities)
                
                progress_bar.progress(80)
                entities = self.extractor.link_to_pelagios(entities)
                
                # Step 4: Geocoding (after Pelagios linking to use Pelagios coordinates)
                status_text.text("Getting coordinates for locations...")
                progress_bar.progress(90)
                place_entities = [e for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']]
                
                if place_entities:
                    try:
                        geocoded_entities = self.extractor.get_coordinates(place_entities)
                        for geocoded_entity in geocoded_entities:
                            for idx, entity in enumerate(entities):
                                if (entity['text'] == geocoded_entity['text'] and 
                                    entity['type'] == geocoded_entity['type'] and
                                    entity['start'] == geocoded_entity['start']):
                                    entities[idx] = geocoded_entity
                                    break
                    except Exception as e:
                        st.warning(f"Some geocoding failed: {e}")
                
                progress_bar.progress(100)
                
                # Generate visualization
                html_content = self.create_highlighted_html(text, entities)
                
                # Store results
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                progress_bar.empty()
                status_text.empty()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"Found {len(entities)} entities!")
                with col2:
                    st.info("Powered by sustainable AI")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create highlighted HTML visualization."""
        import html as html_module
        
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        highlighted = html_module.escape(text)
        
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = self.extractor.colors.get(entity['type'], '#E7E2D2')
            
            tooltip = f"Type: {entity['type']}"
            if entity.get('wikidata_description'):
                tooltip += f" | {entity['wikidata_description']}"
            
            # Create highlighted span with link (priority: Pelagios API > Pleiades Search > Wikidata > Wikipedia)
            if entity.get('pelagios_api_url'):
                url = html_module.escape(entity["pelagios_api_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('pleiades_search_url'):
                url = html_module.escape(entity["pleiades_search_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{tooltip}">{escaped_entity_text}</span>'
            
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Extract Entities' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entities", len(entities))
        with col2:
            linked_count = len([e for e in entities if e.get('pelagios_api_url') or e.get('pleiades_search_url') or e.get('wikidata_url') or e.get('wikipedia_url')])
            st.metric("Linked Entities", linked_count)
        with col3:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            st.markdown(st.session_state.html_content, unsafe_allow_html=True)
        
        # Entity details
        with st.expander("Entity Details", expanded=False):
            if entities:
                df_data = []
                for entity in entities:
                    links = []
                    if entity.get('pelagios_api_url'):
                        links.append("Pelagios API")
                    if entity.get('pleiades_search_url'):
                        links.append("Pleiades Search")
                    if entity.get('wikidata_url'):
                        links.append("Wikidata")
                    if entity.get('wikipedia_url'):
                        links.append("Wikipedia")
                    
                    row = {
                        'Entity': entity['text'],
                        'Type': entity['type'],
                        'Links': ' | '.join(links) if links else 'None'
                    }
                    if entity.get('pelagios_description'):
                        row['Description'] = entity['pelagios_description'][:100] + "..." if len(entity['pelagios_description']) > 100 else entity['pelagios_description']
                    elif entity.get('wikidata_description'):
                        row['Description'] = entity['wikidata_description'][:100] + "..." if len(entity['wikidata_description']) > 100 else entity['wikidata_description']
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

    def run(self):
        """Main application runner."""
        # Add custom CSS for Farrow & Ball Slipper Satin background and your color scheme
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
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            height: 3rem !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background-color: #B5998A !important;
            color: black !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }
        .stButton > button:active {
            background-color: #A68977 !important;
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.render_header()
        self.render_sidebar()
        
        text_input, analysis_title = self.render_input_section()
        
        if st.button("Extract Entities", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyze.")
        
        st.markdown("---")
        self.render_results()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
            <p>Sustainable AI | Generic Prompt Engineering | Intelligent Caching</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitSustainableApp()
    app.run()


if __name__ == "__main__":
    main()
