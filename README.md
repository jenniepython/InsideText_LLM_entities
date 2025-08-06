InsideText_NER

# Streamlit LLM Entity Linker Application

A Streamlit-based web application for named entity extraction and linking using Google's Gemini 1.5 Flash LLM. This tool processes text to identify named entities, linking them to external knowledge sources like Wikidata, Wikipedia, Britannica, and OpenStreetMap, and provides geographic coordinates for location-based entities.

## Features

- **Context-Aware Named Entity Recognition (NER)**: Utilises Google's Gemini 1.5 Flash for precise contextual interpretation.
- **Entity Linking**: Automatically associates extracted entities with relevant knowledge bases.
- **Geocoding**: Retrieves geographic coordinates and descriptive metadata for location entities.
- **Interactive Interface**: User-friendly Streamlit frontend with visual highlighting of entities.
- **Export Options**: Allows data export in JSON-LD and HTML formats.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

## Running the Application

Launch the Streamlit app:

```bash
streamlit run NER_LLM_entities_streamlit.py
```

Open your browser and navigate to `http://localhost:8501`.

## Requirements

- Python 3.10 or higher
- Streamlit
- Gemini API key

## Important Note on API Usage Limits

This application currently uses a shared Gemini API key with a limited quota (free tier). Each query consumes part of this quota, and once the quota is exhausted, no further entity extraction will be possible until the quota resets or a paid plan is activated.

If you encounter issues such as timeouts, missing results, or error messages, it's likely due to the quota being exceeded.

### To use your own Gemini API key:

1. Sign up for access at [Google Generative AI](https://ai.google.dev/).
2. Generate an API key in the API Console.
3. Set your key as an environment variable before running the app:
   ```bash
   export GEMINI_API_KEY='your-own-api-key'
   ```

Using your own key ensures uninterrupted usage and gives you full control over your quota.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

