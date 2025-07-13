
"""
Offline Language Translator using pre-trained models

This script uses the transformers library to perform offline translation
without requiring an internet connection after the initial model download.
"""

import os
import argparse
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Dictionary of supported languages
LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'de': 'German',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic'
}

# Dictionary to map language pairs to models
LANGUAGE_MODELS = {
    'en-es': 'Helsinki-NLP/opus-mt-en-es',
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
    'en-it': 'Helsinki-NLP/opus-mt-en-it',
    'en-pt': 'Helsinki-NLP/opus-mt-en-pt',
    'en-de': 'Helsinki-NLP/opus-mt-en-de',
    'es-en': 'Helsinki-NLP/opus-mt-es-en',
    'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
    'it-en': 'Helsinki-NLP/opus-mt-it-en',
    'pt-en': 'Helsinki-NLP/opus-mt-en-ROMANCE',
    'de-en': 'Helsinki-NLP/opus-mt-de-en',
    'en-ROMANCE': 'Helsinki-NLP/opus-mt-en-ROMANCE',  # This covers Spanish, French, Italian, Portuguese
}

def display_languages():
    """Display available languages."""
    print("\nAvailable languages:")
    for code, name in LANGUAGES.items():
        print(f"{code}: {name}")

def get_model_name(src_lang, tgt_lang):
    """Get the appropriate model name for the language pair."""
    lang_pair = f"{src_lang}-{tgt_lang}"
    
    if lang_pair in LANGUAGE_MODELS:
        return LANGUAGE_MODELS[lang_pair]
    
    # If specific pair not found, try to use a more general model
    if tgt_lang in ['es', 'fr', 'it', 'pt'] and src_lang == 'en':
        return LANGUAGE_MODELS['en-ROMANCE']
    
    return None

def translate_text(text, src_lang='en', tgt_lang='es', use_cached=True):
    """
    Translate text from source language to target language.
    
    Args:
        text (str): Text to translate
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        use_cached (bool): Whether to use cached models
        
    Returns:
        str: Translated text
    """
    model_name = get_model_name(src_lang, tgt_lang)
    
    if not model_name:
        return f"Translation from {LANGUAGES.get(src_lang, src_lang)} to {LANGUAGES.get(tgt_lang, tgt_lang)} is not supported."
    
    try:
        # Create directories for caching
        if use_cached:
            os.makedirs('models_cache', exist_ok=True)
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'models_cache')
            
        print(f"Loading model for {src_lang} to {tgt_lang} translation...")
        translator = pipeline("translation", model=model_name)
        
        print("Translating...")
        result = translator(text, max_length=512)
        
        return result[0]['translation_text']
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def interactive_mode():
    """Run the translator in interactive mode."""
    print("\n=== Offline Language Translator ===")
    print("(Models will be downloaded on first use and then cached for offline use)")
    
    display_languages()
    
    while True:
        print("\n" + "-"*50)
        text = input("Enter text to translate (or 'q' to quit): ")
        
        if text.lower() == 'q':
            break
            
        src_lang = input(f"Enter source language code [{', '.join(LANGUAGES.keys())}] (default: en): ").strip()
        if not src_lang:
            src_lang = 'en'
            
        tgt_lang = input(f"Enter target language code [{', '.join(LANGUAGES.keys())}] (default: es): ").strip()
        if not tgt_lang:
            tgt_lang = 'es'
            
        if src_lang not in LANGUAGES or tgt_lang not in LANGUAGES:
            print("Invalid language code. Please try again.")
            continue
            
        translation = translate_text(text, src_lang, tgt_lang)
        print(f"\nTranslated text ({LANGUAGES.get(tgt_lang, tgt_lang)}):")
        print(f">>> {translation}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Offline Language Translator')
    parser.add_argument('--text', type=str, help='Text to translate')
    parser.add_argument('--src', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt', type=str, default='es', help='Target language code')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or not args.text:
        interactive_mode()
    else:
        translation = translate_text(args.text, args.src, args.tgt)
        print(f"Original ({LANGUAGES.get(args.src, args.src)}): {args.text}")
        print(f"Translated ({LANGUAGES.get(args.tgt, args.tgt)}): {translation}")

if __name__ == "__main__":
    main()