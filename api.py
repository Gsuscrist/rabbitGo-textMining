from flask import Flask, request, jsonify
import spacy
from googletrans import Translator
import re
from transformers import pipeline

app = Flask(__name__)

# Initialize the translator
translator = Translator()
# Load English language model from spaCy
nlp = spacy.load("en_core_web_sm")
# Load pre-trained BERT model for sentiment analysis
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define categories and critical terms for classification
categories = {
    'Service': ['service', 'speed', 'latency', 'response', 'application', 'app', 'support', 'performance', 'bug',
                'error', 'crash', 'fail', 'clumsy'],
    'Driver': ['driver', 'conduct', 'attitude', 'behavior', 'manners', 'speed', 'fast', 'slow', 'quick', 'velocity',
               'rude', 'polite', 'courteous'],
    'Traffic Signs': ['traffic', 'signs', 'signal', 'light', 'red light', 'stop sign'],
    'Navigation': ['routes', 'route', 'path', 'directions', 'map', 'navigation', 'location', 'slow'],
    'Stops': ['stops', 'stop', 'station', 'terminal', 'pickup', 'dropoff', 'drop off'],
    'Travel Time': ['time', 'minutes', 'hours', 'delay', 'late', 'slow', 'fast', 'duration', 'wait', 'waiting time'],
    'Behavior': ['attitude', 'driver', 'behavior', 'manners', 'rude', 'polite', 'courteous', 'respectful', 'unfriendly']
}

# Critical terms for each category
critical_terms = {
    'Driver': ['driver'],
    'Navigation': ['route', 'routes'],
    'Travel Time': ['time'],
    # Add more critical terms for other categories if needed
}

# Function to detect profanity in a text
def detect_profanity(text):
    pattern = r'\b(?:cabr[ao]*[nadas]*\d*nes*|cagad[ao]*[das]*\d*s*|ching[ao]*[das]*\d*s*|coñ[oa]*[das]*\d*s*|culer[oa]*[das]*\d*s*|culp[ao]*[das]*\d*s*|desmadre*\d*s*|estúpid[oa]*[deces]*\d*s*|huevón[es]*[adas]*\d*|idiot[ao]*[eses]*\d*s*|jodid[ao]*[s]*\d*s*|madre*\d*s*|mierd[ao]*\d*s*|pendej[oa]*[das]*\d*s*|perr[ao]*[das]*\d*s*|pinch[ea]*\d*s*|put[ao]*\d*s*|verg[ao]*\d*s*)\b'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

# Function to correct grammar before translation
def correct_grammar(text):
    corrected_text = re.sub(r'(?<![{\[])x', 'por', text, flags=re.IGNORECASE)
    corrected_text = re.sub(r'(?<![{\[])k', 'que', corrected_text, flags=re.IGNORECASE)
    corrected_text = re.sub(r'(?<![{\[])q', 'que', corrected_text, flags=re.IGNORECASE)
    return corrected_text

# Function to translate text to English
def translate(text):
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text

# Function to segment text into sentences
def segment_sentences(text):
    doc = nlp(text)
    sentences = []
    start = 0

    for token in doc:
        if token.text in ['.', '!', '?']:
            sentences.append(doc[start:token.i + 1].text.strip())
            start = token.i + 1
        elif token.text in [',', 'and', 'or', 'moreover', 'however']:
            sentences.append(doc[start:token.i].text.strip())
            start = token.i + 1

    if start < len(doc):
        sentences.append(doc[start:len(doc)].text.strip())

    return sentences

# Function to classify tokens into categories
def classify(tokens):
    classifications = set()
    tokens = [token.lower() for token in tokens]

    valid_categories = set(categories.keys())
    for category, critical_terms_list in critical_terms.items():
        if not any(term in tokens for term in critical_terms_list):
            valid_categories.discard(category)

    for token in tokens:
        for category, terms in categories.items():
            if category in valid_categories and token in terms:
                classifications.add(category)

    return list(classifications)

# Function to tokenize a sentence and extract nouns and adjectives
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    if detect_profanity(text):
        return jsonify({'error': 'Text contains profanity'}), 400

    corrected_sentence = correct_grammar(text)
    translated_sentence = translate(corrected_sentence)
    sentences = segment_sentences(translated_sentence)

    total_stars = 0
    total_score = 0

    all_classifications = set()

    for sentence in sentences:
        sentiment = classifier(sentence)
        stars = int(sentiment[0]['label'].split()[0])
        score = sentiment[0]['score']
        total_stars += stars
        total_score += score

        tokens = tokenize(sentence)
        classifications = classify(tokens)
        all_classifications.update(classifications)

    average_stars = total_stars / len(sentences) if sentences else 0
    average_score = total_score / len(sentences) if sentences else 0

    return jsonify({
        'Classifications': list(all_classifications),
        'stars': int(average_stars),
        'score': round(average_score, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
