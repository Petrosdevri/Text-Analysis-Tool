# Importing necessary libraries
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import mplcursors
import plotly.express as px
import plotly.graph_objects as go
from nltk import ne_chunk
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = Flask(__name__, static_url_path='/static')
CORS(app)

app.config['CONTENT_SECURITY_POLICY'] = {
    'default-src': ["'self'"],
    'style-src': ["'self'", 'https://cdn.jsdelivr.net'],
    'script-src': ["'self'", 'https://code.jquery.com', 'https://cdn.jsdelivr.net']
}

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

text = ""

# Function to analyze sentiment
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Function to calculate and display word frequency distribution
def display_word_frequency(text):
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    # Calculate word frequency distribution
    freq_dist = nltk.FreqDist(filtered_tokens)
    # Get top 15 words and their frequencies
    top_words = freq_dist.most_common(15)
    words, frequencies = zip(*top_words)
    return words, frequencies

# Function to generate word cloud
def generate_word_cloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    return wordcloud

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and perform text analysis
@app.route('/analyze', methods=['GET','POST'])
def analyze():
    try:
        global text
        text = request.form['text']
        print(f"Received text: {text}")
        
        # Analyze sentiment
        sentiment_scores = analyze_sentiment(text)
        
        # Display word frequency
        words, frequencies = display_word_frequency(text)
        
        # Generate word cloud
        wordcloud = generate_word_cloud(text)
        
        return jsonify({
            'sentiment_scores': sentiment_scores,
            'word_frequency': {'words': words, 'frequencies': frequencies},
            'wordcloud': wordcloud.to_image()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
