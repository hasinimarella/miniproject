import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
from typing import Dict, List, Tuple
import warnings

# Try to import transformers, but make it optional for compatibility
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception as e:
    HAS_TRANSFORMERS = False
    print(f"Warning: Transformers not available: {e}")

# Try to import googletrans with fallback
try:
    from google_trans_new import google_translator
    Translator = google_translator
    HAS_GOOGLETRANS = True
except:
    try:
        from googletrans import Translator
        HAS_GOOGLETRANS = True
    except:
        Translator = None
        HAS_GOOGLETRANS = False

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        print("Warning: Could not download vader_lexicon")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        print("Warning: Could not download punkt tokenizer")


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis using multiple NLP techniques
    """
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader = SentimentIntensityAnalyzer()
        self.translator = None
        
        # Initialize translator
        if HAS_GOOGLETRANS:
            try:
                self.translator = Translator()
            except Exception as e:
                print(f"Warning: Could not initialize translator: {e}")
        
        # Initialize transformer-based sentiment analysis
        self.transformer = None
        if HAS_TRANSFORMERS:
            try:
                self.transformer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text
        Returns language code
        """
        try:
            if self.translator and HAS_GOOGLETRANS:
                result = self.translator.detect(text)
                return result.lang if result else 'en'
        except Exception as e:
            pass
        
        # Fallback: assume English if translation fails
        return 'en'
    
    def translate_to_english(self, text: str, source_lang: str = None) -> str:
        """
        Translate text to English for analysis
        """
        try:
            # If source language is English, return as-is
            if source_lang == 'en':
                return text
            
            # Try to detect language if not provided
            if source_lang is None and self.translator:
                try:
                    source_lang = self.detect_language(text)
                except:
                    source_lang = 'en'
            
            # If it's English after detection, return original
            if source_lang == 'en' or source_lang is None:
                return text
            
            # Try translation if we have a translator
            if self.translator and HAS_GOOGLETRANS:
                try:
                    translated = self.translator.translate(text, src_lang=source_lang, dest_lang='en')
                    return translated.get('text', text) if isinstance(translated, dict) else str(translated)
                except:
                    pass
        except Exception as e:
            pass
        
        # If translation fails, return original text
        return text
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
        Returns compound score and individual sentiment components
        """
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        Returns polarity and subjectivity
        """
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using transformer-based model
        """
        if self.transformer is None:
            return {'positive': 0.5, 'negative': 0.5, 'score': 0.0}
        
        try:
            result = self.transformer(text[:512])[0]  # Truncate for model limit
            label = result['label'].lower()
            score = result['score']
            
            return {
                'label': label,
                'score': score if label == 'positive' else -score
            }
        except Exception as e:
            print(f"Transformer analysis error: {e}")
            return {'label': 'neutral', 'score': 0.0}
    
    def extract_emotions(self, text: str) -> Dict[str, float]:
        """
        Extract specific emotions from text
        Maps sentiment to specific emotions: joy, sadness, anger, fear, surprise, trust
        """
        emotion_keywords = {
            'joy': ['happy', 'wonderful', 'excellent', 'great', 'amazing', 'love', 'perfect'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'terrible'],
            'anger': ['angry', 'furious', 'outraged', 'frustrated', 'irritated', 'disgusted'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'unexpected'],
            'trust': ['trust', 'reliable', 'dependable', 'safe', 'secure', 'confident']
        }
        
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(count / len(keywords), 1.0)
        
        return emotions
    
    def identify_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key sentiment-bearing words from text
        """
        try:
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    try:
                        nltk.download('stopwords', quiet=True)
                        stop_words = set(stopwords.words('english'))
                    except:
                        stop_words = set()
                
                tokens = word_tokenize(text.lower())
                keywords = [
                    word for word in tokens 
                    if word.isalpha() and word not in stop_words and len(word) > 3
                ]
                return keywords[:top_n]
            except:
                # Fallback: simple word splitting if NLTK fails
                words = text.lower().split()
                keywords = [
                    word for word in words
                    if word.isalpha() and len(word) > 3
                ]
                return keywords[:top_n]
        except:
            # Final fallback: return empty list
            return []
    
    def comprehensive_analysis(self, text: str) -> Dict:
        """
        Perform comprehensive sentiment analysis
        Combines multiple techniques for robust results
        """
        # Detect language and translate if needed
        original_language = self.detect_language(text)
        english_text = self.translate_to_english(text, original_language)
        
        # VADER analysis
        vader_scores = self.analyze_with_vader(english_text)
        
        # TextBlob analysis
        textblob_scores = self.analyze_with_textblob(english_text)
        
        # Transformer analysis
        transformer_scores = self.analyze_with_transformer(english_text)
        
        # Extract emotions
        emotions = self.extract_emotions(english_text)
        
        # Extract keywords
        keywords = self.identify_keywords(english_text)
        
        # Calculate overall sentiment score (normalized to -1 to 1)
        overall_score = (
            vader_scores['compound'] * 0.4 +
            textblob_scores['polarity'] * 0.3 +
            transformer_scores.get('score', 0) * 0.3
        )
        
        # Determine sentiment label
        # Small rule-based fallback: look for explicit negative keywords
        negative_keywords = [
            'bad','poor','terrible','worst','horrible','disappointed','awful',
            'pain','delay','slow','rude','angry','error','problem','issue','needs improvement'
        ]
        neg_count = sum(1 for kw in negative_keywords if kw in english_text.lower())

        # If clear negative signals from lexicons or keywords, bias toward NEGATIVE
        if overall_score > 0.3:
            sentiment_label = 'POSITIVE'
        elif overall_score < -0.3 or (neg_count > 0 and (vader_scores['compound'] < 0 or textblob_scores['polarity'] < 0)):
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'original_language': original_language,
            'translated_text': english_text,
            'overall_score': round(overall_score, 3),
            'sentiment_label': sentiment_label,
            'confidence': round(abs(overall_score), 3),
            'vader': vader_scores,
            'textblob': textblob_scores,
            'transformer': transformer_scores,
            'emotions': emotions,
            'dominant_emotion': max(emotions, key=emotions.get),
            'keywords': keywords,
            'subjectivity': round(textblob_scores['subjectivity'], 3)
        }


class MultilingualSentimentAnalyzer(SentimentAnalyzer):
    """
    Extended sentiment analyzer with advanced multilingual support
    """
    
    def __init__(self):
        super().__init__()
        self.language_codes = {
            'english': 'en', 'spanish': 'es', 'french': 'fr',
            'german': 'de', 'italian': 'it', 'portuguese': 'pt',
            'hindi': 'hi', 'arabic': 'ar', 'chinese': 'zh', 'japanese': 'ja'
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts efficiently
        """
        results = []
        for text in texts:
            results.append(self.comprehensive_analysis(text))
        return results
    
    def sentiment_distribution(self, analyses: List[Dict]) -> Dict:
        """
        Calculate sentiment distribution across multiple analyses
        """
        positive = sum(1 for a in analyses if a['sentiment_label'] == 'POSITIVE')
        negative = sum(1 for a in analyses if a['sentiment_label'] == 'NEGATIVE')
        neutral = sum(1 for a in analyses if a['sentiment_label'] == 'NEUTRAL')
        total = len(analyses)
        
        return {
            'positive_percentage': round(positive / total * 100, 2) if total > 0 else 0,
            'negative_percentage': round(negative / total * 100, 2) if total > 0 else 0,
            'neutral_percentage': round(neutral / total * 100, 2) if total > 0 else 0,
            'total_reviews': total,
            'average_score': round(np.mean([a['overall_score'] for a in analyses]), 3)
        }
    
    def identify_critical_issues(self, text: str, threshold: float = -0.7) -> bool:
        """
        Identify if review contains critical issues based on sentiment score
        """
        analysis = self.comprehensive_analysis(text)
        return analysis['overall_score'] < threshold


# Initialize global analyzer instance
sentiment_analyzer = MultilingualSentimentAnalyzer()
