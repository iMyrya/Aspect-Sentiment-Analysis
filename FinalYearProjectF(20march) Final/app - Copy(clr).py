import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SimpleRNN, BatchNormalization
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
import os

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Analyze movie reviews by different aspects")

# Constants
MAX_LEN = 100
MAX_WORDS = 20000

# Updated aspect keywords from the training files
aspect_keywords = {
    "acting": [
        "acting", "actor", "actors", "performance", "performances", "cast", "role", "roles",
        "portrayal", "expressions", "dialogue", "screen presence", "chemistry",
        "delivery", "facial expressions", "body language", "acting skills", "actress", "portrayed", "played", "character"
    ],
    "plot": [
        "plot", "story", "storyline", "script", "twist", "twists", "narrative", "theme",
        "screenplay", "structure", "pace", "climax", "intro", "ending", "build-up", "backstory",
        "story arc", "writing", "subplots", "conflict", "chronology", "plotline", "pacing"
    ],
    "sound": [
        "sound", "music", "score", "audio", "background score", "background music",
        "track", "melody", "tune", "composition", "soundtrack", "beats", "rhythm", "bass",
        "volume", "sound effects", "sound quality", "mixing", "sfx", "song", "composer"
    ],
    "visuals": [
        "visuals", "cinematography", "cgi", "graphics", "animation", "camera", "effects",
        "focus", "lighting", "color", "color grading", "frame", "shot", "angles", "vfx",
        "aesthetics", "scenery", "set design", "visual appeal", "cinematic", "cinematic style", 
        "visual", "scene", "imagery", "picture quality", "cinematographer", "beautiful"
    ],
    "direction": [
        "direction", "director", "directing", "vision", "filmmaking", "execution", "style",
        "tone", "narrative flow", "pacing", "creative choices", "scene setup", "visionary",
        "command", "directorial style", "cohesion", "presentation", "scene composition", "creative", "artistic"
    ],
    "thriller": [
        "thriller", "thrilling", "thrills", "suspense", "tension", "mystery", "crime",
        "investigation", "detective", "clues", "twists", "twist", "shock", "surprise",
        "mind-bending", "edge of the seat", "on edge", "plot twist", "unexpected", "unexpected twists"
    ],
    "overall": ["movie", "film", "overall", "recommend", "worth", "watching", "experience"]  # General terms
}

@st.cache_resource
def create_and_load_model(model_type="LSTM"):
    """Create model architecture and load weights separately based on model type"""
    # Create the model architecture based on selected model type
    if model_type == "CNN":
        # Updated CNN model based on cnnTrain.py
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            Dropout(0.4),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.4),
            Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
    elif model_type == "RNN":
        # Updated RNN model based on rnnTrain.py
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            Bidirectional(SimpleRNN(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.4),
            Bidirectional(SimpleRNN(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
    else:  # Default LSTM based on lstmTrain.py
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.4),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Build the model with a dummy input so the layers are created
    dummy_input = np.zeros((1, MAX_LEN))
    model.predict(dummy_input)
    
    # Get model file path based on model type
    model_path = f"Model/{model_type.lower()}_aspect_sentiment_model.h5"
    
    # Try to load weights
    try:
        # Try to load model weights
        if os.path.exists(model_path):
            model.load_weights(model_path)
            st.sidebar.success(f"Successfully loaded {model_type} model weights!")
        else:
            st.sidebar.warning(f"{model_type} model file not found at {model_path}. Using uninitialized model.")
    except Exception as e:
        st.sidebar.warning(f"Couldn't load {model_type} model weights: {str(e)}. Using uninitialized model.")
    
    return model

@st.cache_resource
def load_tokenizer(model_type="LSTM"):
    """Load tokenizer based on model type"""
    # Determine file path based on model type (lowercase for consistency)
    tokenizer_path = f"Model/{model_type.lower()}_tokenizer.pkl"
    
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
            st.sidebar.success(f"Successfully loaded {model_type} tokenizer!")
            return tokenizer
    except FileNotFoundError:
        st.sidebar.error(f"Failed to load tokenizer: File not found at {tokenizer_path}")
        return None
    except Exception as e:
        st.sidebar.error(f"Failed to load tokenizer: {str(e)}")
        return None

@st.cache_resource(hash_funcs={type: id})
def load_label_encoder(model_type="LSTM"):
    """Load label encoder based on model type"""
    # Determine file path based on model type (lowercase for consistency)
    encoder_path = f"Model/{model_type.lower()}_label_encoders.pkl"
    
    try:
        with open(encoder_path, "rb") as handle:
            label_encoder = pickle.load(handle)
            st.sidebar.success(f"Successfully loaded {model_type} label encoder!")
            return label_encoder
    except Exception as e:
        st.sidebar.warning(f"Couldn't load {model_type} label encoder: {str(e)}. Using default labels.")
        return None

def extract_aspect_text(review, aspect):
    """Improved aspect text extraction based on training files"""
    review_lower = review.lower()
    
    # Get the keywords for the specified aspect
    keywords = aspect_keywords.get(aspect.lower(), [])
    
    # For "overall", always return the full review
    if aspect.lower() == "overall":
        return review
        
    # Split on sentence boundaries and contrasting conjunctions
    splitters = r'[.!?;]| but | and | although | however | though | while '
    sentences = re.split(splitters, review_lower)
    matched = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and any(keyword in sentence for keyword in keywords):
            matched.append(sentence)
    
    # Return matched sentences or full review as fallback
    return " ".join(matched) if matched else review

def detect_aspect_in_review(review, aspect):
    """Detect if an aspect is mentioned in the review"""
    review_lower = review.lower()
    
    # Get the keywords for the specified aspect
    keywords = aspect_keywords.get(aspect.lower(), [])
    
    # For "overall", always return true
    if aspect.lower() == "overall":
        return True
        
    # Check for direct mentions
    for keyword in keywords:
        if keyword in review_lower:
            # Find the sentence containing this keyword
            sentences = re.split(r'[.!?]', review_lower)
            for sentence in sentences:
                if keyword in sentence:
                    return True
    
    # If no direct mention, return False
    return False

def analyze_mixed_sentiments(review):
    """Analyze a review for mixed sentiments across different aspects"""
    results = {}
    for aspect, keywords in aspect_keywords.items():
        if aspect == "overall":
            continue
            
        # Check if aspect is mentioned
        is_mentioned = detect_aspect_in_review(review, aspect)
        if is_mentioned:
            # Extract aspect-specific text
            aspect_text = extract_aspect_text(review, aspect)
            results[aspect] = {
                "mentioned": True,
                "text": aspect_text,
                # We'll predict sentiment later
            }
        else:
            results[aspect] = {"mentioned": False}
    
    # Always include overall
    results["overall"] = {
        "mentioned": True,
        "text": review
    }
    
    return results

def predict_aspect_sentiment(text, model, tokenizer, label_encoder=None):
    """Predict sentiment for a specific text"""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Get prediction
    prediction = model.predict(padded_sequence)[0]
    predicted_class = np.argmax(prediction)
    
    # Map to sentiment
    if label_encoder is not None:
        try:
            # Handle different formats of label_encoder
            if isinstance(label_encoder, tuple):
                # This is for the models where we have (label_encoder, aspect_encoder)
                sentiments = label_encoder[0].inverse_transform([0, 1, 2])
            else:
                sentiments = label_encoder.inverse_transform([0, 1, 2])
        except:
            sentiments = ["Negative", "Neutral", "Positive"]
    else:
        sentiments = ["Negative", "Neutral", "Positive"]
    
    sentiment = sentiments[predicted_class]
    confidence = prediction[predicted_class]
    
    return sentiment, confidence

# Initialize session state for the sample review
if 'sample_review_clicked' not in st.session_state:
    st.session_state.sample_review_clicked = False

# Main app
try:
    # Add App Options in sidebar
    st.sidebar.header("App Options")
    use_mock_data = st.sidebar.checkbox("Use mock predictions (for testing)", value=False)
    
    # Add model selection dropdown to sidebar
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=["LSTM", "CNN", "RNN"],
        index=0  # Default to LSTM
    )
    
    # Display a note about the model
    st.sidebar.info(f"Using {model_type} model for sentiment analysis")
    
    # Clear cache if model type changes to ensure new model is loaded
    if 'previous_model_type' not in st.session_state:
        st.session_state.previous_model_type = model_type
    elif st.session_state.previous_model_type != model_type:
        # Model type has changed, update the session state
        st.session_state.previous_model_type = model_type
    
    # Load model and tokenizer based on selected model
    model = create_and_load_model(model_type)
    tokenizer = load_tokenizer(model_type)
    label_encoder = load_label_encoder(model_type)
    
    # Sample review button (before text area)
    if st.button("Try a sample mixed review"):
        st.session_state.sample_review_clicked = True
        
    # Set up the UI
    st.markdown("### Enter your movie review")
    
    # Display sample review if button was clicked
    initial_value = ""
    if st.session_state.sample_review_clicked:
        initial_value = "The cinematography was breathtaking with stunning visuals, but the acting felt a bit forced and unconvincing. The plot was predictable but entertaining enough."
    
    review = st.text_area("", height=150, value=initial_value,
                          placeholder="Example: The cinematography was breathtaking, but the acting felt a bit forced.")
    
    if review:
        # Get available aspects from keywords
        available_aspects = list(aspect_keywords.keys())
        
        # Auto-detect aspects and analyze
        st.markdown("### Analysis Results")
        
        # If review is empty, don't proceed
        if not review.strip():
            st.warning("Please enter a review to analyze")
        else:
            aspect_analysis = analyze_mixed_sentiments(review)
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Aspect-by-Aspect Analysis", "Mixed Sentiment Detection"])
            
            with tab1:
                # Let user select the aspect to analyze
                aspect = st.selectbox("Select specific aspect to analyze:", available_aspects)
                
                # Check if aspect is detected
                if aspect_analysis[aspect.lower()].get("mentioned", False):
                    text_to_analyze = aspect_analysis[aspect.lower()]["text"]
                    
                    # Sentiment prediction
                    if use_mock_data:
                        # For testing - mock predictions
                        if aspect.lower() == "visuals" and "breathtaking" in review.lower():
                            sentiment, confidence = "Positive", 0.85
                        elif aspect.lower() == "acting" and "forced" in review.lower():
                            sentiment, confidence = "Negative", 0.78
                        elif aspect.lower() == "overall" and ("breathtaking" in review.lower() and "forced" in review.lower()):
                            sentiment, confidence = "Neutral", 0.65
                        else:
                            sentiment, confidence = "Neutral", 0.60
                    else:
                        # Real prediction using the selected model
                        sentiment, confidence = predict_aspect_sentiment(text_to_analyze, model, tokenizer, label_encoder)
                    
                    # Display result
                    if sentiment == "Positive":
                        st.markdown(f"### 😄 {aspect.capitalize()} Sentiment: {sentiment}")
                        color = "green"
                    elif sentiment == "Negative":
                        st.markdown(f"### 😞 {aspect.capitalize()} Sentiment: {sentiment}")
                        color = "red"
                    else:
                        st.markdown(f"### 😐 {aspect.capitalize()} Sentiment: {sentiment}")
                        color = "orange"
                    
                    # Display confidence with appropriate color and model type
                    st.markdown(f"<h4 style='color:{color};'>Confidence: {confidence*100:.1f}% (using {model_type} model)</h4>", unsafe_allow_html=True)
                    
                    # Show what part of the review was analyzed
                    st.markdown("**Text analyzed:**")
                    st.info(text_to_analyze)
                else:
                    st.warning(f"This review doesn't mention the '{aspect}' aspect.")
                    
            with tab2:
                st.markdown("### Mixed Sentiment Detection")
                
                # Analyze each aspect
                mentioned_aspects = []
                
                for aspect, data in aspect_analysis.items():
                    if data.get("mentioned", False):
                        mentioned_aspects.append(aspect)
                
                if len(mentioned_aspects) > 1:  # If multiple aspects detected
                    st.success(f"Detected {len(mentioned_aspects)} aspects in your review (analyzing with {model_type} model):")
                    
                    for aspect in mentioned_aspects:
                        text_to_analyze = aspect_analysis[aspect]["text"]
                        
                        # Predict for each aspect
                        if use_mock_data:
                            # Mock data for testing
                            if aspect == "visuals" and "breathtaking" in review.lower():
                                sentiment, confidence = "Positive", 0.85
                            elif aspect == "acting" and "forced" in review.lower():
                                sentiment, confidence = "Negative", 0.78
                            elif aspect == "overall":
                                sentiment, confidence = "Neutral", 0.65
                            else:
                                sentiment, confidence = "Neutral", 0.60
                        else:
                            # Real prediction using the selected model
                            sentiment, confidence = predict_aspect_sentiment(text_to_analyze, model, tokenizer, label_encoder)
                        
                        # Display color-coded results
                        if sentiment == "Positive":
                            emoji = "😄"
                            color = "green"
                        elif sentiment == "Negative":
                            emoji = "😞"
                            color = "red"
                        else:
                            emoji = "😐"
                            color = "orange"
                            
                        st.markdown(f"<div style='padding:10px; border-left:5px solid {color};'>"
                                    f"<b>{aspect.capitalize()}</b>: {emoji} {sentiment} ({confidence*100:.1f}%)</div>", 
                                    unsafe_allow_html=True)
                else:
                    st.info("Only one aspect detected in this review. Try using a more detailed review that mentions multiple aspects.")

except Exception as e:
    st.error(f"Error setting up the application: {str(e)}")
    st.info(f"""
    ### Troubleshooting
    
    This app needs:
    1. A properly trained model file ({model_type.lower()}_aspect_sentiment_model.h5) or "Use mock predictions" enabled in the sidebar
    2. The {model_type.lower()}_tokenizer.pkl file for the selected model
    3. The {model_type.lower()}_label_encoders.pkl file for the selected model
    
    Make sure these files are in the correct Model directory.
    
    Expected file paths:
    - Model/{model_type.lower()}_aspect_sentiment_model.h5
    - Model/{model_type.lower()}_tokenizer.pkl
    - Model/{model_type.lower()}_label_encoders.pkl
    
    If you're having issues with the model, enable "Use mock predictions" in the sidebar for testing.
    """)