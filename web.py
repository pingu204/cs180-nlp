"""
Climate Sentiment Text Classification
CS 180 24-25 S2
Team Machine Unlearning
"""

import streamlit as st
import pickle
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def handle_negations(s: str):
    negation_pattern = r'\b(not|no|never|none|cannot|cant|wont|dont)\b[\w\s]+'
    return re.sub(negation_pattern, lambda match: match.group(0).replace(' ', '_'), s)

def clean_text(s: str):
    # Only retain alphanumeric and whitespace characters
    s = re.sub(pattern=rf"|[^a-zA-Z0-9\s]", repl="", string=s, flags=re.IGNORECASE)

    # Convert to lowercase
    s = s.lower()

    # Remove extra whitespaces
    s = re.sub(pattern=r"\s+", repl=" ", string=s).strip()

    return s

def preprocess(text: str):
    return handle_negations(clean_text(text))

def bert_clean_text(s: str):
    s = re.sub(r'[“”]', '"', s)
     
    # Replace all curly single quotes with '
    s = re.sub(r"[‘’]", "'", s)
    
    # Replace en dash and em dash with hyphen
    s = re.sub(r"[–—]", "-", s)

    # Only retain alphanumeric, whitespace characters, single and double quotes, and hyphens
    s = re.sub(pattern=rf"[^a-zA-Z0-9\s\-\'\"]", repl="", string=s, flags=re.IGNORECASE)

    # Remove extra whitespaces
    s = re.sub(pattern=r"\s+", repl=" ", string=s).strip()

    return s

def bert_preprocess(text: str):
    return clean_text(text)

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']

def predict(x, sentiment_scores):
    # Predict on the two models
    mnb_pred = nb_model.predict_proba(x)
    lr_pred = lr_model.predict_proba(np.array(sentiment_scores).reshape(-1,1))

    # Model weights
    mnb_w, lr_w = 0.4, 0.6

    tot_pred = mnb_pred * mnb_w + lr_pred * lr_w

    # Get the index of the highest probability
    # return np.argmax(mnb_pred, axis=1), [max(prob) for prob in mnb_pred]
    return np.argmax(tot_pred, axis=1), [max(prob) for prob in tot_pred]

def bert_predict(text: str):
    text = bert_preprocess(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits  # shape: [1, num_classes] or [num_classes]

    # Ensure logits is 2D for softmax
    probs = torch.softmax(logits, dim=1).squeeze()

    pred_class = int(torch.argmax(probs))
    confidence = max(probs)
    return pred_class, confidence


# Load vectorizer
vectorizer = pickle.load(open("./models/vectorizer.sav", "rb"))
tokenizer = AutoTokenizer.from_pretrained("njlr/cs180-project")

# Load model
nb_model = pickle.load(open("./models/trad_model.sav", "rb"))
lr_model = pickle.load(open("./models/lr_model.sav", "rb"))
bert_model = AutoModelForSequenceClassification.from_pretrained("njlr/cs180-project")

st.title("🍃 Climate Sentiment Analysis")

with st.form("nlp", enter_to_submit=True):
    txt = st.text_area(
        "Text to Classify",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, (...)",
    )

    method = st.pills(
        "Method To Use", ["Traditional", "Deep Learning"], selection_mode="single"
    )

    submitted = st.form_submit_button("Submit")

    if submitted:
        if txt and method:
            st.subheader("Classification")
            class_names = {0: "Risk", 1: "Neutral", 2: "Opportunity"}

            if method == "Traditional":
                test_input = list(map(preprocess, [txt]))
                x_test = vectorizer.transform(test_input).toarray()
                sentiment_score = [sentiment_scores(txt)]
                result, prob = predict(x_test, sentiment_score)
                result, prob = result[0], prob[0]
            elif method == "Deep Learning":
                result, prob = bert_predict(txt)

            match class_names[result]:
                case "Risk":
                    st.error("Risk")
                case "Neutral":
                    st.info("Neutral")
                case "Opportunity":
                    st.success("Opportunity")
                case default:
                    st.warning("Error")
            st.write(f"{prob}")
        else:
            st.write("Please fill in all the required fields.")

st.caption(
    "The goal of this project is to perform sentiment analysis on an expert-annotated dataset containing climate-related paragraphs in corporate disclosures in order to mitigate the negative effects of climate change."
)