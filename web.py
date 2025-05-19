"""
Climate Sentiment Text Classification
CS 180 24-25 S2
Team Machine Unlearning
"""

import streamlit as st
import pickle
import re


# Data preprocessing
def clean_text(s: str):
    # Only retain alphanumeric and whitespace characters
    s = re.sub(pattern=rf"|[^a-zA-Z0-9\s]", repl="", string=s, flags=re.IGNORECASE)

    # Convert to lowercase
    s = s.lower()

    # Remove extra whitespaces
    s = re.sub(pattern=r"\s+", repl=" ", string=s).strip()

    return s


def preprocess(text: str):
    return clean_text(text)


# Load vectorizer
vectorizer = pickle.load(open("vectorizer.sav", "rb"))

# Load model
trad_model = pickle.load(open("model.sav", "rb"))

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

            test_input = list(map(preprocess, [txt]))

            result = trad_model.predict(vectorizer.transform(test_input))[0]

            class_names = {0: "Risk", 1: "Neutral", 2: "Opportunity"}

            match class_names[result]:
                case "Risk":
                    st.error("Risk")
                case "Neutral":
                    st.info("Neutral")
                case "Opportunity":
                    st.success("Gowch")
                case default:
                    st.warning("Error")
        else:
            st.write("Please fill in all the required fields.")

st.caption(
    "The goal of this project is to perform sentiment analysis on an expert-annotated dataset containing climate-related paragraphs in corporate disclosures in order to mitigate the negative effects of climate change."
)
