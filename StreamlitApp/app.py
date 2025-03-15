import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Force Transformers to use PyTorch

import streamlit as st
import nltk
import fitz  # PyMuPDF for PDF text extraction
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from summarize import NewsSummarization
import pandas as pd
import altair as alt
import numpy as np
from collections import Counter
import heapq
import re
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from transformers import pipeline
from summarize import *

############################################
# 1. Put Model Loading in a Separate Function
############################################
def load_summarizer(
    primary_model_id: str = "rahuljainx911/t5-small-finetuned-cnn-news",
    fallback_model_id: str = "sshleifer/distilbart-cnn-12-6",
):
    """
    Tries to load the primary model. If it fails, 
    silently falls back to the known summarization model.
    Returns (summarizer, used_fallback).
    """
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(primary_model_id)
        tokenizer = AutoTokenizer.from_pretrained(primary_model_id)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")
        return summarizer, False
    except Exception:
        # Fallback to a known summarization model
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model_id)
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")
        return summarizer, True

############################################
# 2. Use a Spinner While the Model Loads
############################################
with st.spinner("Loading summarization model..."):
    summarizer, used_fallback = load_summarizer()

# Optionally inform the user if fallback was used
if used_fallback:
    st.warning("Using fallback model (DistilBart) instead of the primary model.")

# Streamlit UI
st.write("""# 📰 HIGHLIGHTS! \n### A News Summarizer""")
st.write("Provide a news article or upload a PDF to generate a summary!")

# Display Image
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "newspaper.jpg")

# Try to load the image, and only display if it is found.
try:
    image = Image.open(image_path)
    st.image(image)
except FileNotFoundError:
    st.error("Image not found. Please check the file path.")


# Sidebar options
st.sidebar.header('Summary Settings')
with st.sidebar.form("input_form"):
    st.write('Select summary length for extractive summary')
    max_sentences = st.slider('Summary Length', 1, 10, step=1, value=3)
    st.write('Select word limits for abstractive summary')
    max_words = st.slider('Max words', 50, 500, step=10, value=200)
    min_words = st.slider('Min words', 10, 450, step=10, value=100)
    submit_button = st.form_submit_button("Summarize!")

# User input: text or PDF upload
article = st.text_area("Enter the article:", height=300)
uploaded_file = st.file_uploader("Or upload a PDF", type="pdf")

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file"""
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in pdf_document])
    return text

# If a PDF is uploaded, extract its text
if uploaded_file:
    article = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and text extracted!")

news_summarizer = NewsSummarization()

# Generate summary on button click
if submit_button:
    if article.strip():
        st.write("## 📝 Extractive Summary")
        ex_summary = news_summarizer.extractive_summary(article, num_sentences=max_sentences)
        st.write(ex_summary)

        st.write("## 🔍 Abstractive Summary")
        summary = summarizer(article, max_length=max_words, min_length=min_words, do_sample=False)
        abs_summary = summary[0]['summary_text']
        st.write(abs_summary)
    else:
        st.warning("⚠️ Please enter text or upload a PDF.")

with st.sidebar.expander("More About Summarization"):
    st.markdown("""
    In extractive summarization, we identify important sentences from the article and make a summary by selecting the most important sentences. <br>
    Whereas, for abstractive summarization the model understands the context and generates a summary with the important points using new phrases and language. 
    Abstractive summarization is more similar to the way a human summarizes content. A person might read the entire document, 
    remember a few key points and while writing the summary, will make new sentences that include these points. Abstractive summarization follows the same concept.
    """)
