# AI Audience Review Analyzer for Netflix Content Evaluation

This project builds a Streamlit application for audience review analysis. It uses deep learning models to classify movie or TV review sentiment and summarize key audience feedback for content evaluation.

## Business Objective

The application helps streaming platforms such as Netflix analyze large-scale audience reviews automatically. It classifies reviews into positive or negative sentiment, summarizes key feedback, and provides simple business suggestions.

## Main Functions

1. Single review sentiment analysis
2. Review summarization
3. Business suggestion generation
4. Batch CSV review analysis
5. Downloadable analysis results

## Model Pipeline

1. Fine-tuned DistilBERT model for sentiment classification
2. DistilBART summarization pipeline for review summarization

## Dataset

The sentiment classification model was fine-tuned on the IMDB Dataset of 50K Movie Reviews.

## Model

Fine-tuned model: Zhu199/distilbert-imdb-sentiment

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
