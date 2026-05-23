import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time

# -----------------------------
# Page setting
# -----------------------------
st.set_page_config(
    page_title="AI Audience Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Audience Review Analyzer for Netflix Content Evaluation")

st.write(
    "This application uses deep learning models to classify audience review sentiment "
    "and summarize key feedback for content evaluation."
)

# -----------------------------
# Model name
# Replace this with your own Hugging Face model URL/name if needed
# -----------------------------
MODEL_NAME = "Zhu199/distilbert-imdb-sentiment"

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

@st.cache_resource
def load_summarization_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

sentiment_pipeline = load_sentiment_model()
summarization_pipeline = load_summarization_model()


# -----------------------------
# Helper functions
# -----------------------------
def map_sentiment_label(label):
    """
    Convert model labels into readable sentiment names.
    IMDB fine-tuned DistilBERT usually returns LABEL_0 and LABEL_1.
    LABEL_0 = Negative
    LABEL_1 = Positive
    """
    label = label.upper()

    if label == "LABEL_0" or "NEG" in label:
        return "Negative"
    elif label == "LABEL_1" or "POS" in label:
        return "Positive"
    else:
        return label


def generate_business_suggestion(sentiment):
    """
    Generate simple business suggestions based on sentiment.
    """
    if sentiment == "Negative":
        return (
            "This review indicates negative audience feedback. "
            "The platform should monitor similar reviews and review the content quality "
            "before heavy promotion."
        )
    else:
        return (
            "This review indicates positive audience feedback. "
            "The content may be suitable for further recommendation or promotion."
        )


def summarize_review(text):
    """
    Summarize long reviews.
    For short reviews, return the original text directly.
    """
    word_count = len(text.split())

    if word_count < 30:
        return text

    # Limit input length to avoid Streamlit Cloud memory issues
    text = text[:1500]

    try:
        summary = summarization_pipeline(
            text,
            max_length=60,
            min_length=15,
            do_sample=False
        )[0]["summary_text"]

        return summary

    except Exception:
        return "Summary could not be generated for this review."


def analyze_review(text):
    """
    Main function:
    1. Sentiment classification
    2. Summarization
    3. Business suggestion
    4. Runtime calculation
    """
    start_time = time.time()

    # Limit input length for sentiment model
    sentiment_result = sentiment_pipeline(text[:1000])[0]

    raw_label = sentiment_result["label"]
    confidence = sentiment_result["score"]

    sentiment = map_sentiment_label(raw_label)
    summary = summarize_review(text)
    suggestion = generate_business_suggestion(sentiment)

    runtime = time.time() - start_time

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "summary": summary,
        "suggestion": suggestion,
        "runtime": runtime
    }


# -----------------------------
# Streamlit UI
# -----------------------------
tab1, tab2 = st.tabs(["Single Review Analysis", "Batch CSV Analysis"])


# -----------------------------
# Tab 1: Single review
# -----------------------------
with tab1:
    st.subheader("Analyze One Audience Review")

    review_text = st.text_area(
        "Enter a movie or TV review:",
        height=200,
        placeholder="Example: The movie was too slow and boring, although the acting was good."
    )

    if st.button("Analyze Review"):
        if review_text.strip() == "":
            st.warning("Please enter a review text.")
        else:
            result = analyze_review(review_text)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", result["sentiment"])

            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")

            with col3:
                st.metric("Runtime", f"{result['runtime']:.2f}s")

            st.subheader("Review Summary")
            st.write(result["summary"])

            st.subheader("Business Suggestion")
            st.write(result["suggestion"])


# -----------------------------
# Tab 2: Batch CSV analysis
# -----------------------------
with tab2:
    st.subheader("Analyze Reviews from CSV")

    st.write("Upload a CSV file with a column named `review`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("The CSV file must contain a column named `review`.")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            max_rows = st.slider(
                "Number of reviews to analyze",
                min_value=1,
                max_value=min(len(df), 50),
                value=min(len(df), 10)
            )

            if st.button("Run Batch Analysis"):
                results = []

                progress_bar = st.progress(0)

                selected_df = df.head(max_rows)

                for i, review in enumerate(selected_df["review"].astype(str)):
                    result = analyze_review(review)

                    results.append({
                        "review": review,
                        "sentiment": result["sentiment"],
                        "confidence": result["confidence"],
                        "summary": result["summary"],
                        "suggestion": result["suggestion"],
                        "runtime": result["runtime"]
                    })

                    progress_bar.progress((i + 1) / len(selected_df))

                result_df = pd.DataFrame(results)

                st.subheader("Batch Analysis Results")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="review_analysis_results.csv",
                    mime="text/csv"
                )