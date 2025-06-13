import streamlit as st
import pandas as pd
import base64
import requests
import os

# --- CONFIG ---
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
LLAMAPARSE_ENDPOINT = "https://api.llamaindex.ai/api/parsing/upload"

st.set_page_config(page_title="📊 Financial Statement Analyzer", layout="wide")
st.title("📊 Financial Statement Analyzer with LlamaParse")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a Financial Statement PDF", type="pdf")

def extract_financial_data(text):
    # Very basic keyword-based extraction – replace with LLM chunking/Q&A for accuracy
    lines = text.split('\n')
    results = {}
    keys = ['Revenue', 'Net Income', 'EBITDA', 'Operating Income', 'Total Assets', 'Total Liabilities']
    for key in keys:
        for line in lines:
            if key.lower() in line.lower():
                results[key] = line
                break
    return results

if uploaded_file:
    st.info("Uploading to LlamaParse...")

    pdf_bytes = uploaded_file.read()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Bearer {LLAMAPARSE_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(LLAMAPARSE_ENDPOINT, headers=headers, json={"base64_file": encoded})
    
    if response.status_code == 200:
        doc_id = response.json().get("id")
        st.success("Parsed successfully. Extracting content...")

        # Now retrieve the parsed content
        doc_url = f"https://api.llamaindex.ai/api/parsing/document/{doc_id}"
        doc_resp = requests.get(doc_url, headers=headers)

        if doc_resp.status_code == 200:
            parsed_text = doc_resp.json().get("text", "")
            financial_data = extract_financial_data(parsed_text)

            st.subheader("🔍 Extracted Financial Data")
            df = pd.DataFrame(list(financial_data.items()), columns=["Metric", "Value"])
            st.dataframe(df)

            st.subheader("📃 Raw Extracted Text")
            with st.expander("View Full Text"):
                st.text_area("Text", parsed_text, height=400)

        else:
            st.error("Error fetching parsed content.")
    else:
        st.error(f"Upload failed: {response.text}")
