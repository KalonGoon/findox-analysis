import streamlit as st
import pandas as pd
import base64
import requests
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
LLAMAPARSE_API_KEY = os.getenv("llx-kQRtQBOWUW4xSwTqjKMNiASQM2caVUT6ly1S7MUdbTsurYEp")
LLAMAPARSE_ENDPOINT = "https://api.llamaindex.ai/api/parsing/upload"
DOC_RETRIEVE_ENDPOINT = "https://api.llamaindex.ai/api/parsing/document/"
GROQ_API_KEY = os.getenv("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")

st.set_page_config(page_title="📊 Financial Statement Analyzer", layout="wide")
st.title("📊 Multi-Year Financial Statement Analyzer with LLM")

# --- Upload Multiple PDFs ---
uploaded_files = st.file_uploader("Upload up to 5 Financial Statement PDFs", type="pdf", accept_multiple_files=True)

def upload_and_parse(file):
    pdf_bytes = file.read()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {LLAMAPARSE_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(LLAMAPARSE_ENDPOINT, headers=headers, json={"base64_file": encoded})
    if response.status_code == 200:
        doc_id = response.json().get("id")
        doc_resp = requests.get(DOC_RETRIEVE_ENDPOINT + doc_id, headers=headers)
        if doc_resp.status_code == 200:
            return doc_resp.json().get("text", "")
    return ""

def ask_groq(text, question):
    prompt = f"Given the following financial document, answer the question as accurately as possible.\n\n{text}\n\nQuestion: {question}\nAnswer:"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return "Error retrieving answer."

if uploaded_files:
    all_texts = []
    st.info("Parsing uploaded documents...")

    for i, file in enumerate(uploaded_files):
        with st.spinner(f"Parsing file {file.name}..."):
            parsed_text = upload_and_parse(file)
            all_texts.append(parsed_text)

    # Extract metrics using Groq for each year
    metrics = ["Revenue", "Net Income", "EBITDA", "Operating Margin", "Gross Margin"]
    extracted_data = []

    for text in all_texts:
        year_data = {}
        for metric in metrics:
            answer = ask_groq(text, f"What is the {metric}?")
            year_data[metric] = answer
        extracted_data.append(year_data)

    df = pd.DataFrame(extracted_data)
    df.insert(0, "Year", [f"Year {i+1}" for i in range(len(df))])
    st.subheader("📊 Extracted Key Metrics")
    st.dataframe(df)

    # --- Visuals ---
    st.subheader("📈 Revenue and Margin Trends")
    for metric in ["Revenue", "Net Income", "EBITDA"]:
        try:
            df[metric] = df[metric].replace({"[$,]": ""}, regex=True).astype(float)
        except:
            continue

    fig1, ax1 = plt.subplots()
    ax1.plot(df["Year"], df["Revenue"], marker="o")
    ax1.set_title("Revenue Over Time")
    ax1.set_ylabel("Revenue")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Year"], df["Net Income"], marker="o", label="Net Income")
    ax2.plot(df["Year"], df["EBITDA"], marker="o", label="EBITDA")
    ax2.set_title("Profit Metrics Over Time")
    ax2.legend()
    st.pyplot(fig2)

    # Optional LLM-powered Q&A
    st.subheader("💬 Ask a Question About These Docs")
    user_q = st.text_input("Type your question here")
    if user_q:
        combined_text = "\n".join(all_texts)
        answer = ask_groq(combined_text, user_q)
        st.write("**Answer:**", answer)
