# app.py — Streamlit Cloud Compatible App (Using Hugging Face Inference API Directly)

import streamlit as st
st.set_page_config(page_title="NutriSyn AI", layout="centered")

import pandas as pd
import requests
import json

# Load Hugging Face API key from Streamlit Secrets
hf_api_key = st.secrets["huggingface_api_key"]
api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {hf_api_key}"}

# Load mock data
@st.cache_data
def load_data():
    return pd.read_csv("NutriSyn_Mock_Data.csv")

data = load_data()

# Title and description
st.title("NutriSyn: Nutrition + Therapeutics AI Agent")
st.markdown("""
This AI agent helps public health professionals, clinicians, and agronomists identify nutrition-enhancing crop interventions aligned with chronic disease trends.

Select a region and health condition to see tailored crop-based dietary recommendations.
""")

# Input fields
region = st.selectbox("Select a Region", sorted(data['Region'].unique()))
condition = st.selectbox("Select a Health Condition", sorted(data['Condition'].unique()))
age_group = st.selectbox("Age Group", sorted(data['Age Group'].unique()))
target = st.button("Get Recommendations")

# Prompt template for Zephyr chat format
def build_prompt(region, condition, age_group):
    return f"""
<|system|>
You are a helpful public health nutrition advisor.
<|user|>
Based on the condition: {condition}, region: {region}, and age group: {age_group},
suggest three crop-based nutritional interventions and explain briefly why each is suitable.
<|assistant|>
"""

# Function to query Hugging Face API
def query_huggingface(payload):
    response = requests.post(api_url, headers=headers, json={"inputs": payload})
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Output area
if target:
    st.subheader("Static Recommendations from Dataset")
    filtered = data[(data['Region'] == region) &
                    (data['Condition'] == condition) &
                    (data['Age Group'] == age_group)]
    if not filtered.empty:
        for crop in filtered['Crop'].unique():
            st.markdown(f"- {crop}")
    else:
        st.markdown("No dataset match found. Using AI agent for recommendations:")

    st.subheader("AI Agent Recommendation")
    prompt = build_prompt(region, condition, age_group)
    result = query_huggingface(prompt)
    st.write(result)

    st.markdown("\n**Note:** This is a prototype using public and synthetic data. For clinical use, consult dietary professionals.")
