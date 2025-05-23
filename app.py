# app.py — Fixed Version 1: Extract generated text properly

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

# Simplified prompt for better results
def build_prompt(region, condition, age_group):
    return f"""Based on the health condition "{condition}" in {region} for {age_group}, provide exactly 3 specific crop recommendations. For each crop, explain its nutritional benefits and why it's suitable for this condition.

Format your response as:
1. [Crop Name]: [Detailed explanation]
2. [Crop Name]: [Detailed explanation] 
3. [Crop Name]: [Detailed explanation]

Recommendations:"""

# Function to query Hugging Face API with better text extraction
def query_huggingface(payload):
    try:
        response = requests.post(
            api_url, 
            headers=headers, 
            json={
                "inputs": payload,
                "parameters": {
                    "max_new_tokens": 500,  # Increased for longer responses
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False,
                    "stop": ["<|user|>", "<|system|>"],  # Stop tokens to prevent rambling
                    "repetition_penalty": 1.1  # Reduce repetition
                }
            }
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                
                # If the full prompt is still included, extract only the new part
                if payload in generated_text:
                    generated_text = generated_text.replace(payload, '').strip()
                
                return generated_text if generated_text else "No response generated."
            else:
                return "Unexpected response format."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error querying API: {str(e)}"

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
    with st.spinner("Generating recommendations..."):
        prompt = build_prompt(region, condition, age_group)
        result = query_huggingface(prompt)
        st.write(result)

    st.markdown("\n**Note:** This is a prototype using public and synthetic data. For clinical use, consult dietary professionals.")