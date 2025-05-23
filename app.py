# app.py — Streamlit Cloud Compatible App (Using Together.AI)

import streamlit as st
import pandas as pd
import os
from langchain.chains import LLMChain
from langchain.llms import Together
from langchain.prompts import PromptTemplate

# Load Together API key from Streamlit Secrets
together_api_key = st.secrets["together_api_key"]

# Load mock data
@st.cache_data
def load_data():
    return pd.read_csv("NutriSyn_Mock_Data.csv")

data = load_data()

# Title and description
st.set_page_config(page_title="NutriSyn AI", layout="centered")
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

# LangChain setup with Together.AI model
llm = Together(
    model="togethercomputer/llama-2-13b-chat",
    temperature=0.5,
    together_api_key=together_api_key
)

prompt = PromptTemplate(
    input_variables=["region", "condition", "age_group"],
    template="""
You are a public health nutrition advisor. Based on the condition: {condition}, region: {region}, and age group: {age_group}, suggest three appropriate crop-based nutritional interventions. Explain briefly why each is suitable.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

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
    result = chain.run({"region": region, "condition": condition, "age_group": age_group})
    st.write(result)

    st.markdown("\n**Note:** This is a prototype using public and synthetic data. For clinical use, consult dietary professionals.")
