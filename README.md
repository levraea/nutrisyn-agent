# NutriSyn: Nutrition + Therapeutics AI Agent (Together.AI Version)

NutriSyn is a prototype AI agent that recommends regionally and clinically tailored dietary interventions by linking public health data with agricultural insights.

## 🔍 What It Does
- Accepts inputs: Region, Health Condition, Age Group
- Shows static recommendations from a mock dataset
- Uses an AI agent (Together.AI LLaMA 2) to suggest alternatives when data is unavailable

## 📁 Project Structure
```
NutriSyn/
├── NutriSyn_Mock_Data.csv
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 How to Deploy (Streamlit Cloud)

1. Push this repo to GitHub.
2. Go to [streamlit.io](https://streamlit.io) and sign in.
3. Link your GitHub repo.
4. Set the Together.AI API key in Streamlit Secrets:
   ```
   together_api_key = "your-api-key"
   ```

## 🧠 Note
This app uses public mock data. Not intended for clinical or diagnostic use.