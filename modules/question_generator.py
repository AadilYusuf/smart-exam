import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

# model = genai.GenerativeModel("gemini-2.5-flash")   # fast + cheap
model = genai.GenerativeModel("models/gemini-flash-lite-latest") # for updated limits

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)

# def generate_questions(text):
#     prompt = f"""
#     You are an exam paper generator.

#     Based on the following study material, generate:

#     1. 10 MCQs with:
#        - 4 options each
#        - Correct answer clearly marked
#     2. 5 short questions
#     3. 3 long questions

#     STUDY MATERIAL:
#     {text}
#     """

#     response = model.generate_content(prompt)
#     return response.text


def generate_questions(text):
    prompt = f"""
    You are an exam generator. Produce STRICT JSON ONLY. 
    NO explanation. NO formatting. NO markdown. NO backticks.

    JSON FORMAT:
    {{
    "mcqs": [
        {{
        "question": "string",
        "options": ["A","B","C","D"],
        "answer": "A"
        }}
    ],
    "short_questions": ["string"],
    "long_questions": ["string"]
    }}

    Now based on the following study material, generate:
    - 10 MCQs
    - 5 Short Questions
    - 3 Long Questions

    STUDY MATERIAL:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text