import streamlit as st
from modules.text_extractor import extract_text
from pathlib import Path
import os
from modules.question_generator import generate_questions
import json
import re

import streamlit as st
import re
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- App title and sidebar ---
st.set_page_config(page_title="SmartExam AI", layout="wide")
st.sidebar.title("🧠 SmartExam Dashboard")

# Sidebar navigation
page = st.sidebar.radio(
    "Select a module",
    ["📘 Mock Test Generator", "📄 Plagiarism Checker"]
)

# Create uploads folder if not exists
os.makedirs("uploads", exist_ok=True)

def extract_json(text):
    try:
        # Extract JSON using regex
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        return json.loads(json_str)
    except:
        st.error("❌ Could not extract JSON from model response")
        st.code(text)
        return None

# -------------------------------------------------------------------
# Helper for safe JSON parsing
# -------------------------------------------------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        cleaned = text.strip()

        # Remove accidental backticks
        cleaned = cleaned.replace("```json", "").replace("```", "")

        # Try again
        try:
            return json.loads(cleaned)
        except Exception as e:
            st.error("❌ Model output is not valid JSON")
            st.code(cleaned)
            raise e

# -----------------------------------------------
# MOCK TEST GENERATOR MODULE
# -----------------------------------------------
if page == "📘 Mock Test Generator":
    st.title("📘 Mock Test Generator")
    st.write("Upload your course materials to generate mock exams automatically!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload file (PDF, DOCX, PPTX, or Image)",
        type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        # Save uploaded file temporarily
        file_path = Path("uploads") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Detect file type
        ext = uploaded_file.name.split(".")[-1].lower()

        st.info(f"📂 Uploaded file: **{uploaded_file.name}**")
        st.write("🔄 Extracting text...")

        try:
            extracted_text = extract_text(str(file_path), ext)
            if extracted_text:
                st.success("✅ Text extracted successfully!")
                # Show text in a large textarea
                extracted_text_area = st.text_area(
                    "Extracted Text", extracted_text, height=300
                )

                # Submit button
                if st.button("🚀 Generate Mock Exam"):
                    st.info("🧠 Generating mock questions (MCQs, subjective, objective)...")
                    # (Placeholder for next step)
                    with st.spinner("Generating questions..."):
                        output = generate_questions(extracted_text_area)

                    st.subheader("📘 Mock Test Generated")
                    # st.write(output)

                    # Save to session state for later solving
                    st.session_state["mock_questions"] = output
                    
            else:
                st.warning("⚠️ No text could be extracted from this file.")
        except Exception as e:
            st.error(f"❌ Error extracting text: {e}")
    
    
    # --------------------------------------------------------
    # DISPLAY & SOLVE MOCK TEST
    # --------------------------------------------------------
    if "mock_questions" in st.session_state:
        qdata = st.session_state["mock_questions"]
        qdata = extract_json(qdata)

        # st.write("🔍 Parsed JSON:", qdata)
        print("🔍 Parsed JSON:", qdata)

        st.header("📝 Attempt Your Test")

        # --------------------- MCQs -------------------------
        st.subheader("📍 Multiple Choice Questions (MCQs)")
        mcq_answers = {}

        for i, q in enumerate(qdata["mcqs"]):
            st.write(f"**Q{i+1}: {q['question']}**")
            mcq_answers[i] = st.radio(
                "Choose your answer:",
                q["options"],
                key=f"mcq_{i}"
            )
            st.write("---")

        # --------------------- Short Questions -------------------------
        st.subheader("🟦 Short Questions")
        short_answers = []
        for i, q in enumerate(qdata["short_questions"]):
            st.write(f"**Q{i+1}: {q}**")
            # ans = st.text_area(f"Q{i+1}: {q}", key=f"short_{i}")
            # short_answers.append(ans)
            st.write("---")

        # --------------------- Long Questions -------------------------
        st.subheader("🟩 Long Questions")
        long_answers = []
        for i, q in enumerate(qdata["long_questions"]):
            st.write(f"**Q{i+1}: {q}**")
            # ans = st.text_area(f"Q{i+1}: {q}", key=f"long_{i}")
            # long_answers.append(ans)
            st.write("---")

        # --------------------- Submission Button ----------------------
        if st.button("📊 Submit Test"):
            st.header("📊 Results")

            correct = 0
            total = len(qdata["mcqs"])

            for i, q in enumerate(qdata["mcqs"]):

                student_ans = mcq_answers[i].strip().lower()

                # Normalize model answer
                correct_ans = q["answer"].strip().lower()

                if student_ans == correct_ans:
                    correct += 1

                # CASE 1 — model returns "A"
                if len(correct_ans) == 1:
                    # Example: "A"
                    correct_ans_full = next(
                        (opt.lower() for opt in q["options"] if opt.lower().startswith(correct_ans.lower())),
                        None
                    )
                else:
                    correct_ans_full = correct_ans

                # CASE 2 — model returns "A. Statement"
                # Extract letter
                correct_letter = correct_ans[0] if correct_ans[0].isalpha() else None
                if correct_letter:
                    correct_by_letter = next(
                        (opt.lower() for opt in q["options"] if opt.lower().startswith(correct_letter.lower())),
                        None
                    )
                else:
                    correct_by_letter = None

                # Final matching
                if (
                    student_ans == correct_ans
                    or student_ans == correct_ans_full
                    or student_ans == correct_by_letter
                ):
                    correct += 1

            st.success(f"MCQ Score: **{correct}/{total}** 🎉")

            st.info("Short and Long questions require manual checking.")


# -----------------------------------------------
# PLAGIARISM CHECKER MODULE (Coming soon)
# -----------------------------------------------
elif page == "📄 Plagiarism Checker":
    st.title("📄 Plagiarism Checker")
    st.info("This module will allow checking similarity between documents.")

    # Optional: for heatmap visualization
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False

    # -------------------- PAGE SETUP --------------------
    st.set_page_config(page_title="PDF Similarity Checker", layout="wide")
    st.title("📚 PDF Similarity Checker")
    st.write("Upload multiple PDF documents to compare their content similarity and detect possible plagiarism.")


    # -------------------- HELPER FUNCTIONS --------------------
    def extract_text_from_pdf(file):
        """Extract text from a PDF file using PyPDF2."""
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip().replace('\n', ' ')


    def clean_text(text):
        """Clean extracted text for better comparison."""
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces/newlines
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())  # remove special chars
        return text


    def get_remark(score):
        """Return a plagiarism remark based on similarity score."""
        if score > 0.8:
            return "🔴 High"
        elif score > 0.5:
            return "🟠 Moderate"
        else:
            return "🟢 Low"


    def make_unique_names(names):
        """Ensure all filenames are unique to avoid duplicate DataFrame columns."""
        seen = {}
        unique_names = []
        for name in names:
            if name not in seen:
                seen[name] = 1
                unique_names.append(name)
            else:
                seen[name] += 1
                new_name = f"{name} ({seen[name]})"
                unique_names.append(new_name)
        return unique_names

    if "clear_uploads" not in st.session_state:
        st.session_state.clear_uploads = False

    # -------------------- FILE UPLOAD --------------------
    uploaded_files = st.file_uploader("📁 Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.write("**Uploaded files:**")
        for file in uploaded_files:
            st.write("-", file.name)

        # -------------------- TEXT EXTRACTION --------------------
        with st.spinner("Extracting and processing text..."):
            texts = []
            names = []
            for file in uploaded_files:
                raw_text = extract_text_from_pdf(file)
                cleaned = clean_text(raw_text)
                texts.append(cleaned)
                names.append(file.name)

            # Make names unique if duplicates exist
            names = make_unique_names(names)

        # -------------------- TF-IDF SIMILARITY --------------------
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        df_sim = pd.DataFrame(similarity_matrix, index=names, columns=names)

        # -------------------- DISPLAY RESULTS --------------------
        st.subheader("📊 Similarity Matrix (Cosine Similarity)")
        st.dataframe(df_sim.style.format("{:.2f}"))

        # -------------------- PLAGIARISM REMARKS --------------------
        remarks = df_sim.copy()
        for i in range(len(df_sim)):
            for j in range(len(df_sim)):
                if i != j:
                    remarks.iloc[i, j] = get_remark(df_sim.iloc[i, j])
                else:
                    remarks.iloc[i, j] = "—"

        st.subheader("📑 Plagiarism Remarks")
        st.dataframe(remarks)

        # -------------------- HEATMAP VISUALIZATION --------------------
        if SEABORN_AVAILABLE:
            st.subheader("🔥 Visual Similarity Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_sim, annot=True, cmap="YlGnBu", xticklabels=names, yticklabels=names)
            st.pyplot(fig)
        else:
            st.info("Install seaborn to see a heatmap visualization: `pip install seaborn`")

        # -------------------- CLEAR BUTTON --------------------
        # if st.button("🔄 Clear Files and Start Over"):
        #     st.session_state.clear_uploads = True
        #     st.rerun()
            # try:
            #     st.rerun()  # Streamlit 1.30+ (new)
            # except AttributeError:
            #     st.experimental_rerun()  # older versions

        if st.session_state.clear_uploads:
            uploaded_files = []  # effectively clear the uploaded files
            st.session_state.clear_uploads = False
        
        # st.write("Prepared by Muhammad Aadil Yusuf (aadilyusuf99@gmail.com)")

    else:
        st.info("👆 Please upload at least two PDF files to compare.")


# -----------------------------------------------
# AI ASSIGNMENT DETECTOR MODULE (Coming soon)
# -----------------------------------------------
# elif page == "🤖 AI Assignment Detector":
#     st.title("🤖 AI Assignment Detector")
#     st.info("This module will detect whether the uploaded assignment is AI-generated or human-written.")
