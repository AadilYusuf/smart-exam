from modules.question_generator import generate_questions

# Sample text to test
sample_text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on building machines 
that can think and act like humans. Applications of AI include robotics, language translation, 
medical diagnosis, image recognition, and automated driving systems.
"""

print("🔄 Generating questions using Gemini API...\n")

try:
    output = generate_questions(sample_text)
    print("✅ API Response Received!\n")
    print(output)
except Exception as e:
    print("❌ Error during API call:")
    print(e)
