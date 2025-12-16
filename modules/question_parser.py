import json

def parse_questions(raw_output):
    try:
        data = json.loads(raw_output)
        return data
    except json.JSONDecodeError:
        print("JSON parsing failed!")
        return None
