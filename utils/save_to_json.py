import json
import os
def save_conversation(question, response, file_path="logs/conversation_log.json"):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    response_text = response.content if hasattr(response, "content") else str(response)

    conversation = {
        "question": question,
        "response": response_text
    }

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(conversation)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

