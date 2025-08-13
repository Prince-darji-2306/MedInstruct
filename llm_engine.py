import os
import re
from groq import Groq
from markupsafe import Markup
from utils.messages import get_messages


API = os.getenv("GROQ")
client = Groq(api_key=API)

def ask_chatbot(device_prompt):
    messages = get_messages()
    full_messages = messages + [{'role': 'user', 'content': device_prompt}]
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=full_messages,
        max_tokens=500,
        temperature=0.15,
        stream=False
    )
    return response.choices[0].message.content

def cleanRes(text):
    bolded = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    bolded = re.sub(r'\n','<br>',bolded)
    return Markup(bolded)

def get_response(queryu, results):
    device_prompt = f"User Question: {queryu}\n\nHere are the top {len(results)} candidate devices:\n"
    for idx, dev in enumerate(results, start=1):
        device_prompt += f"""
        Device {idx}
        - Name: {dev['name']}
        - Definition: {dev['definition']}
        - Instructions: {dev['instructions']}
        """
    answer = ask_chatbot(device_prompt)
    if "**Conversational Speech Version:**" in answer:
        result_markdown, speak_instructions = answer.split("**Conversational Speech Version:**", 1)
    else:
        result_markdown = answer
    
    return cleanRes(result_markdown), speak_instructions