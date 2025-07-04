import os
import re
from flask import Flask, render_template, request
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from flask import Markup
from markupsafe import Markup
from functools import lru_cache

# ----------------------------
# ENV + INIT
# ----------------------------
load_dotenv()

# Ensure Hugging Face and Transformers caches live in your Render disk
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/transformers")

GROQ_API_KEY = os.getenv("GROQ")
assert GROQ_API_KEY, "GROQ API key missing! Check your .env file."
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

# ----------------------------
# LOAD MODEL, INDEX, DATA
# ----------------------------
@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer('model/BGE-Medico-Model')

@lru_cache(maxsize=1)
def load_index():
    return faiss.read_index('model/device_index.faiss')

@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv('model/Final_devices.csv')
    return df

# ----------------------------
# PROMPT CONFIG
# ----------------------------
messages = [
    {
        "role": "system",
        "content": (
            "You are a calm and professional medical assistant. "
            "When given the user query plus a list of candidate devices, "
            "pick the single most appropriate device. "
            "Then output, in this order:\n\n"
            "1. **Name**: The exact Device Name (unchanged)\n\n"
            "2. **Definition**: The exact Device Definition (unchanged)\n\n"
            "3. **Step-by-Step Instructions**:\n"
            "   - A short title for each step (like a heading)\n"
            "   - Then on the next line, a large, easy, step-by-step explanation starting with a dash.\n"
            "   - The instructions should be very clear and thorough for a nursing student.\n\n"
            "4. **Conversational Speech Version:**\n"
            "   - Convert the step-by-step instructions into a natural, spoken paragraph as if explaining verbally to a new nurse in very simple manner.\n"
            "   - Avoid bullet points or markdown, write it as a smooth, spoken paragraph.\n\n"
            "Do not include disclaimers or any extra commentary outside these four sections."
        )
    }
]
extras = [{'role': 'user',
        'content': 'User Question: Mri machine\n\nHere are the top 3 candidate devices:\n\n    Device 1\n    - Name: Full-body MRI system, permanent magnet\n    - Definition: A general-purpose magnetic resonance imaging (MRI) system designed to scan any targeted area of the body. It includes a permanent magnet assembly and can be fixed-location, mobile, or transportable. In addition to producing conventional MR images, it can be designed or modified through additional software/hardware to perform MR spectroscopy and other real-time imaging procedures necessary for physiologically gated imaging procedures, or MRI mammography and other MRI guided interventional, therapeutic, or surgical applications. It is available in a variety of gantry configurations including closed bore, open bore, open-sided or other patient accessible designs.\n    - Instructions: 1. Power on the MRI system and allow it to complete its self-calibration.\n 2. Position the patient comfortably on the imaging table, ensuring proper alignment.\n 3. Select the desired imaging sequence and parameters for the targeted body area.\n 4. Initiate the scan, ensuring the permanent magnet assembly is active.\n 5. Monitor the image acquisition and patient comfort throughout the procedure.\n 6. If applicable, perform MR spectroscopy or other real-time imaging procedures.\n 7. Review and analyze the conventional MR images on the display.\n 8. Power off the system and assist the patient off the table.\n 9. Perform daily quality assurance checks and routine maintenance.\n    \n    Device 2\n    - Name: MRI system workstation\n    - Definition: A freestanding image processing workstation specifically designed to be networked with one or more magnetic resonance imaging (MRI) systems. An MRI workstation differs from the operator\'s console in that it does not contain the controls for the direct operation of the diagnostic imaging system. It is designed to receive and transmit data both on-line and off-line and is typically located at a site remote from the MRI system\'s operator console. It is configured to provide the capability to further process, manipulate and/or view patient images and information collected from one or more MRI systems.\n    - Instructions: 1. Ensure the workstation is properly networked with one or more MRI systems.\n 2. Power on the workstation and access the image processing software.\n 3. Receive and transmit MRI image data both online and offline.\n 4. Utilize the workstation\'s capabilities to further process, manipulate, and view patient images.\n 5. Ensure data security and patient privacy are maintained.\n 6. Perform regular software updates and system maintenance.\n    \n    Device 3\n    - Name: Extremity MRI system, resistive magnet\n    - Definition: A diagnostic magnetic resonance imaging (MRI) system employing resistive magnet technology specifically designed to image only the head and neck or limbs. An extremity imaging system is typically closed bore design with cylindrical or rectangular bore openings but can be "open" bore design. This device includes MRI systems with conventional extremity capabilities, conventional systems capable of performing spectroscopy or other real-time imaging procedures necessary for interventional and therapeutic MRI applications, and dedicated magnetic resonance (MR) spectroscopy systems.\n    - Instructions: 1. Power on the MRI system and ensure it is connected to a power source.\n 2. Position the patient for imaging only the head and neck or limbs.\n 3. Ensure the closed bore or open bore design accommodates the patient.\n 4. Select the desired imaging protocol for diagnostic MRI procedures.\n 5. If applicable, perform MR spectroscopy or other real-time imaging procedures.\n 6. Monitor the image display for diagnostic information.\n 7. Power off the system and safely remove the patient after the scan.\n 8. Perform routine maintenance and safety checks of the MRI system.\n    '},
        {'role': 'assistant',
        'content': "**Name:** Full-body MRI system, permanent magnet\n\n**Definition:** A general-purpose magnetic resonance imaging (MRI) system designed to scan any targeted area of the body.\n\n**Step-by-Step Instructions:**\n\n**1. Prepare the System**\n- Power on the MRI system and allow it to complete its self-calibration.\n\n**2. Position the Patient**\n- Position the patient comfortably on the imaging table, ensuring proper alignment.\n\n**3. Select the Imaging Sequence**\n- Select the desired imaging sequence and parameters for the targeted body area.\n\n**4. Initiate the Scan**\n- Initiate the scan, ensuring the permanent magnet assembly is active.\n\n**5. Monitor the Scan**\n- Monitor the image acquisition and patient comfort throughout the procedure.\n\n**6. Perform Additional Procedures (Optional)**\n- If applicable, perform MR spectroscopy or other real-time imaging procedures.\n\n**7. Review the Images**\n- Review and analyze the conventional MR images on the display.\n\n**8. Power Off and Assist the Patient**\n- Power off the system and assist the patient off the table.\n\n**9. Perform Maintenance**\n- Perform daily quality assurance checks and routine maintenance.\n\n**Conversational Speech Version:**\nWhen using the full-body MRI system, first power it on and let it complete its self-calibration. Then, position the patient comfortably on the imaging table, making sure they're properly aligned. Next, select the desired imaging sequence and parameters for the targeted body area. After that, initiate the scan, ensuring the permanent magnet assembly is active. Throughout the procedure, monitor the image acquisition and patient comfort. If necessary, you can also perform MR spectroscopy or other real-time imaging procedures. Once the scan is complete, review and analyze the conventional MR images on the display. Finally, power off the system, assist the patient off the table, and perform daily quality assurance checks and routine maintenance."}]

messages.extend(extras)

# ----------------------------
# SEARCH FUNCTION
# ----------------------------
def search_device(query, top_k=3):
    df = load_data()
    model = load_model()
    index = load_index()
    
    cols = ['name', 'definition', 'instructions']
    if query.isdigit():
        results = df[cols][df["code"] == int(query)]
        if not results.empty:
            return results.to_dict(orient="records")
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [df.iloc[idx][cols].to_dict() for idx in I[0]]

# ----------------------------
# ASK GROQ
# ----------------------------
def ask_chatbot(device_prompt):
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

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/", methods=["GET", "POST"])
@app.route("/device", methods=["GET", "POST"])
def device():
    result_markdown = None
    speak_instructions = None
    queryu = ""
    if request.method == "POST":
        queryu = request.form.get("query")
        results = search_device(queryu)
        if results:
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
            
            result_markdown = cleanRes(result_markdown)
    return render_template("device.html", result_markdown=result_markdown, speak_instructions=speak_instructions, queryu=queryu)

@app.route("/about")
def about():
    return render_template("about.html")

# ----------------------------
# MAIN
# ----------------------------
application = app

# Local debug server (not used by Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
