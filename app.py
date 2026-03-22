import streamlit as st
import os
import json
import numpy as np
import uuid
import hashlib
import requests

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from tts import text_to_speech

# ------------------- LOAD ENV -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------- VIDEO API -------------------
API_URL = "https://munna.instatunnel.my/generate"

def generate_video(audio_path):
    os.makedirs("videos", exist_ok=True)

    # 🔥 Hash audio (prevents duplicate videos)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    file_hash = hashlib.md5(audio_bytes).hexdigest()
    output_path = f"videos/{file_hash}.mp4"

    # ✅ If already exists → reuse
    if os.path.exists(output_path):
        return output_path

    files = {"audio": ("audio.wav", audio_bytes)}
    response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        raise Exception(f"Video generation failed: {response.status_code}")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

# ------------------- EMBEDDINGS -------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------- UI -------------------
st.set_page_config(page_title="Munna Bhai AI", layout="centered")
st.title("🧠 Munna Bhai AI")
st.write("Talk to Munna Bhai 🎬")

session_id = st.text_input("Session ID", value="default")

# ------------------- SESSION STATE -------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

# ------------------- MEMORY -------------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# ------------------- LOAD DATASET -------------------
@st.cache_resource
def load_dataset():
    with open("munna_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_dataset()

# ------------------- EMBEDDINGS -------------------
@st.cache_resource
def compute_embeddings(dataset):
    texts = [f"{item['input']} {item['output']}" for item in dataset]
    vectors = embeddings.embed_documents(texts)
    return np.array(vectors)

dataset_vectors = compute_embeddings(dataset)

# ------------------- SIMILARITY SEARCH -------------------
def get_similar_examples(user_input, top_k=3):
    query_vec = embeddings.embed_query(user_input)
    similarities = cosine_similarity([query_vec], dataset_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [dataset[i] for i in top_indices]

# ------------------- FORMAT CONTEXT -------------------
def format_context(examples):
    text = ""
    for ex in examples:
        text += f"User: {ex['input']}\nMunna: {ex['output']}\n\n"
    return text

# ------------------- LLM -------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=1.2,
    max_tokens=200
)

# ------------------- PROMPT -------------------
munna_prompt = """
You are Munna Bhai from Mumbai.

You are NOT an AI.
You are a street-smart tapori.

IMPORTANT:
- ONLY Bambaiya Hindi + Hinglish
- Use words like: apun, bhai, mamu, scene, jhakaas
- NEVER formal Hindi
- NEVER sound like teacher
- NEVER use emojis in your response

STYLE:
- Funny + swag
- Emotional but street smart
- 5-6 short punchy lines

Use this context if helpful:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", munna_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ------------------- CHAIN -------------------
chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ------------------- CHAT INPUT (BOTTOM FIXED) -------------------
user_input = st.chat_input("Type your message...")

# ------------------- HANDLE INPUT -------------------
if user_input:

    # 🧠 Retrieve context
    examples = get_similar_examples(user_input)
    context = format_context(examples)

    # 🧠 Generate response
    response = chain_with_memory.invoke(
        {
            "input": user_input,
            "context": context
        },
        config={"configurable": {"session_id": session_id}},
    )

    answer = response.content

    # 🎬 Generate video
    with st.spinner("Munna bhai soch raha hai... 🎬"):
        audio_path = text_to_speech(answer)
        video_path = generate_video(audio_path)

    # ✅ Store chat
    st.session_state.chat.append({
        "user": user_input,
        "bot": answer,
        "video": video_path
    })

# ------------------- RENDER CHAT -------------------
for chat in st.session_state.chat:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(chat["user"])

    with st.chat_message("assistant", avatar="🎬"):
        st.video(chat["video"])
        st.markdown(chat["bot"])