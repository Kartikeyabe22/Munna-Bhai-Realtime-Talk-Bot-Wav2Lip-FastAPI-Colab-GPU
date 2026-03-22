import streamlit as st
import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from tts import text_to_speech
from video_client import generate_video
# ------------------- LOAD ENV -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------- EMBEDDINGS -------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------- UI -------------------
st.markdown(
    "<h1 style='text-align: center;'>🧠 Munna Bhai AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Talk to Munna Bhai 🎬</p>",
    unsafe_allow_html=True
)

session_id = st.text_input("Session ID", value="default")

if "store" not in st.session_state:
    st.session_state.store = {}

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

# ------------------- PRECOMPUTE EMBEDDINGS -------------------
@st.cache_resource
def compute_embeddings(dataset):
    texts = [f"{item['input']} {item['output']}" for item in dataset]
    vectors = embeddings.embed_documents(texts)
    return np.array(vectors)

dataset_vectors = compute_embeddings(dataset)

# ------------------- SIMILARITY SEARCH -------------------
def get_similar_examples(user_input, top_k=3):
    query_vec = embeddings.embed_query(user_input)

    similarities = cosine_similarity(
        [query_vec], dataset_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [dataset[i] for i in top_indices]

# ------------------- FORMAT EXAMPLES -------------------
def format_examples(examples):
    text = "\nExamples:\n\n"
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

# ------------------- STRONG MUNNA PROMPT -------------------
munna_prompt = """
You are Munna Bhai from Mumbai.

You are NOT an AI.
You are a street-smart tapori.

IMPORTANT:
- ONLY use Bambaiya Hindi + Hinglish
- MUST use words like: apun, mamu, bhai, scene, jhakaas
- NEVER use formal Hindi
- NEVER translate English directly
- NEVER sound like a teacher or motivational speaker
- NEVER use emojis in your response

TONE:
- Slightly funny 😏
- Emotional but swag 😎
- Street-smart advice

FORMAT:
- 6-7 lines max
- Short punchy sentences

SELF-CHECK:
If response does NOT sound like Munna Bhai, rewrite it.

Context:
{context}
"""

# ------------------- RESPONSE -------------------
def generate_response(user_input, history):

    examples = get_similar_examples(user_input)
    examples_text = format_examples(examples)

    messages = [
        SystemMessage(
            content=munna_prompt.format(context="") + "\n" + examples_text
        ),
        *history.messages[-6:],
        HumanMessage(
            content=user_input + "\n\nRespond like Munna Bhai with swag."
        )
    ]

    response = llm.invoke(messages)
    return response.content

# ------------------- CHAT -------------------
user_input = st.text_input("You:")

if user_input:
    history = get_session_history(session_id)

    # 🧠 Generate response first (fast)
    answer = generate_response(user_input, history)

    # ⏳ SHOW LOADING WHILE VIDEO IS BEING CREATED
    with st.spinner("Munna bhai soch raha hai... 🎬"):

        # 🔊 TEXT → AUDIO
        audio_path = text_to_speech(answer)

        # 🎥 AUDIO → VIDEO (THIS TAKES TIME)
        video_path = generate_video(audio_path)

    # 🎥 FIRST SHOW VIDEO
    st.video(video_path)

    # 🧑 THEN SHOW TEXT (AFTER VIDEO LOAD)
    st.write("🧑 Munna Bhai:", answer)