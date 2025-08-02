import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- Custom Styling ----------
st.markdown("""
    <style>
        body {
            background-color: #001f3f !important;
            color: white !important;
        }
        .stApp {
            background-color: #001f3f;
        }
        .stTextInput>div>div>input, .stTextArea>div>textarea {
            background-color: #e0f0ff;
            color: black;
        }
        .stButton>button {
            background-color: #0059b3;
            color: white;
        }
        .stSidebar, .stSidebarContent {
            background-color: #003366;
            color: white;
        }
        .css-1cpxqw2, .css-10trblm, .css-q8sbsg {
            color: white !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Embedding Model ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Sidebar ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Google_Gemini_logo.svg/2048px-Google_Gemini_logo.svg.png", width=150)
st.sidebar.title("🔐 Gemini 2.0 Flash Auth")
st.sidebar.markdown("""
Enter your **Gemini API Key** to get started. You can upload multiple PDFs and ask study-related questions.
""")
api_key = st.sidebar.text_input("🔑 API Key", type="password")
st.sidebar.markdown("---")

# ---------- Title & Intro ----------
st.title("📘 StudyMate: Multi-PDF Q&A Assistant")
st.markdown("""
Welcome to **StudyMate**, your AI study assistant powered by Gemini 2.0 Flash. Upload multiple PDFs, ask complex questions, and get smart answers with source context.
""")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        st.sidebar.success("✅ Authenticated")
        pdfs = st.sidebar.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)
        reset = st.sidebar.button("🔄 Reset Session")

        if reset:
            for key in ["chunks", "chunk_embeddings", "qa_history"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

        def extract_text_from_pdf(pdf_file):
            text = ""
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text

        if pdfs and "chunks" not in st.session_state:
            with st.spinner("📚 Extracting and embedding content from PDFs..."):
                all_text = []
                for pdf in pdfs:
                    file_bytes = pdf.read()
                    pdf_text = extract_text_from_pdf(file_bytes)
                    all_text.append(pdf_text)

                full_text = "\n".join(all_text)
                chunks = [" ".join(full_text.split()[i:i+500]) for i in range(0, len(full_text.split()), 500)]
                chunk_embeddings = embed_model.encode(chunks)
                st.session_state["chunks"] = chunks
                st.session_state["chunk_embeddings"] = chunk_embeddings
                st.session_state["qa_history"] = []

        if "chunks" in st.session_state and "chunk_embeddings" in st.session_state:
            with st.expander("💬 Ask Complex Questions", expanded=True):
                st.markdown("""
                Type your questions below. You can ask about definitions, explanations, summaries, comparisons, or specific parts of the uploaded PDFs.
                """)
                question = st.text_input("🧠 Your Question")
                extra_detail = st.text_area("🔎 Add any extra detail or specific section reference (optional)")

                if question:
                    with st.spinner("🤖 Thinking and answering..."):
                        full_question = question + ("\nAdditional context: " + extra_detail if extra_detail else "")
                        q_embedding = embed_model.encode([full_question])
                        index = faiss.IndexFlatL2(len(q_embedding[0]))
                        index.add(np.array(st.session_state["chunk_embeddings"]))
                        _, I = index.search(np.array(q_embedding), 5)
                        top_chunks = [st.session_state["chunks"][i] for i in I[0]]
                        context = "\n".join(top_chunks)
                        prompt = f"""
Use the following context to answer the question:

Context:
{context}

Question:
{full_question}
"""
                        response = model.generate_content(prompt)
                        answer = response.text

                        st.session_state["qa_history"].append({"q": full_question, "a": answer, "c": context})

        if "qa_history" in st.session_state:
            st.markdown("---")
            st.subheader("🧾 Q&A Thread")
            for i, qa in enumerate(reversed(st.session_state["qa_history"])):
                with st.container():
                    st.markdown(f"<div style='background-color:#0b2545;padding:10px;border-radius:10px;color:white;'>", unsafe_allow_html=True)
                    st.markdown(f"#### ❓ Question {len(st.session_state['qa_history']) - i}: {qa['q']}")
                    st.markdown(f"**💡 Answer:** {qa['a']}")
                    with st.expander("📂 Context Used to Answer"):
                        st.write(qa['c'])
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")

    except Exception as e:
        st.sidebar.error(f"❌ Authentication failed: {str(e)}")
else:
    st.info("🔑 Please enter your Gemini API key in the sidebar to continue.")





