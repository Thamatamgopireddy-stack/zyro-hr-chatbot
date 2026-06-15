import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about company HR policies. Powered by RAG.")

# ── Load API key ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
CORPUS_PATH = "./hr_docs/"  # Upload PDFs here on Streamlit Cloud

# ── Build RAG pipeline (cached) ───────────────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )
    return retriever, llm

retriever, llm = build_pipeline()

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY using the HR policy context provided below.
Include specific details like numbers, days, and procedures when available.

Context:
{context}

Question: {question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
Classify if this question is related to HR policies, employee benefits,
leave, compensation, onboarding, conduct, travel, IT security, or performance.
Reply ONLY with: in_scope or out_of_scope

Question: {question}
""")

REFUSAL = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. Please contact the "
    "relevant department for other queries."
)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source','Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

def ask_bot(question):
    cls_prompt = OOS_PROMPT.invoke({"question": question})
    cls = StrOutputParser().invoke(llm.invoke(cls_prompt)).strip().lower()
    if "out_of_scope" in cls:
        return REFUSAL, []
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer_prompt = RAG_PROMPT.invoke({"context": context, "question": question})
    answer = StrOutputParser().invoke(llm.invoke(answer_prompt))
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
    return answer, sources

# ── Chat UI ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(s)

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources = ask_bot(prompt)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources cited"):
                for s in sources:
                    st.caption(f"• {s}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
