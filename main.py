import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


st.set_page_config(
    page_title="RAG Based Question Answering System",
    page_icon="🤖",
    layout="centered"
)

st.title("RAG Based Question Answering System")


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["ANONYMIZED_TELEMETRY"] = "False"

@st.cache_resource
def create_rag_chain():

    llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

    loader = WebBaseLoader(
        "https://python.langchain.com/docs/introduction/"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Use the given context to answer the question.

    If the answer is present in the context, use it.

    If the answer is not present in the context,
    then answer using your own knowledge clearly and accurately.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


with st.spinner("Loading RAG system..."):
    retrieval_chain = create_rag_chain()


query = st.text_input(
    "Ask your question:",
    placeholder="Type your question here..."
)


if st.button("Enter Here"):

    if query.strip():

        with st.spinner("Generating answer..."):

            response = retrieval_chain.invoke({
                "input": query
            })

        st.subheader("Answer")
        st.write(response["answer"])

    else:
        st.warning("Please enter a question.")
