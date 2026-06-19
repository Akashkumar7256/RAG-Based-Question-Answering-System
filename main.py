from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()

print("Loaded Docs:", len(docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(docs)

print("Chunks:", len(documents))

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

If the answer is not present in the context, then answer using your own knowledge clearly and accurately.

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

query = input("Ask Question: ")

response = retrieval_chain.invoke({
    "input": query
})

print("\nAnswer:\n")
print(response["answer"])  