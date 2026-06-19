# 🤖 LangChain RAG Chatbot

A powerful **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, Groq's ultra-fast LLaMA 3.3 70B model, and ChromaDB vector store. This app loads web content, indexes it using HuggingFace embeddings, and answers your questions using both retrieved context and the LLM's own knowledge.

---

## ✨ Features

- 🔍 **Web Content Loading** — Automatically scrapes and indexes any webpage
- 🧠 **Smart Retrieval** — Uses vector similarity search to find the most relevant context
- ⚡ **Groq-Powered LLM** — Ultra-fast inference with LLaMA 3.3 70B Versatile
- 🗃️ **ChromaDB Vector Store** — Efficient local vector storage for embeddings
- 🤗 **HuggingFace Embeddings** — Uses `all-MiniLM-L6-v2` for high-quality sentence embeddings
- 💬 **Fallback to LLM Knowledge** — If context doesn't have the answer, LLM answers from its own knowledge

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Retrieval Chain             │
│                                     │
│  ┌─────────┐     ┌───────────────┐  │
│  │ ChromaDB│────▶│  Retrieved    │  │
│  │ Retriever│    │  Documents    │  │
│  └─────────┘     └──────┬────────┘  │
│                         │           │
│                         ▼           │
│               ┌──────────────────┐  │
│               │  Prompt Template │  │
│               │  (Context +      │  │
│               │   Question)      │  │
│               └────────┬─────────┘  │
│                        │            │
│                        ▼            │
│               ┌──────────────────┐  │
│               │  Groq LLM        │  │
│               │  (LLaMA 3.3 70B) │  │
│               └────────┬─────────┘  │
└────────────────────────┼────────────┘
                         │
                         ▼
                      Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| [LangChain](https://python.langchain.com/) | RAG pipeline & orchestration |
| [Groq](https://groq.com/) | LLM inference (LLaMA 3.3 70B) |
| [ChromaDB](https://www.trychroma.com/) | Vector store for embeddings |
| [HuggingFace](https://huggingface.co/) | Sentence embeddings (`all-MiniLM-L6-v2`) |
| [Python Dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

## 📋 Prerequisites

- Python 3.9 or higher
- A free **Groq API Key** → Get it at [console.groq.com](https://console.groq.com)
- Internet connection (for web scraping & HuggingFace model download)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements1.txt
```

> ⚠️ First run may take a few minutes as HuggingFace downloads the embedding model (~90MB).

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 🔒 Never share or commit your `.env` file. It's already in `.gitignore`.

### 5. Run the App

```bash
python main.py
```

You'll see output like:
```
Loaded Docs: 1
Chunks: 42
Ask Question: _
```

Type your question and press Enter to get an answer!

---

## 💡 Example Usage

```
Ask Question: What is LangChain?

Answer:
LangChain is a framework for developing applications powered by large language 
models (LLMs). It provides tools and abstractions for connecting LLMs with 
external data sources, APIs, and workflows — making it easy to build chatbots, 
RAG systems, agents, and more.
```

---

## 📁 Project Structure

```
📦 your-repo-name/
├── main.py              # Main application code
├── requirements1.txt    # Python dependencies
├── .env                 # API keys (NOT committed to Git)
├── .gitignore           # Files to exclude from Git
└── README.md            # Project documentation
```

---

## ⚙️ How It Works

1. **Load** — `WebBaseLoader` scrapes the target webpage (LangChain docs by default)
2. **Split** — `RecursiveCharacterTextSplitter` breaks content into 1000-char chunks with 200-char overlap
3. **Embed** — HuggingFace `all-MiniLM-L6-v2` converts chunks into vector embeddings
4. **Store** — ChromaDB stores these embeddings locally for fast retrieval
5. **Retrieve** — On user query, the most relevant chunks are fetched via similarity search
6. **Generate** — Groq's LLaMA 3.3 70B uses the context + query to generate a precise answer

---

## 🔧 Customization

### Change the Source URL
In `main.py`, update this line to scrape any webpage:
```python
loader = WebBaseLoader("https://your-target-website.com")
```

### Change the LLM Model
```python
llm = ChatGroq(model_name="llama-3.1-8b-instant")  # Faster, lighter model
```

### Adjust Chunk Size
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Smaller chunks = more precise retrieval
    chunk_overlap=100
)
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**  
GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

> 💬 *"Ask anything — the RAG knows!"*