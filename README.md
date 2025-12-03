📄 Document Intelligence Assistant

A Streamlit-powered application that converts documents (PDF, DOCX, PPTX, HTML) into a fully interactive AI-powered chatbot using Docling, LangChain, OpenRouter, and Chroma vector stores.
The system extracts text, structure, tables, and images from documents using OCR and advanced parsing, indexes the content into a vector database, and allows you to ask natural-language questions with responses grounded in the document.

🚀 Features

🧠 AI-powered Q&A over documents
🔍 Semantic search with vector embeddings
📄 OCR support for scanned PDFs
🗂️ Document structure viewer (tables, hierarchy, images)
💾 Persistent vector index using Chroma
🎛️ Reset button to clear index & chat history
🧩 Multiple file format support:

PDF
DOCX
PPTX
HTML

🏗️ Tech Stack

Streamlit – UI
Docling – Document & OCR processing
LangChain – LLM orchestration & agents
OpenRouter API – LLM + Embeddings
Chroma – Vector storage
Python 3.10+

📦 Installation (Local)

1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Set environment variables

Create a .env file:

OPENROUTER_API_KEY=sk-or-xxxxx

5. Run the app
streamlit run app.py

Visit:
http://localhost:8501

☁️ Deploy on AWS EC2 (Fast OCR + Faster Processing)
1. Launch an EC2 Instance

Recommended:

Ubuntu 22.04 LTS
Instance type:
g4dn, g5 etc.
Open security group ports:
22 (SSH)
8501 (Streamlit)

2. SSH into server
ssh -i key.pem ubuntu@EC2_PUBLIC_IP

3. Install system dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

4. Clone your repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

5. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

6. Add OpenRouter key
nano .env


Insert:

OPENROUTER_API_KEY=sk-or-xxxxx

7. Run Streamlit publicly
streamlit run app.py --server.address 0.0.0.0 --server.port 8501


Visit:
http://EC2_PUBLIC_IP:8501


🧼 Resetting the Index

The sidebar has a button:

🧹 Reset index & chat

This clears:
chroma_db/
outputs/
Document cache
Vector index
Chat messages

Use if:

You upload bad PDFs

OCR extraction fails

You want to load new documents cleanly

🧪 Troubleshooting

OCR is slow

Scanned PDFs require full OCR → use EC2 with more CPU or GPU.

Vectorstore loads slowly

Increase instance size:
c6i.xlarge or m6i.xlarge.

"No usable chunks" error

Document has no extractable text → upload a clearer / higher-quality scan.

"Model does not exist"

OpenRouter expects model names like:

openrouter/openai/gpt-4o-mini


🤝 Contributing

Feel free to open:

Issues

Pull requests

Feature requests

📜 License

MIT License
