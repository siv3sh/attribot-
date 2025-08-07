# HR Assistant Web App

This is an intelligent HR assistant web application built using LangChain, Gemini LLM, ChromaDB, and Streamlit. It is designed to analyze employee attrition data and answer HR-related queries from uploaded documents.

## 🚀 Features

- Upload HR documents (PDF, TXT, DOCX)
- Ask questions from uploaded HR policies (RAG using ChromaDB + Gemini)
- Attrition data visualizations with filtering
- Secure, fast, and reliable web interface

## 🧠 Technologies Used

- **LangChain** – For chaining LLM-based tasks
- **Gemini LLM** – To generate answers for HR queries
- **ChromaDB** – Local vector store for document retrieval
- **Streamlit** – For building the frontend
- **Pandas** – For HR analytics

## 📦 Installation

```bash
git clone https://github.com/siv3sh/attribot.git
cd attribot
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure

```
├── app.py                # Main Streamlit app
├── models/               # LLM and embedding wrappers
├── utils/                # Helper functions for file processing and RAG
├── data/                 # Sample datasets (CSV, documents)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🧩 Use Cases

- HR teams in companies with large-scale employee data
- Startups aiming to automate HR query resolution
- Managers seeking insights on attrition trends

## 📬 Contact

For any queries, please reach out to `siv3sh@gmail.com`

---
© 2025 HR Assistant App. All rights reserved.
