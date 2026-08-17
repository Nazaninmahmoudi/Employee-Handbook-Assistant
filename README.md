# Employee Handbook Assistant
An AI-powered HR buddy that helps employees get quick answers from company policies and documents.
## Project Overview

This project is a multi-media chatbot buddy that helps employees quickly find answers from company policies and internal documents. The data source used in this project is the 37signals Employee Handbook, which includes information about company culture, benefits, workplace policies, and employee guidelines.

The system uses a Hybrid Retrieval approach, which combines two different retrieval methods to improve search accuracy. The first method is **semantic search using BAAI/bge-base-en-v1.5 embeddings with ChromaDB, which helps the system understand the meaning of user queries. The second method is keyword-based search using **BM25, which focuses on finding exact terms from the documents. These two approaches are combined using a custom Hybrid RRF (Reciprocal Rank Fusion) method to provide more relevant document retrieval.

The chatbot uses Llama 3.3 70B through Groq API with LangChain Memory for maintaining conversation context. Kokoro TTS is also integrated to convert generated responses into voice. Finally , The chatbot interface is built using Gradio.

## Installation

Clone the repository:

```bash
git clone https://github.com/Nazaninmahmoudi/employee-handbook-assistant.git
```

Navigate to the project directory:

```bash
cd employee-handbook-assistant
```

Install the required dependencies:

```bash
uv sync
```

Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
python app/app.py
```

## Technologies Used

- Python
- LangChain
- Groq API
- Llama 3.3 70B
- BGE Embeddings
- ChromaDB
- BM25
- Hybrid RRF Retrieval
- Gradio
- Kokoro TTS

- ## Contact

If you have any questions or suggestions, feel free to reach out:

- Email: Nazaninmahmoudy@gmail.com
- LinkedIn: www.linkedin.com/in/nazanin-mahmoudi-495a3a247
- Kaggle: https://www.kaggle.com/nazaninmahmoudy


## License

This project is licensed under the MIT License.
