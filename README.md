# langchain-agentic-llm

🧠 Chain-of-Thought LLM Agent with FastAPI + Streamlit
This project implements a streaming LLM agent powered by LangChain, FastAPI, and Streamlit. It showcases:

🧩 Chain-of-Thought reasoning via custom system prompts

🛠️ Tool usage (e.g., search, image generation, web search ,location, math) using LangChain tools

🌐 Streaming responses from FastAPI to Streamlit via SSE (Server-Sent Events)

🖼️ Image generation with Stability AI (SDXL), dynamically rendered in the frontend

💬 Chat-like history UI powered by Streamlit session state

🔧 Tech Stack
LangChain (LLM + agent orchestration)

FastAPI (backend + streaming API)

Streamlit (frontend UI)

Stability SDK (for image generation)


🚀 How to Run
Follow these steps to get the Chain-of-Thought LLM Agent running locally:

1. Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

3. Create a virtual environment (optional but recommended)
4. 
python -m venv venv
source venv/bin/activate    # On Windows powershell : venv\Scripts\Activate.ps1
5. Install dependencies
pip install -r requirements.txt

7. Set API keys (if needed)
If you’re using external tools (e.g., SerpAPI, StabilityAI), set your API keys in the environment or directly in code (not recommended for production):

export SERPAPI_API_KEY="your_key_here"
export STABILITY_API_KEY="your_key_here"

5. Run the FastAPI backend


uvicorn main:app --reload --port 8001
6. Run the Streamlit frontend
In a new terminal tab or window:

streamlit run app.py or  python -m streamlit run frontend.py
7. Open the app in your browser
Go to: http://localhost:8501

