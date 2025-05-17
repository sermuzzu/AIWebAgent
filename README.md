# Minimal AI Agent Web Chat

This project is a minimal FastAPI-based web chat interface for an AI agent, using only three files:

- `web_chat.py`: FastAPI backend that relays chat to the agent logic in `Example.py`.
- `Example.py`: Contains the `Agent`, `Memory`, `ContextManager`, and `Planner` classes. All agent logic and context management is here.
- `static/index.html`: Simple HTML/JavaScript chat frontend.

## How to Run

1. **Install dependencies:**
   ```powershell
   pip install fastapi uvicorn openai
   ```
2. **Set your Azure OpenAI environment variables** (if using real LLM):
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_KEY`
   - `AZURE_OPENAI_API_VERSION`
   - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - `AZURE_OPENAI_CHAT_DEPLOYMENT`

   *(The provided Example.py uses a dummy reply by default. Uncomment the OpenAI code for real LLM integration.)*

3. **Start the server:**
   ```powershell
   uvicorn web_chat:app --reload
   ```

4. **Open your browser:**
   Go to [http://localhost:8000](http://localhost:8000)

## Project Structure

- `web_chat.py` - FastAPI backend, relays chat to Example.py
- `Example.py` - Agent, Memory, ContextManager, Planner
- `static/index.html` - Simple chat UI

## Notes
- All conversation state is managed by the `Agent` class in memory.
- The web API is a thin relay to the agent logic.
- For production, add authentication and persistent storage.
