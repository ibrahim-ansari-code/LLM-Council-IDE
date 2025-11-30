# Better Cursor - LLM Council IDE

A VS Code-like IDE powered by an LLM Council (inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)). Instead of using a single LLM, this IDE uses a council of multiple LLMs that review and rank each other's responses, with a Chairman LLM synthesizing the final answer.

## Features
  1. **First Opinions**: All council LLMs provide initial responses
  2. **Review**: Each LLM reviews and ranks other responses
  3. **Final Response**: Chairman LLM synthesizes the best insights

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- OpenRouter API key
### Installation

1. **Clone or download this repository**

2. **Set up the backend**:
   ```bash
   # Install Python dependencies (using uv or pip)
   pip install -r requirements.txt
   # OR if you have uv:
   uv sync
   ```

3. **Set up the frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure your API key**:
   ```bash
   # Copy the example env file
   cp env.example .env
   
   # Edit .env and add your OpenRouter API key
   # OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   
   # Optional: Adjust max tokens based on your credits
   # Lower values = less credits needed, but shorter responses
   # MAX_TOKENS=256  # Default (works with low credits)
   # MAX_TOKENS=512  # Medium
   # MAX_TOKENS=2048 # High (needs more credits)
   ```

5. **Configure models (important)**:
   
   **You must verify model names are available on OpenRouter!**
   
   - Visit https://openrouter.ai/models to see available models
   - Check the exact model ID format (e.g., `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)
   - Edit `backend/config.py` to use models that exist:
   ```python
   COUNCIL_MODELS = [
       "openai/gpt-4o",              # Verify this exists
       "anthropic/claude-3.5-sonnet", # Verify this exists
       "openai/gpt-3.5-turbo",       # Fallback option
   ]
   
   CHAIRMAN_MODEL = "openai/gpt-4o"  # Use a reliable model
   ```
   
   **Note**: Model names change frequently. If you get 404 errors, check OpenRouter's model list and update the config.

## Running the Application

### Option 1: Use the start script

```bash
chmod +x start.sh
./start.sh
```

### Option 2: Run manually

**Terminal 1 (Backend)**:
```bash
# Load environment variables and run backend
export $(cat .env | xargs)
python -m backend.main
# OR with uv:
uv run python -m backend.main
```

**Terminal 2 (Frontend)**:
```bash
cd frontend
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

## Usage

1. **Chat with LLM Council**: 
   - Click the chat panel on the right (or the chat button if hidden)
   - Type your question and press Cmd/Ctrl + Enter to send
   - The council will discuss and provide a final answer
   - You can view individual LLM responses by clicking the tabs

2. **View Individual Responses**: 
   - After receiving a response, click on the model tabs to see what each LLM said
   - The "Final Response" tab shows the Chairman's synthesized answer

## Project Structure

```
better-cursor/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Configuration (models, API keys)
│   └── main.py             # FastAPI backend with LLM Council logic
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── App.css         # Styles
│   │   ├── main.jsx        # React entry point
│   │   └── index.css       # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── conversations/      # Saved conversations (auto-created)
├── .env                    # Your API keys (create from env.example)
├── env.example             # Example environment file
├── pyproject.toml          # Python dependencies
├── README.md
└── start.sh                # Startup script
```

## Tech Stack

- **Backend**: FastAPI (Python), OpenRouter API, async httpx
- **Frontend**: React + Vite, Monaco Editor, react-markdown
- **Storage**: JSON files in `data/conversations/`

## Customization

### Change Council Models

Edit `backend/config.py`:
```python
COUNCIL_MODELS = [
    "openai/gpt-4o",
    "google/gemini-2.0-flash-exp",
    # Add your preferred models
]
```

### Change Chairman Model

Edit `backend/config.py`:
```python
CHAIRMAN_MODEL = "google/gemini-2.0-flash-exp"
```

### Change Ports

- Backend: Edit `backend/config.py` (default: 8000)
- Frontend: Edit `frontend/vite.config.js` (default: 5173)

## Notes

- Make sure you have credits on OpenRouter or have automatic top-up enabled
- Conversations are saved in `data/conversations/` as JSON files
- The backend loads environment variables from `.env` file

## License

This project is provided as-is for inspiration and learning purposes.

