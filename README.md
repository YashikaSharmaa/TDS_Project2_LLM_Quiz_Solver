# LLM Quiz Solver

An automated quiz-solving application using LLMs for data analysis tasks.

## Features
- Accepts quiz tasks via API endpoint
- Renders JavaScript-based quiz pages using Playwright
- Solves data-related questions using AI/Pipe API
- Handles multiple quiz types: PDFs, CSVs, web scraping, analysis, visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-quiz-solver.git
cd llm-quiz-solver
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

4. Create `.env` file:
```env
SECRET=your-secret-string
EMAIL=your-email@example.com
AIPIPE_API_KEY=your-aipipe-key
```

5. Run locally:
```bash
uvicorn main:app --reload
```

## Usage

Send POST request to your endpoint:
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "secret": "your-secret",
    "url": "https://example.com/quiz"
  }'
```

## Deployment

Deployed on Render.com. See `render.yaml` for configuration.

## License

MIT License - see LICENSE file for details.