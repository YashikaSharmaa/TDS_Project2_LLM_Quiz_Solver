from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import base64
import re

load_dotenv()

app = FastAPI()

YOUR_SECRET = os.getenv("SECRET", "your-secret-here")
YOUR_EMAIL = os.getenv("EMAIL", "your-email@example.com")
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug: Check if keys are loaded
print(f"SECRET loaded: {bool(YOUR_SECRET)}")
print(f"EMAIL loaded: {bool(YOUR_EMAIL)}")
print(f"AIPIPE_API_KEY loaded: {bool(AIPIPE_API_KEY)}")
print(f"OPENAI_API_KEY loaded: {bool(OPENAI_API_KEY)}")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/")
async def handle_quiz(request: QuizRequest):
    if request.secret != YOUR_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    if request.email != YOUR_EMAIL:
        raise HTTPException(status_code=403, detail="Invalid email")
    
    # Start solving the quiz in background
    asyncio.create_task(solve_quiz(request.url, request.email, request.secret))
    
    return {"status": "received", "message": "Processing quiz"}

async def solve_quiz(url: str, email: str, secret: str, max_retries: int = 2):
    """Main quiz solving logic with retry mechanism"""
    try:
        print(f"Starting quiz: {url}")
        
        # Fetch page
        quiz_content = await fetch_quiz_page(url)
        print(f"Fetched quiz content")
        
        # Parse instructions
        instructions = parse_quiz_instructions(quiz_content)
        print(f"Instructions: {instructions[:500]}...")
        
        # Check if personalized
        if "personalized" in instructions.lower():
            print("Note: This is a personalized task")
        elif "not personalized" in instructions.lower():
            print("Note: This task accepts same answer for everyone")
        
        # Submit URL is always the same
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        print(f"Submit URL: {submit_url}")
        
        # Enhanced GitHub URL detection with multiple patterns
        github_patterns = [
            r'https://github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',  # Full URL
            r'github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',  # Without https
            r'(?:repository|repo|GitHub URL|URL|project)[\s:]+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',  # With keywords
            r'\b([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\b',  # Simple username/repo format
        ]
        
        answer = None
        if 'github' in instructions.lower():
            for pattern in github_patterns:
                github_match = re.search(pattern, instructions, re.IGNORECASE)
                if github_match:
                    repo_path = github_match.group(1)
                    if '/' in repo_path and not repo_path.startswith('http'):
                        answer = f"https://github.com/{repo_path}"
                        print(f"Detected GitHub URL pattern, answer: {answer}")
                        break
        
        # If no GitHub URL detected, solve with LLM
        if not answer:
            answer = await solve_with_llm(instructions, url)
        
        # Submit answer
        result = await submit_answer(submit_url, email, secret, url, answer)
        print(f"Submission result: {result}")
        
        # Handle result
        if result.get("correct"):
            print("Answer correct!")
            if result.get("url"):
                await asyncio.sleep(result.get("delay", 1))
                await solve_quiz(result["url"], email, secret)
        else:
            print(f"Answer incorrect: {result.get('reason')}")
            if max_retries > 0:
                print(f"Retrying... ({max_retries} attempts left)")
                await asyncio.sleep(result.get("delay", 1))
                retry_instructions = f"{instructions}\n\nPrevious attempt was wrong: {result.get('reason')}\nThe correct answer should be in this format based on the error."
                answer = await solve_with_llm(retry_instructions, url)
                print(f"Retry answer: {answer}")
                result = await submit_answer(submit_url, email, secret, url, answer)
                print(f"Retry result: {result}")
                if result.get("correct") and result.get("url"):
                    await asyncio.sleep(result.get("delay", 1))
                    await solve_quiz(result["url"], email, secret, max_retries - 1)
            elif result.get("url"):
                await asyncio.sleep(result.get("delay", 1))
                await solve_quiz(result["url"], email, secret, max_retries)
                
    except Exception as e:
        print(f"Error solving quiz: {e}")
        import traceback
        traceback.print_exc()

async def solve_with_llm(instructions: str, url: str) -> any:
    """Solve quiz using LLM with multi-step fetching"""
    max_fetch_attempts = 3
    fetch_count = 0
    
    answer = await solve_with_aipipe(instructions, url)
    print(f"Generated answer: {answer}")
    
    while fetch_count < max_fetch_attempts:
        if isinstance(answer, str) and answer.startswith("DOWNLOAD:"):
            fetch_count += 1
            download_url = answer.replace("DOWNLOAD:", "").strip()
            print(f"LLM requested to download (attempt {fetch_count}): {download_url}")
            
            from urllib.parse import urljoin
            download_url = urljoin(url, download_url)
            print(f"Downloading: {download_url}")
            
            file_data = await download_and_process_file(download_url)
            print(f"File processed, preview: {str(file_data)[:500]}...")
            
            if "CSV Data:" in file_data and "cutoff" in instructions.lower():
                try:
                    answer = await calculate_from_csv(download_url, instructions)
                    print(f"Calculated answer from CSV: {answer}")
                    break
                except Exception as e:
                    print(f"Could not auto-calculate: {e}")
            
            combined_instructions = f"""Original instructions:
{instructions}

Downloaded and processed data from {download_url}:
{file_data}

Now calculate the answer based on the instructions. Return ONLY the final number/value.
DO NOT respond with another DOWNLOAD or FETCH request. Just give me the NUMBER."""
            
            answer = await solve_with_aipipe(combined_instructions, url)
            print(f"Answer after processing file: {answer}")
            
        elif isinstance(answer, str) and answer.startswith("FETCH:"):
            fetch_count += 1
            fetch_url = answer.replace("FETCH:", "").strip()
            print(f"LLM requested to fetch (attempt {fetch_count}): {fetch_url}")
            
            from urllib.parse import urljoin
            fetch_url = urljoin(url, fetch_url)
            print(f"Fetching: {fetch_url}")
            
            fetched_content = await fetch_quiz_page(fetch_url)
            fetched_instructions = parse_quiz_instructions(fetched_content)
            print(f"Fetched content: {fetched_instructions[:500]}...")
            
            combined_instructions = f"""Original instructions:
{instructions}

Fetched content from {fetch_url}:
{fetched_instructions}

Now extract the answer from the fetched content.
DO NOT respond with another DOWNLOAD or FETCH request."""
            
            answer = await solve_with_aipipe(combined_instructions, url)
            print(f"Answer after fetch: {answer}")
        else:
            break
    
    print(f"Final answer: {answer}")
    return answer

async def fetch_quiz_page(url: str) -> str:
    """Fetch and render JavaScript page using Playwright"""
    from playwright.async_api import async_playwright
    
    print(f"Launching browser to render: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to URL...")
        await page.goto(url, wait_until="networkidle")
        
        print("Waiting for JavaScript to execute...")
        await page.wait_for_timeout(3000)
        
        result_div = await page.query_selector("#result")
        if result_div:
            result_text = await result_div.inner_text()
            print(f"Found #result div with content: {result_text[:200]}...")
        else:
            print("No #result div found")
        
        print("Extracting page content...")
        content = await page.content()
        await browser.close()
        
        print(f"Content extracted, length: {len(content)} chars")
        return content

def parse_quiz_instructions(html_content: str) -> str:
    """Extract quiz instructions from HTML"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result_div = soup.find(id="result")
    if result_div:
        text = result_div.get_text(strip=True)
        if text:
            return text
    
    scripts = soup.find_all('script')
    for script in scripts:
        script_text = script.string
        if script_text and 'atob' in script_text:
            base64_match = re.search(r'atob\([`\'"]([A-Za-z0-9+/=\s]+)[`\'"]', script_text)
            if base64_match:
                try:
                    base64_content = base64_match.group(1).replace('\n', '').replace(' ', '')
                    decoded = base64.b64decode(base64_content).decode('utf-8')
                    return decoded
                except Exception as e:
                    print(f"Failed to decode base64: {e}")
    
    body = soup.find("body")
    if body:
        return body.get_text(strip=True)
    
    return html_content

async def calculate_from_csv(csv_url: str, instructions: str) -> int:
    """Direct calculation from CSV when cutoff is mentioned"""
    import pandas as pd
    from io import StringIO
    
    cutoff_match = re.search(r'cutoff[:\s]+(\d+)', instructions, re.IGNORECASE)
    if not cutoff_match:
        raise ValueError("No cutoff value found")
    
    cutoff = int(cutoff_match.group(1))
    print(f"Found cutoff value: {cutoff}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(csv_url) as response:
            content = await response.read()
            df = pd.read_csv(StringIO(content.decode('utf-8')), header=None)
            
            numbers = df[0].tolist()
            print(f"Found {len(numbers)} numbers in CSV")
            
            result = sum(n for n in numbers if n > cutoff)
            print(f"Sum of numbers > {cutoff}: {result}")
            
            return result

async def download_and_process_file(url: str) -> str:
    """Download and process files (CSV, PDF, Excel, etc.)"""
    import pandas as pd
    from io import BytesIO, StringIO
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
            content_type = response.headers.get('Content-Type', '')
            
            if 'csv' in content_type.lower() or url.endswith('.csv'):
                try:
                    df = pd.read_csv(StringIO(content.decode('utf-8')), header=None)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    return f"CSV Data:\nColumns: {list(df.columns)}\nRows: {len(df)}\nFull Data (all rows):\n{df.to_string(index=False)}"
                except Exception as e:
                    return f"Error parsing CSV: {e}\nRaw content: {content.decode('utf-8')[:1000]}"
            
            elif 'excel' in content_type.lower() or url.endswith(('.xlsx', '.xls')):
                try:
                    df = pd.read_excel(BytesIO(content))
                    return f"Excel Data:\nColumns: {list(df.columns)}\nRows: {len(df)}\nFull Data:\n{df.to_string()}"
                except Exception as e:
                    return f"Error parsing Excel: {e}"
            
            elif 'pdf' in content_type.lower() or url.endswith('.pdf'):
                try:
                    import pdfplumber
                    with pdfplumber.open(BytesIO(content)) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                        return f"PDF Content:\n{text[:2000]}"
                except Exception as e:
                    return f"Error parsing PDF: {e}"
            
            elif 'json' in content_type.lower() or url.endswith('.json'):
                return content.decode('utf-8')
            
            elif 'html' in content_type.lower():
                from bs4 import BeautifulSoup
                html_text = content.decode('utf-8')
                soup = BeautifulSoup(html_text, 'html.parser')
                
                file_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls', '.pdf', '.json']):
                        file_links.append(f"{link.get_text()}: {href}")
                
                cutoff_span = soup.find(id='cutoff')
                cutoff_value = cutoff_span.get_text() if cutoff_span else "Not found"
                
                result = f"HTML Page Content:\n"
                if file_links:
                    result += f"Found file links:\n" + "\n".join(file_links) + "\n\n"
                result += f"Cutoff value: {cutoff_value}\n\n"
                result += f"Full text: {soup.get_text()[:1000]}"
                return result
            
            else:
                return content.decode('utf-8')[:2000]

async def solve_with_aipipe(instructions: str, quiz_url: str) -> any:
    """Use LLM API to solve the quiz"""
    
    system_prompt = """You are an expert data analyst and web scraper.

Your task:
1. Read the quiz instructions carefully and understand what answer format is required
2. **CRITICAL: If you see a GitHub repository reference (like "username/repo" or "sanand0/tools"), you MUST return it as a full URL: https://github.com/username/repo**
3. If instructions mention downloading a file: tell me "DOWNLOAD: <file_url>"
4. If instructions ask to scrape a webpage: tell me "FETCH: <page_url>"
5. If you have all data needed, return ONLY the final answer value

Answer Format Examples:
- GitHub URL: If you see "sanand0/tools" → return "https://github.com/sanand0/tools"
- GitHub URL: If you see "username/repo" → return "https://github.com/username/repo"
- Number: 12345
- Text: "SECRET123"
- Boolean: true or false
- DO NOT return JSON submission payload format

**IMPORTANT**: Always convert repository paths to full GitHub URLs!"""

    user_prompt = f"""Quiz instructions:

{instructions}

Current URL: {quiz_url}

CRITICAL: Read carefully and identify what is being asked.

Common question types:
- GitHub repository (like "sanand0/tools"): Return https://github.com/sanand0/tools
- GitHub URL question: Always format as https://github.com/username/repo
- Download file: respond "DOWNLOAD: <url>"
- Scrape page: respond "FETCH: <url>"
- Calculation: provide the number
- Text/code: provide the exact text

Extract the answer. Return ONLY the answer value in the correct format.

Response:"""

    if AIPIPE_API_KEY:
        try:
            return await call_aipipe(system_prompt, user_prompt)
        except Exception as e:
            print(f"AI/Pipe failed: {e}, falling back to OpenAI")
    
    if OPENAI_API_KEY:
        return await call_openai(system_prompt, user_prompt)
    
    raise ValueError("No API key configured")

async def call_aipipe(system_prompt: str, user_prompt: str) -> any:
    """Call AI/Pipe API via OpenRouter"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://aipipe.org/openrouter/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {AIPIPE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"AI/Pipe API error: {response.status} - {error_text}")
            result = await response.json()
            return parse_llm_response(result["choices"][0]["message"]["content"])

async def call_openai(system_prompt: str, user_prompt: str) -> any:
    """Call OpenAI API directly"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")
            result = await response.json()
            return parse_llm_response(result["choices"][0]["message"]["content"])

def parse_llm_response(answer_text: str) -> any:
    """Parse LLM response into appropriate type"""
    answer_text = answer_text.strip()
    
    # Remove markdown code blocks
    if answer_text.startswith('```'):
        lines = answer_text.split('\n')
        answer_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else answer_text
        answer_text = answer_text.strip()
    
    # Remove "Answer:" prefix
    if answer_text.lower().startswith('answer:'):
        answer_text = answer_text[7:].strip()
    
    # Check for GitHub URL pattern and ensure it's properly formatted
    github_pattern = r'(?:https://github\.com/)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)'
    github_match = re.search(github_pattern, answer_text)
    if github_match and ('github' in answer_text.lower() or '/' in answer_text):
        repo_path = github_match.group(1)
        if '/' in repo_path:
            return f"https://github.com/{repo_path}"
    
    # Handle JSON responses
    if answer_text.startswith('{') or answer_text.startswith('['):
        try:
            parsed = json.loads(answer_text)
            if isinstance(parsed, dict) and all(k in parsed for k in ['email', 'secret', 'url']):
                if 'answer' in parsed:
                    return parsed['answer']
            return parsed
        except:
            pass
    
    # Boolean
    if answer_text.lower() in ['true', 'false']:
        return answer_text.lower() == 'true'
    
    # Number
    try:
        cleaned = ''.join(c for c in answer_text if c.isdigit() or c in '.-')
        if cleaned:
            if '.' in cleaned:
                return float(cleaned)
            return int(cleaned)
    except:
        pass
    
    return answer_text

async def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: any) -> dict:
    """Submit answer to the endpoint"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    print(f"Submitting to {submit_url}: {payload}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(submit_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response_text = await response.text()
                
                if response.status == 404:
                    print(f"404 Error - URL not found: {submit_url}")
                    print(f"Response: {response_text}")
                    submit_url = "https://tds-llm-analysis.s-anand.net/submit"
                    print(f"Retrying with: {submit_url}")
                    async with session.post(submit_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as retry_response:
                        return await retry_response.json()
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response: {response_text}")
                    return {"correct": False, "reason": "Invalid response format"}
                    
        except Exception as e:
            print(f"Error submitting answer: {e}")
            return {"correct": False, "reason": str(e)}

@app.get("/")
async def root():
    return {"status": "Quiz solver is running"}
    
