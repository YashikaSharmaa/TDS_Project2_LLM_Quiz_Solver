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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fallback to OpenAI

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
        
        # Step 1: Fetch and render the quiz page
        quiz_content = await fetch_quiz_page(url)
        print(f"Fetched quiz content")
        
        # Step 2: Parse instructions
        instructions = parse_quiz_instructions(quiz_content)
        print(f"Instructions: {instructions[:500]}...")  # Print first 500 chars
        
        # Step 3: Extract submit URL from instructions
        submit_url = extract_submit_url(instructions, url)
        print(f"Submit URL: {submit_url}")
        
async def solve_quiz_with_llm(instructions: str, url: str) -> any:
    """Solve quiz using LLM with multi-step fetching"""
    max_fetch_attempts = 3
    fetch_count = 0
    
    answer = await solve_with_aipipe(instructions, url)
    print(f"Generated answer: {answer}")
    
    while fetch_count < max_fetch_attempts:
        # Check if LLM wants to download a file
        if isinstance(answer, str) and answer.startswith("DOWNLOAD:"):
                fetch_count += 1
                download_url = answer.replace("DOWNLOAD:", "").strip()
                print(f"LLM requested to download (attempt {fetch_count}): {download_url}")
                
                # Make download_url absolute if relative
                from urllib.parse import urljoin
                download_url = urljoin(url, download_url)
                print(f"Downloading: {download_url}")
                
                # Download and process the file
                file_data = await download_and_process_file(download_url)
                print(f"File processed, preview: {str(file_data)[:500]}...")
                
                # If it's CSV data, try to calculate answer directly
                if "CSV Data:" in file_data and "cutoff" in instructions.lower():
                    try:
                        answer = await calculate_from_csv(download_url, instructions)
                        print(f"Calculated answer from CSV: {answer}")
                        break
                    except Exception as e:
                        print(f"Could not auto-calculate: {e}")
                
                # Ask LLM to analyze the data
                combined_instructions = f"""Original instructions:
{instructions}

Downloaded and processed data from {download_url}:
{file_data}

Now calculate the answer based on the instructions. Return ONLY the final number/value.
DO NOT respond with another DOWNLOAD or FETCH request. Just give me the NUMBER."""
                
                answer = await solve_with_aipipe(combined_instructions, url)
                print(f"Answer after processing file: {answer}")
                
            # Check if LLM wants to fetch another URL
            elif isinstance(answer, str) and answer.startswith("FETCH:"):
                fetch_count += 1
                fetch_url = answer.replace("FETCH:", "").strip()
                print(f"LLM requested to fetch (attempt {fetch_count}): {fetch_url}")
                
                # Make fetch_url absolute if relative
                from urllib.parse import urljoin
                fetch_url = urljoin(url, fetch_url)
                print(f"Fetching: {fetch_url}")
                
                # Fetch the additional page
                fetched_content = await fetch_quiz_page(fetch_url)
                fetched_instructions = parse_quiz_instructions(fetched_content)
                print(f"Fetched content: {fetched_instructions[:500]}...")
                
                # Ask LLM again with the fetched content
                combined_instructions = f"""Original instructions:
{instructions}

Fetched content from {fetch_url}:
{fetched_instructions}

Now extract the answer from the fetched content.
DO NOT respond with another DOWNLOAD or FETCH request."""
                
                answer = await solve_with_aipipe(combined_instructions, url)
                print(f"Answer after fetch: {answer}")
            else:
                # Got a real answer, break the loop
                break
        
        # Step 5: Submit the answer
        result = await submit_answer(submit_url, email, secret, url, answer)
        print(f"Submission result: {result}")
        
        # Step 6: Handle result
        if result.get("correct"):
            print("Answer correct!")
            if result.get("url"):
                # Recursively solve next quiz
                await solve_quiz(result["url"], email, secret)
        else:
            print(f"Answer incorrect: {result.get('reason')}")
            # Retry if we have attempts left
            if max_retries > 0:
                print(f"Retrying... ({max_retries} attempts left)")
                answer = await solve_with_aipipe(
                    f"{instructions}\n\nPrevious attempt was wrong: {result.get('reason')}",
                    url
                )
                result = await submit_answer(submit_url, email, secret, url, answer)
                if result.get("correct") and result.get("url"):
                    await solve_quiz(result["url"], email, secret)
            elif result.get("url"):
                # Skip to next question if provided
                await solve_quiz(result["url"], email, secret)
                
    except Exception as e:
        print(f"Error solving quiz: {e}")
        import traceback
        traceback.print_exc()

async def solve_quiz_with_llm(instructions: str, url: str) -> any:
    """Solve quiz using LLM with multi-step fetching"""
    max_fetch_attempts = 3
    fetch_count = 0
    
    answer = await solve_with_aipipe(instructions, url)
    print(f"Generated answer: {answer}")
    
    while fetch_count < max_fetch_attempts:
        # Check if LLM wants to download a file
        if isinstance(answer, str) and answer.startswith("DOWNLOAD:"):
    """Fetch and render JavaScript page using Playwright"""
    from playwright.async_api import async_playwright
    
    print(f"Launching browser to render: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to URL...")
        await page.goto(url, wait_until="networkidle")
        
        print("Waiting for JavaScript to execute...")
        await page.wait_for_timeout(3000)  # Wait for JS execution
        
        # Check if #result div exists and has content
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

async def download_and_process_file(url: str) -> str:
    """Download and process files (CSV, PDF, Excel, etc.)"""
    import aiohttp
    import pandas as pd
    from io import BytesIO, StringIO
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
            content_type = response.headers.get('Content-Type', '')
            
            # Handle CSV files
            if 'csv' in content_type.lower() or url.endswith('.csv'):
                try:
                    # First try without headers
                    df = pd.read_csv(StringIO(content.decode('utf-8')), header=None)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    
                    # If first row looks like data (all numbers), use it
                    return f"CSV Data:\nColumns: {list(df.columns)}\nRows: {len(df)}\nFull Data (all rows):\n{df.to_string(index=False)}"
                except Exception as e:
                    return f"Error parsing CSV: {e}\nRaw content: {content.decode('utf-8')[:1000]}"
            
            # Handle Excel files
            elif 'excel' in content_type.lower() or url.endswith(('.xlsx', '.xls')):
                try:
                    df = pd.read_excel(BytesIO(content))
                    return f"Excel Data:\nColumns: {list(df.columns)}\nRows: {len(df)}\nFull Data:\n{df.to_string()}"
                except Exception as e:
                    return f"Error parsing Excel: {e}"
            
            # Handle PDF files
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
            
            # Handle JSON
            elif 'json' in content_type.lower() or url.endswith('.json'):
                return content.decode('utf-8')
            
            # Handle HTML (extract links to files)
            elif 'html' in content_type.lower():
                from bs4 import BeautifulSoup
                html_text = content.decode('utf-8')
                soup = BeautifulSoup(html_text, 'html.parser')
                
                # Find all links to files
                file_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls', '.pdf', '.json']):
                        file_links.append(f"{link.get_text()}: {href}")
                
                # Extract cutoff value if present
                cutoff_span = soup.find(id='cutoff')
                cutoff_value = cutoff_span.get_text() if cutoff_span else "Not found"
                
                result = f"HTML Page Content:\n"
                if file_links:
                    result += f"Found file links:\n" + "\n".join(file_links) + "\n\n"
                result += f"Cutoff value: {cutoff_value}\n\n"
                result += f"Full text: {soup.get_text()[:1000]}"
                return result
            
            # Default: return as text
            else:
                return content.decode('utf-8')[:2000]

def parse_quiz_instructions(html_content: str) -> str:
    """Extract quiz instructions from HTML"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find result div (should already have decoded content after JS execution)
    result_div = soup.find(id="result")
    if result_div:
        text = result_div.get_text(strip=True)
        if text:
            return text
    
    # Fallback: Try to find and decode base64 content in script tags
    scripts = soup.find_all('script')
    for script in scripts:
        script_text = script.string
        if script_text and 'atob' in script_text:
            # Try to extract base64 content
            import re
            base64_match = re.search(r'atob\([`\'"]([A-Za-z0-9+/=\s]+)[`\'"]', script_text)
            if base64_match:
                try:
                    base64_content = base64_match.group(1).replace('\n', '').replace(' ', '')
                    decoded = base64.b64decode(base64_content).decode('utf-8')
                    return decoded
                except Exception as e:
                    print(f"Failed to decode base64: {e}")
    
    # Final fallback to body text
    body = soup.find("body")
    if body:
        return body.get_text(strip=True)
    
    return html_content

async def calculate_from_csv(csv_url: str, instructions: str) -> int:
    """Direct calculation from CSV when cutoff is mentioned"""
    import aiohttp
    import pandas as pd
    from io import StringIO
    import re
    
    # Extract cutoff value from instructions
    cutoff_match = re.search(r'cutoff[:\s]+(\d+)', instructions, re.IGNORECASE)
    if not cutoff_match:
        raise ValueError("No cutoff value found")
    
    cutoff = int(cutoff_match.group(1))
    print(f"Found cutoff value: {cutoff}")
    
    # Download CSV
    async with aiohttp.ClientSession() as session:
        async with session.get(csv_url) as response:
            content = await response.read()
            df = pd.read_csv(StringIO(content.decode('utf-8')), header=None)
            
            # Get all numeric values
            numbers = df[0].tolist()
            print(f"Found {len(numbers)} numbers in CSV")
            
            # Calculate sum of numbers greater than cutoff
            result = sum(n for n in numbers if n > cutoff)
            print(f"Sum of numbers > {cutoff}: {result}")
            
            return result

def extract_submit_url(instructions: str, current_url: str = None) -> str:
    """Extract submit URL from instructions"""
    # Always use the known correct submit endpoint
    # The instructions often have malformed URLs like "submitJSON" or "submitwith"
    return "https://tds-llm-analysis.s-anand.net/submit"

async def solve_with_aipipe(instructions: str, quiz_url: str) -> any:
    """Use LLM API to solve the quiz (tries AI/Pipe first, falls back to OpenAI)"""
    
    system_prompt = """You are an expert data analyst and web scraper with file processing capabilities.

Your task:
1. Read the quiz instructions carefully and understand what answer format is required
2. If instructions mention downloading a file (CSV, PDF, Excel, image), tell me: DOWNLOAD: <file_url>
3. If instructions ask to scrape a webpage, tell me: FETCH: <page_url>
4. If you already have all the data needed, extract and return ONLY the final answer value

Answer Format (READ CAREFULLY):
- If asked for a URL: Return the full URL (e.g., https://github.com/user/repo)
- If asked for a number: Return just the number (e.g., 12345)
- If asked for text/string: Return just the text (e.g., "SECRET123")
- If asked for boolean: Return true or false
- If asked for a GitHub URL: Return complete GitHub URL starting with https://github.com/
- DO NOT return submission payload format (no email/secret/url fields)

File Processing:
- To download a file: "DOWNLOAD: <full_url>"
- To fetch a webpage: "FETCH: <full_url>"
- To provide answer: Just the answer value in the required format"""

    user_prompt = f"""Quiz instructions:

{instructions}

Current URL: {quiz_url}

IMPORTANT: Read the instructions carefully and identify what is being asked.

Common question types:
- If asking for a GitHub URL (like "username/repo"): Return the full URL in format https://github.com/username/repo
- If asking to download a file: respond "DOWNLOAD: <url>"
- If asking to scrape a page: respond "FETCH: <url>"
- If asking for a calculation: provide the number
- If asking for text/code: provide the exact text

Extract the answer from the instructions. Return ONLY the answer value in the correct format.

Response:"""

    # Try AI/Pipe first
    if AIPIPE_API_KEY:
        try:
            return await call_aipipe(system_prompt, user_prompt)
        except Exception as e:
            print(f"AI/Pipe failed: {e}, falling back to OpenAI")
    
    # Fallback to OpenAI
    if OPENAI_API_KEY:
        return await call_openai(system_prompt, user_prompt)
    
    raise ValueError("No API key configured (AIPIPE_API_KEY or OPENAI_API_KEY)")

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
                "model": "openai/gpt-4o-mini",  # or "anthropic/claude-3.5-sonnet"
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
    
    # Remove markdown code blocks if present
    if answer_text.startswith('```'):
        lines = answer_text.split('\n')
        answer_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else answer_text
        answer_text = answer_text.strip()
    
    # Remove "Answer:" prefix if present
    if answer_text.lower().startswith('answer:'):
        answer_text = answer_text[7:].strip()
    
    # Try to parse as JSON (but filter out submission payload format)
    if answer_text.startswith('{') or answer_text.startswith('['):
        try:
            parsed = json.loads(answer_text)
            # Check if it's the submission payload format (has email, secret, url keys)
            if isinstance(parsed, dict) and all(k in parsed for k in ['email', 'secret', 'url']):
                # Extract just the answer field
                if 'answer' in parsed:
                    return parsed['answer']
            # Otherwise return the parsed JSON
            return parsed
        except:
            pass
    
    # Try to parse as boolean
    if answer_text.lower() in ['true', 'false']:
        return answer_text.lower() == 'true'
    
    # Try to parse as number
    try:
        # Remove any non-numeric characters except . and -
        cleaned = ''.join(c for c in answer_text if c.isdigit() or c in '.-')
        if cleaned:
            if '.' in cleaned:
                return float(cleaned)
            return int(cleaned)
    except:
        pass
    
    # Return as string
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
                
                # Check if response is JSON
                if response.status == 404:
                    print(f"404 Error - URL not found: {submit_url}")
                    print(f"Response: {response_text}")
                    # Try default submit URL
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

# No if __name__ == "__main__" block needed for Render
