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
        
        # Step 4: Solve the quiz using AI/Pipe
        answer = await solve_with_aipipe(instructions, url)
        print(f"Generated answer: {answer}")
        
        # Check if LLM wants to fetch another URL
        if isinstance(answer, str) and answer.startswith("FETCH:"):
            fetch_url = answer.replace("FETCH:", "").strip()
            print(f"LLM requested to fetch: {fetch_url}")
            
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

Now extract the answer from the fetched content."""
            
            answer = await solve_with_aipipe(combined_instructions, url)
            print(f"Final answer after fetch: {answer}")
        
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

def extract_submit_url(instructions: str, current_url: str = None) -> str:
    """Extract submit URL from instructions"""
    # Look for URLs in the instructions
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, instructions)
    
    # Find the submit URL (usually contains 'submit')
    for url in urls:
        if 'submit' in url.lower():
            return url
    
    # Check for relative URLs like /submit or POST to /submit
    relative_pattern = r'(?:POST|to|back to)\s+([/\w-]+)'
    relative_matches = re.findall(relative_pattern, instructions, re.IGNORECASE)
    
    for match in relative_matches:
        if 'submit' in match.lower():
            # Convert relative URL to absolute
            if current_url:
                from urllib.parse import urljoin
                return urljoin(current_url, match)
            else:
                # Fallback to common base URL
                return f"https://tds-llm-analysis.s-anand.net{match}"
    
    # If no submit URL found, return the last URL or use default
    if urls:
        return urls[-1]
    
    # Default fallback
    return "https://tds-llm-analysis.s-anand.net/submit"

async def solve_with_aipipe(instructions: str, quiz_url: str) -> any:
    """Use LLM API to solve the quiz (tries AI/Pipe first, falls back to OpenAI)"""
    
    system_prompt = """You are an expert data analyst and web scraper. 

Your task:
1. Read the quiz instructions carefully
2. If it asks to scrape/visit another URL, you MUST tell me to fetch that URL
3. Extract the required information
4. Return ONLY the final answer value

CRITICAL: Return ONLY the answer value itself, nothing else.
- If answer is a number: return just the number (e.g., 12345)
- If answer is text: return just the text (e.g., "SECRET123")
- If answer is boolean: return just true or false
- DO NOT return JSON with email/secret/url
- DO NOT include explanations

If the instructions mention scraping or visiting another page, your response should be:
FETCH: <the URL to fetch>

After I provide the fetched content, extract the answer from it."""

    user_prompt = f"""Quiz instructions:

{instructions}

Current URL: {quiz_url}

What should I do? If there's a URL to scrape/fetch, respond with "FETCH: <url>". Otherwise, provide the answer directly.

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
        async with session.post(submit_url, json=payload) as response:
            return await response.json()

@app.get("/")
async def root():
    return {"status": "Quiz solver is running"}

# No if __name__ == "__main__" block needed for Render