"""
Main FastAPI application for the LLM Council IDE backend.
"""
import asyncio
import json
import os
import subprocess
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from backend.config import (
    CHAIRMAN_MODEL,
    COUNCIL_MODELS,
    CORS_ORIGINS,
    MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

app = FastAPI(title="LLM Council IDE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/conversations", exist_ok=True)
os.makedirs("workspace", exist_ok=True)


class CodeCompletionRequest(BaseModel):
    """Request for code completion."""
    code: str
    cursor_position: int
    language: Optional[str] = None
    file_path: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for chat with LLM Council."""
    message: str
    conversation_id: Optional[str] = None
    context: Optional[str] = None
    write_to_file: Optional[str] = None


class LLMResponse(BaseModel):
    """Response from a single LLM."""
    model: str
    response: str
    ranking: Optional[int] = None


class CouncilResponse(BaseModel):
    """Response from the LLM Council."""
    individual_responses: List[LLMResponse]
    reviews: List[Dict]
    final_response: str
    conversation_id: str
    file_written: Optional[str] = None
    files_written: Optional[List[str]] = None


async def call_llm(
    model: str, messages: List[Dict], temperature: float = 0.7
) -> str:
    """Call a single LLM via OpenRouter."""
    # Sometimes the API can be slow, so we give it plenty of time
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/better-cursor",
                    "X-Title": "Better Cursor IDE",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": MAX_TOKENS,
                },
            )
            response.raise_for_status()
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("No choices in response")
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        raise Exception(f"OpenRouter API error for {model}: {error_detail}")
    except Exception as e:
        raise Exception(f"Error calling {model}: {str(e)}")


async def get_initial_responses(query: str, context: Optional[str] = None, write_to_file: Optional[str] = None) -> List[LLMResponse]:
    """Get initial responses from all council models - each one gives their take on the problem."""
    system_prompt = "You are a helpful coding assistant. Provide clear, accurate, and helpful responses."
    
    if context:
        import re
        file_matches = re.findall(r'File:\s*([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))', context, re.IGNORECASE)
        files_in_context = list(set(file_matches))
    else:
        files_in_context = []
    
    if write_to_file:
        system_prompt += f"""

CRITICAL FILE CREATION INSTRUCTIONS:
- You MUST write code to the file: {write_to_file}
- ALWAYS format your code response EXACTLY like this:
  ```python
  def function_name():
      pass
  ```
- The code block MUST start with ```python, ```javascript, or the appropriate language
- Include ALL necessary code - complete, runnable files
- Do NOT include explanations outside code blocks when writing to files
- The code block is the ONLY thing that will be written to the file
- If creating MULTIPLE files, format each file like this:
  File: filename1.py
  ```python
  ```
  
  File: filename2.py
  ```python
  ```"""
    elif files_in_context:
        files_list = ", ".join(files_in_context)
        system_prompt += f"""

CRITICAL FILE EDITING INSTRUCTIONS:
- The user wants to EDIT the following file(s): {files_list}
- You MUST provide the COMPLETE, UPDATED code for each file
- Format your response EXACTLY like this for EACH file:
  File: {files_in_context[0]}
  ```python
  ```
- If editing multiple files, format each like:
  File: filename1.py
  ```python
  ```
  
  File: filename2.py
  ```python
  ```
- IMPORTANT: Provide the COMPLETE file content, not just the modified parts
- The entire file will be replaced with your code"""
    
    if context:
        system_prompt += f"\n\nCode context:\n{context}"
    
    user_query = query
    if write_to_file:
        user_query += f"\n\nREQUIREMENT: Write the complete code to the file '{write_to_file}' in a properly formatted code block."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    tasks = [
        call_llm(model, messages) for model in COUNCIL_MODELS
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    llm_responses = []
    for model, response in zip(COUNCIL_MODELS, responses):
        if isinstance(response, Exception):
            llm_responses.append(
                LLMResponse(
                    model=model,
                    response=f"Error: {str(response)}",
                )
            )
        else:
            llm_responses.append(
                LLMResponse(model=model, response=response)
            )
    
    return llm_responses


async def get_reviews(
    query: str, responses: List[LLMResponse], context: Optional[str] = None
) -> List[Dict]:
    """Stage 2: Each LLM reviews and ranks other responses."""
    reviews = []
    
    for reviewer_model in COUNCIL_MODELS:
        anonymized_responses = [
            {"id": i, "response": resp.response}
            for i, resp in enumerate(responses)
        ]
        
        review_prompt = f"""You are reviewing responses to the following question:

{query}
"""
        if context:
            review_prompt += f"\n\nCode context:\n{context}"
        
        review_prompt += f"""

Here are the anonymized responses from other assistants:

{json.dumps(anonymized_responses, indent=2)}

Please rank these responses from best (1) to worst ({len(responses)}) based on accuracy, insight, and helpfulness. 
Respond with a JSON object mapping response IDs to their rankings (1 = best).
Example: {{"0": 2, "1": 1, "2": 3, "3": 4}}
"""
        
        messages = [
            {"role": "system", "content": "You are an expert evaluator of AI responses."},
            {"role": "user", "content": review_prompt},
        ]
        
        try:
            review_text = await call_llm(reviewer_model, messages, temperature=0.3)
            review_json = json.loads(review_text.strip().replace("```json", "").replace("```", ""))
            reviews.append({
                "reviewer": reviewer_model,
                "rankings": review_json,
            })
        except Exception as e:
            reviews.append({
                "reviewer": reviewer_model,
                "rankings": {},
                "error": str(e),
            })
    
    return reviews


async def get_final_response(
    query: str, responses: List[LLMResponse], reviews: List[Dict], context: Optional[str] = None, write_to_file: Optional[str] = None
) -> str:
    """Stage 3: Chairman synthesizes final response."""
    rankings_sum = {}
    for review in reviews:
        for resp_id, rank in review.get("rankings", {}).items():
            resp_id = int(resp_id)
            if resp_id not in rankings_sum:
                rankings_sum[resp_id] = []
            rankings_sum[resp_id].append(rank)
    
    avg_rankings = {
        resp_id: sum(ranks) / len(ranks)
        for resp_id, ranks in rankings_sum.items()
    }
    
    sorted_responses = sorted(
        enumerate(responses),
        key=lambda x: avg_rankings.get(x[0], 999)
    )
    
    files_in_context = []
    if context:
        import re
        file_matches = re.findall(r'File:\s*([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))', context, re.IGNORECASE)
        files_in_context = list(set(file_matches))
    
    chairman_prompt = f"""You are the Chairman of an LLM Council. The council has discussed the following question:

{query}
"""
    if context:
        chairman_prompt += f"\n\nCode context:\n{context}"
    
    if write_to_file:
        chairman_prompt += f"""

CRITICAL FILE OUTPUT REQUIREMENT:
- You MUST write code to the file: {write_to_file}
- Your response MUST START with a code block formatted EXACTLY like this:
  ```python
  [complete code here - ALL of it, no truncation]
  ```
- The code block MUST be the PRIMARY and COMPLETE content of your response
- Include ALL necessary imports, functions, classes, methods - a complete, runnable file
- Do NOT truncate the code - provide the ENTIRE file content
- Do NOT include explanations before or after the code block when writing to files
- The code block is the ONLY thing that will be extracted and written to the file
- IMPORTANT: Make sure your response is complete and not cut off mid-code
- If the user requests MULTIPLE files, create them ALL in your response like this:
  File: filename1.py
  ```python
  ```
  
  File: filename2.py
  ```python
  ```"""
    elif files_in_context:
        files_list = ", ".join(files_in_context)
        chairman_prompt += f"""

CRITICAL FILE EDITING REQUIREMENT:
- The user wants to EDIT the following file(s): {files_list}
- You MUST provide the COMPLETE, UPDATED code for EACH file mentioned
- Format your response EXACTLY like this for EACH file:
  File: {files_in_context[0]}
  ```python
  ```
- If editing multiple files, format each like:
  File: filename1.py
  ```python
  ```
  
  File: filename2.py
  ```python
  ```
- IMPORTANT: Provide the COMPLETE file content with changes applied, not just the modified parts
- The entire file(s) will be replaced with your code"""
    
    chairman_prompt += "\n\nHere are the responses from council members (ordered by quality):\n\n"
    for i, (_, resp) in enumerate(sorted_responses, 1):
        chairman_prompt += f"Response {i}:\n{resp.response}\n\n"
    
    if write_to_file:
        chairman_prompt += f"""CRITICAL: Synthesize the best code from all responses.
- If creating ONE file, provide it in a SINGLE code block: ```python\n[complete code]\n```
- If creating MULTIPLE files, format each like: File: filename.py\n```python\n[code]\n```
- ALL files must be complete and runnable
- This code will be written to '{write_to_file}' (or multiple files if specified)."""
    elif files_in_context:
        chairman_prompt += f"""CRITICAL: Synthesize the best code from all responses for editing the file(s).
- Provide COMPLETE, UPDATED code for: {files_list}
- Format each file as: File: filename.py\n```python\n[complete updated code]\n```
- Include ALL code - the entire file will be replaced
- Make sure the code is complete and runnable"""
    else:
        chairman_prompt += """Please synthesize these responses into a single, comprehensive, and accurate final answer. 
Combine the best insights from each response while avoiding redundancy."""
    
    messages = [
        {"role": "system", "content": "You are the Chairman of an LLM Council, responsible for synthesizing the best insights from multiple AI responses."},
        {"role": "user", "content": chairman_prompt},
    ]
    
    final_response = await call_llm(CHAIRMAN_MODEL, messages, temperature=0.5)
    return final_response


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "LLM Council IDE Backend"}


@app.get("/api/test-model/{model_name}")
async def test_model(model_name: str):
    """Test if a specific model works with OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'hello' in one word."},
        ]
        response = await call_llm(model_name, test_messages)
        return {
            "model": model_name,
            "status": "success",
            "response": response,
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "error",
            "error": str(e),
        }


@app.post("/api/chat", response_model=CouncilResponse)
async def chat(request: ChatRequest):
    """The main chat endpoint - this is where the magic happens. Multiple LLMs discuss, then the chairman makes the final call."""
    # This is the heart of the system - all the LLMs work together here
    try:
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
        
        conversation_history = []
        if request.conversation_id:
            try:
                safe_conversation_id = request.conversation_id.replace(":", "-").replace("/", "-")
                conversation_file = f"data/conversations/{safe_conversation_id}.json"
                if os.path.exists(conversation_file):
                    with open(conversation_file, "r") as f:
                        prev_conversation = json.load(f)
                        conversation_history.append({
                            "role": "user",
                            "content": prev_conversation.get("message", "")
                        })
                        conversation_history.append({
                            "role": "assistant",
                            "content": prev_conversation.get("final_response", "")
                        })
            except Exception as e:
                pass
                        
        enhanced_context = request.context or ""
        if conversation_history:
            history_text = "\n\n=== Previous Conversation History ===\n"
            for msg in conversation_history:
                role_label = "USER" if msg['role'] == 'user' else "ASSISTANT"
                content = msg['content']
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                history_text += f"{role_label}: {content}\n\n"
            history_text += "=== End of Previous Conversation ===\n"
            enhanced_context = f"{enhanced_context}\n\n{history_text}" if enhanced_context else history_text
        
        individual_responses = await get_initial_responses(
            request.message, 
            enhanced_context, 
            request.write_to_file
        )
        
        reviews = await get_reviews(request.message, individual_responses, enhanced_context)
        
        final_response = await get_final_response(
            request.message, individual_responses, reviews, enhanced_context, request.write_to_file
        )
        
        files_written = []
        
        import re
        multi_file_patterns = [
            r'(?:File|file|Create|create):\s*([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))\s*\n```(?:python|javascript|typescript|js|py|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)```',
            r'^([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))\s*\n```(?:python|javascript|typescript|js|py|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)```',
        ]
        
        extracted_files = {}
        for i, pattern in enumerate(multi_file_patterns, 1):
            matches = re.finditer(pattern, final_response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            for match in matches:
                filename = match.group(1).strip()
                code_content = match.group(2).strip()
                if not code_content or len(code_content) < 10:
                    truncated_match = re.search(pattern.replace(r'(.*?)```', r'(.*)'), final_response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                    if truncated_match:
                        code_content = truncated_match.group(2).strip()
                
                if filename and code_content and len(code_content) > 10:
                    if filename not in extracted_files or len(code_content) > len(extracted_files[filename]):
                        extracted_files[filename] = code_content
                                
                
        files_in_context = []
        if request.context:
            file_matches = re.findall(r'File:\s*([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))', request.context, re.IGNORECASE)
            files_in_context = list(set(file_matches))
                                
        should_use_multi_file = len(extracted_files) > 0
        
        if should_use_multi_file:
            file_pattern_count = len(re.findall(r'(?:File|file):\s*[a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh)', final_response, re.IGNORECASE))
            if file_pattern_count > 1 or len(extracted_files) > 1:
                for filename, code_content in extracted_files.items():
                    try:
                        safe_filename = filename.strip()
                        safe_filename = re.sub(r'[<>:"|?*\x00-\x1f\\]', '', safe_filename)
                        safe_filename = safe_filename.replace(' ', '_').replace(',', '_').replace(':', '_')
                        safe_filename = safe_filename.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                        safe_filename = re.sub(r'_+', '_', safe_filename)
                        
                        file_path_obj = Path("workspace") / safe_filename
                        if str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
                            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            with open(file_path_obj, "w", encoding="utf-8") as f:
                                f.write(code_content)
                            if file_path_obj.exists():
                                files_written.append(safe_filename)
                    except Exception as e:
                        pass
            elif len(extracted_files) == 1:
                filename, code_content = list(extracted_files.items())[0]
                should_write = (
                    'File:' in final_response or 
                    'file:' in final_response.lower() or
                    filename in files_in_context
                )
                
                if should_write:
                    try:
                        safe_filename = filename.strip()
                        safe_filename = re.sub(r'[<>:"|?*\x00-\x1f\\]', '', safe_filename)
                        safe_filename = safe_filename.replace(' ', '_').replace(',', '_').replace(':', '_')
                        safe_filename = safe_filename.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                        safe_filename = re.sub(r'_+', '_', safe_filename)
                        
                        file_path_obj = Path("workspace") / safe_filename
                        if str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
                            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            file_exists = file_path_obj.exists()
                            with open(file_path_obj, "w", encoding="utf-8") as f:
                                f.write(code_content)
                            if file_path_obj.exists():
                                files_written.append(safe_filename)
                                action = "Edited" if file_exists else "Created"
                    except Exception as e:
                        pass
                                
        if files_in_context and not files_written:
                                                            
            code_blocks = re.findall(r'```(?:python|javascript|typescript|js|py|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)```', final_response, re.DOTALL | re.IGNORECASE)
                        
            if not code_blocks:
                truncated_blocks = re.findall(r'```(?:python|javascript|typescript|js|py|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)(?:\n\n|$)', final_response, re.DOTALL | re.IGNORECASE)
                code_blocks = truncated_blocks
                            
            if not code_blocks:
                marker_match = re.search(r'```(?:python|javascript|typescript|js|py|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*)', final_response, re.DOTALL | re.IGNORECASE)
                if marker_match:
                    code_blocks = [marker_match.group(1)]
                                
            if not code_blocks:
                lines = final_response.split('\n')
                code_lines = []
                in_code_section = False
                for i, line in enumerate(lines):
                    if '```' in line:
                        in_code_section = True
                        continue
                    if in_code_section and '```' in line:
                        in_code_section = False
                        continue
                    if in_code_section or (line.strip() and (
                        any(kw in line for kw in ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ', 'print(', 'return ', 'if ', 'for ', 'while ', '=']) or
                        (line.startswith(' ') or line.startswith('\t'))
                    )):
                        code_lines.append(line)
                
                if code_lines and len(code_lines) > 3:
                    code_blocks = ['\n'.join(code_lines)]
                                
            if code_blocks:
                for filename in files_in_context:
                    code_content = max(code_blocks, key=len).strip()
                    
                    if len(code_content) > 20:
                        looks_like_code = (
                            any(kw in code_content for kw in ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ', 'print(', 'return ', 'if ', 'for ', 'while ', '=', '(', ')', '{', '}']) or
                            code_content.count('\n') > 2
                        )
                        
                        if looks_like_code:
                            try:
                                safe_filename = filename.strip()
                                safe_filename = re.sub(r'[<>:"|?*\x00-\x1f\\]', '', safe_filename)
                                file_path_obj = Path("workspace") / safe_filename
                                if str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
                                    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                                    file_exists = file_path_obj.exists()
                                    
                                    with open(file_path_obj, "w", encoding="utf-8") as f:
                                        f.write(code_content)
                                    
                                    import time
                                    time.sleep(0.1)
                                    
                                    if file_path_obj.exists() and file_path_obj.stat().st_size > 0:
                                        files_written.append(safe_filename)
                                        action = "Edited" if file_exists else "Created"
                                        break
                            except Exception as e:
                                import traceback
                                pass
                                        
        file_written = None
        if not files_written and request.write_to_file:
                                                                        
            try:
                import re
                code_content = None
                
                code_blocks_with_lang = []
                
                pattern_complete = r'```(?:python|javascript|typescript|js|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)```'
                matches = re.finditer(pattern_complete, final_response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    code_block = match.group(1).strip()
                    if code_block and len(code_block) > 10:
                        code_blocks_with_lang.append(code_block)
                
                if not code_blocks_with_lang:
                    pattern_truncated = r'```(?:python|javascript|typescript|js|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*?)(?:\n\n|\Z)'
                    matches = re.finditer(pattern_truncated, final_response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        code_block = match.group(1).strip()
                        if code_block and len(code_block) > 10:
                            code_blocks_with_lang.append(code_block)
                
                if not code_blocks_with_lang and final_response.strip().startswith('```'):
                    first_block_match = re.search(r'```(?:python|javascript|typescript|js|jsx|ts|tsx|html|css|json|bash|sh|java|cpp|c|go|rust|ruby|php|text|plain)?\s*\n(.*)', final_response, re.DOTALL | re.IGNORECASE)
                    if first_block_match:
                        code_block = first_block_match.group(1).strip()
                        if code_block and len(code_block) > 10:
                            code_blocks_with_lang.append(code_block)
                                            
                                
                if not code_blocks_with_lang:
                    matches = re.finditer(r'```\s*\n(.*?)```', final_response, re.DOTALL | re.MULTILINE)
                    for match in matches:
                        code_block = match.group(1).strip()
                        if code_block and len(code_block) > 10:
                            code_blocks_with_lang.append(code_block)
                    
                    if not code_blocks_with_lang:
                        match = re.search(r'```\s*\n(.*)', final_response, re.DOTALL)
                        if match:
                            code_block = match.group(1).strip()
                            if code_block and len(code_block) > 10:
                                code_blocks_with_lang.append(code_block)
                                                    
                                    
                if code_blocks_with_lang:
                    code_content = max(code_blocks_with_lang, key=len).strip()
                    code_content = code_content.strip()
                    
                    if len(code_content) < 20:
                        markers = [
                            r'(?:Here is|Here\'s|Code:|```python|```javascript|```typescript|```js|```py)\s*\n(.*?)(?:\n\n|```|\Z)',
                        ]
                        for marker in markers:
                            match = re.search(marker, final_response, re.DOTALL | re.IGNORECASE)
                            if match:
                                potential_code = match.group(1).strip()
                                if any(keyword in potential_code for keyword in ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ']):
                                    code_content = potential_code
                                    break
                    
                    if not code_content:
                        lines = final_response.split('\n')
                        code_lines = []
                        in_code_block = False
                        code_started = False
                        
                        for line in lines:
                            if '```' in line:
                                in_code_block = not in_code_block
                                if in_code_block:
                                    code_started = True
                                continue
                            
                            if in_code_block or code_started:
                                if line.strip():
                                    code_lines.append(line)
                                elif any(keyword in line for keyword in ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ', 'return ', 'if ', 'for ', 'while ', 'print(', 'console.']):
                                    code_lines.append(line)
                                    code_started = True
                                elif code_started and (line.strip() or line.startswith(' ') or line.startswith('\t')):
                                    code_lines.append(line)
                                elif code_started and line.strip():
                                    code_lines.append(line)
                                elif code_started and not line.strip():
                                    code_lines.append(line)
                        
                        if code_lines and len(code_lines) > 3:
                            code_content = '\n'.join(code_lines).strip()
                                                
                    if not code_content or len(code_content.strip()) < 20:
                        code_content = final_response.strip()
                                        
                safe_filename = request.write_to_file.strip()
                                
                if len(safe_filename) > 100:
                    if '.' in safe_filename:
                        safe_filename = safe_filename.split('.')[-2] + '.' + safe_filename.split('.')[-1]
                    else:
                        words = re.findall(r'[a-zA-Z0-9_]+', safe_filename)
                        safe_filename = words[-1] if words else "file"
                
                safe_filename = re.sub(r'[<>:"|?*\x00-\x1f\\]', '', safe_filename)
                safe_filename = safe_filename.replace(' ', '_').replace(',', '_').replace(':', '_').replace(';', '_')
                safe_filename = safe_filename.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                safe_filename = safe_filename.replace('{', '').replace('}', '').replace('&', 'and')
                safe_filename = re.sub(r'_+', '_', safe_filename)
                safe_filename = re.sub(r'\.+', '.', safe_filename)
                safe_filename = safe_filename.strip('._')
                
                if len(safe_filename) > 50 or safe_filename.count('_') > 10:
                    safe_filename = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                base_name = safe_filename.split('/')[-1]
                if '.' not in base_name:
                    if any(kw in code_content for kw in ['def ', 'import ', 'class ', 'print(']):
                        safe_filename += '.py'
                    elif any(kw in code_content for kw in ['function ', 'const ', 'let ', 'var ', 'console.']):
                        safe_filename += '.js'
                    else:
                        safe_filename += '.py'
                                        
                if not safe_filename or len(safe_filename) < 3:
                    safe_filename = f"generated_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                                    
                                
                file_path_obj = Path("workspace") / safe_filename
                abs_path = file_path_obj.resolve()
                workspace_abs = Path("workspace").resolve()
                
                                                                
                if not str(abs_path).startswith(str(workspace_abs)):
                    raise ValueError(f"Security violation: path outside workspace")
                
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path_obj, "w", encoding="utf-8") as f:
                    f.write(code_content)
                
                import time
                time.sleep(0.1)
                
                if not file_path_obj.exists():
                    raise FileNotFoundError(f"File was not created: {abs_path}")
                
                file_size = file_path_obj.stat().st_size
                if file_size == 0:
                    raise ValueError(f"File was created but is empty: {abs_path}")
                
                with open(file_path_obj, "r", encoding="utf-8") as f:
                    verify_content = f.read()
                if verify_content != code_content:
                    pass
                
                file_written = safe_filename
                    
            except Exception as e:
                import traceback
                
                try:
                    safe_filename = request.write_to_file.strip()
                    safe_filename = re.sub(r'[<>:"|?*\x00-\x1f\\]', '', safe_filename)
                    safe_filename = safe_filename.replace(' ', '_').replace(',', '_').replace(':', '_')
                    if '.' not in safe_filename.split('/')[-1]:
                        safe_filename += '.py'
                    
                    file_path_obj = Path("workspace") / safe_filename
                    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    
                    fallback_content = f"""# File created automatically
{final_response}

"""
                    with open(file_path_obj, "w", encoding="utf-8") as f:
                        f.write(fallback_content)
                    
                    if file_path_obj.exists() and file_path_obj.stat().st_size > 0:
                        file_written = safe_filename
                except Exception as e2:
                    pass
        
        conversation_id = request.conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        conversation_data = {
            "id": conversation_id,
            "message": request.message,
            "context": request.context,
            "individual_responses": [r.dict() for r in individual_responses],
            "reviews": reviews,
            "final_response": final_response,
            "file_written": file_written,
            "timestamp": datetime.now().isoformat(),
        }
        
        safe_conversation_id = conversation_id.replace(":", "-").replace("/", "-")
        os.makedirs("data/conversations", exist_ok=True)
        with open(f"data/conversations/{safe_conversation_id}.json", "w") as f:
            json.dump(conversation_data, f, indent=2)
        
        response_data = {
            "individual_responses": individual_responses,
            "reviews": reviews,
            "final_response": final_response,
            "conversation_id": conversation_id,
        }
        if files_written:
            response_data["files_written"] = files_written
            response_data["file_written"] = files_written[0] if files_written else None
        elif file_written:
            response_data["file_written"] = file_written
            response_data["files_written"] = [file_written]
        
        return CouncilResponse(**response_data)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/completion")
async def code_completion(request: CodeCompletionRequest):
    """Code completion endpoint using LLM Council."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    prompt = f"""Complete the following code at the cursor position (indicated by |):

{request.code[:request.cursor_position]}|{request.code[request.cursor_position:]}

Provide only the completion code, not the entire file."""
    
    if request.language:
        prompt = f"Language: {request.language}\n\n{prompt}"
    
    individual_responses = await get_initial_responses(prompt, request.code)
    
    completion = individual_responses[0].response if individual_responses else ""
    
    return {
        "completion": completion,
        "alternatives": [r.response for r in individual_responses[1:]] if len(individual_responses) > 1 else [],
    }


@app.get("/api/conversations")
async def list_conversations():
    """List all conversations."""
    conversations = []
    if os.path.exists("data/conversations"):
        for filename in os.listdir("data/conversations"):
            if filename.endswith(".json"):
                with open(f"data/conversations/{filename}", "r") as f:
                    conversations.append(json.load(f))
    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    filepath = f"data/conversations/{conversation_id}.json"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    with open(filepath, "r") as f:
        return json.load(f)


@app.get("/api/files")
async def list_files(path: str = ""):
    """List files and directories in workspace."""
    workspace_path = Path("workspace") / path
    if not workspace_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not str(workspace_path.resolve()).startswith(str(Path("workspace").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    items = []
    try:
        for item in workspace_path.iterdir():
            items.append({
                "name": item.name,
                "path": str(item.relative_to("workspace")),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"files": sorted(items, key=lambda x: (x["type"] == "file", x["name"]))}


@app.get("/api/files/{file_path:path}")
async def read_file(file_path: str):
    """Read a file from workspace."""
    file_path_obj = Path("workspace") / file_path
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if file_path_obj.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    
    try:
        with open(file_path_obj, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "path": file_path,
            "content": content,
            "size": file_path_obj.stat().st_size,
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not text-based")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WriteFileRequest(BaseModel):
    """Request to write a file."""
    content: str


@app.post("/api/files/{file_path:path}")
async def write_file(file_path: str, request: WriteFileRequest):
    """Write content to a file in workspace."""
    file_path_obj = Path("workspace") / file_path
    
    if not str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path_obj, "w", encoding="utf-8") as f:
            f.write(request.content)
        
        return {
            "path": file_path,
            "message": "File written successfully",
            "size": file_path_obj.stat().st_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/files/{file_path:path}")
async def delete_file(file_path: str):
    """Delete a file or directory from workspace."""
    file_path_obj = Path("workspace") / file_path
    
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        if file_path_obj.is_dir():
            import shutil
            shutil.rmtree(file_path_obj)
        else:
            file_path_obj.unlink()
        
        return {"message": "Deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/files/directory/{dir_path:path}")
async def create_directory(dir_path: str):
    """Create a directory in workspace."""
    dir_path_obj = Path("workspace") / dir_path
    
    if not str(dir_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        dir_path_obj.mkdir(parents=True, exist_ok=True)
        return {"message": "Directory created successfully", "path": dir_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExecuteRequest(BaseModel):
    """Request to execute code."""
    command: Optional[str] = None
    language: Optional[str] = None
    file_path: Optional[str] = None
    working_dir: Optional[str] = None
    input_data: Optional[str] = None


@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    """Execute code or commands."""
    # Make sure we're always working in the safe workspace directory
    working_dir = Path("workspace")
    if request.working_dir:
        working_dir = working_dir / request.working_dir
        if not str(working_dir.resolve()).startswith(str(Path("workspace").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    
    working_dir.mkdir(parents=True, exist_ok=True)
    
    normalized_file_path = None
    if request.file_path:
        normalized_file_path = request.file_path.strip()
        if normalized_file_path.lower().startswith("workspace/"):
            normalized_file_path = normalized_file_path[10:]
        if normalized_file_path.startswith("/workspace/"):
            normalized_file_path = normalized_file_path[11:]
        
        file_path_obj = Path("workspace") / normalized_file_path
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        if not str(file_path_obj.resolve()).startswith(str(Path("workspace").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not request.language:
            ext = normalized_file_path.split('.')[-1].lower()
            lang_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'jsx': 'javascript',
                'tsx': 'typescript',
                'sh': 'bash',
            }
            request.language = lang_map.get(ext, 'bash')
    
    if request.language == "python":
        if normalized_file_path:
            cmd = ["python3", normalized_file_path]
        elif request.command:
            cmd = ["python3", "-c", request.command]
        else:
            raise HTTPException(status_code=400, detail="Either file_path or command must be provided")
    elif request.language == "javascript" or request.language == "typescript":
        if normalized_file_path:
            cmd = ["node", normalized_file_path]
        elif request.command:
            cmd = ["node", "-e", request.command]
        else:
            raise HTTPException(status_code=400, detail="Either file_path or command must be provided")
    elif request.language == "bash" or request.language == "shell":
        if not request.command:
            raise HTTPException(status_code=400, detail="Command is required for bash/shell")
        cmd = ["bash", "-c", request.command]
    else:
        if not request.command:
            raise HTTPException(status_code=400, detail="Command is required")
        cmd = shlex.split(request.command)
    
    try:
        # We use Popen instead of run() so we can send input interactively
        process = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        
        try:
            if request.input_data:
                input_to_send = request.input_data
                lines = input_to_send.split('\n')
                input_to_send = '\n'.join(lines)
                if input_to_send and not input_to_send.endswith('\n'):
                    input_to_send += '\n'
                
                stdout, stderr = process.communicate(input=input_to_send, timeout=30)
            else:
                stdout, stderr = process.communicate(timeout=30)
            
            returncode = process.returncode
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "success": returncode == 0,
            }
        except subprocess.TimeoutExpired:
            process.kill()
            raise HTTPException(status_code=408, detail="Execution timeout")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Execution timeout")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Command not found: {cmd[0]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Alternative approach: could use gunicorn for production
# # gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# # Could add rate limiting here if needed
# # from slowapi import Limiter, _rate_limit_exceeded_handler
# # limiter = Limiter(key_func=get_remote_address)
# # app.state.limiter = limiter

# # Future enhancement: cache LLM responses for identical queries
# # cache = {}
# # cache_key = hashlib.md5(query.encode()).hexdigest()
# # if cache_key in cache:
# #     return cache[cache_key]

# # Could add request logging middleware
# # @app.middleware("http")
# # async def log_requests(request: Request, call_next):
# #     start_time = time.time()
# #     response = await call_next(request)
# #     process_time = time.time() - start_time
# #     logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
# #     return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

