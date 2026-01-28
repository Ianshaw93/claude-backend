from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import hmac
import hashlib
import time
import httpx

app = FastAPI()

# Slack configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

# Track processed events to avoid duplicates
PROCESSED_EVENTS: set = set()

# Persistent data directory (mounted Railway volume)
DATA_DIR = Path("/data")
SESSIONS_DIR = DATA_DIR / "sessions"
FILES_DIR = DATA_DIR / "files"
LOGS_DIR = DATA_DIR / "logs"
AGENTS_DIR = DATA_DIR / "agents"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
AGENTS_DIR.mkdir(exist_ok=True)

# Global task status tracking
TASKS_STATUS: Dict[str, Dict] = {}

# Request models
class Task(BaseModel):
    command: str
    agent_id: Optional[str] = None

class RunAgentsRequest(BaseModel):
    tasks: List[Task]

async def run_agent(task: Task, task_id: str) -> Dict:
    """Run a single agent task in isolation"""
    # Generate unique agent ID if not provided
    agent_id = task.agent_id or f"agent_{task_id}"
    
    # Create isolated config directory for this agent
    agent_config_dir = AGENTS_DIR / agent_id
    agent_config_dir.mkdir(exist_ok=True)
    
    # Set unique CLAUDE_CONFIG_DIR environment variable
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(agent_config_dir)
    
    # Create unique log file for this agent
    log_file = LOGS_DIR / f"{agent_id}_{task_id}.log"
    
    # Update status to running
    TASKS_STATUS[task_id] = {
        "status": "running",
        "agent_id": agent_id,
        "command": task.command,
        "started_at": datetime.now().isoformat(),
        "log_file": str(log_file)
    }
    
    try:
        # Build the command with proper escaping
        # Using shell=True for Railway server safety
        cmd = f'claude -p "{task.command}"'
        
        # Create subprocess with shell=True and isolated environment
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            shell=True
        )
        
        # Capture output
        stdout, stderr = await process.communicate()
        
        # Write to log file
        with open(log_file, "w") as f:
            f.write(f"=== Agent: {agent_id} ===\n")
            f.write(f"=== Task ID: {task_id} ===\n")
            f.write(f"=== Command: {task.command} ===\n")
            f.write(f"=== Started: {TASKS_STATUS[task_id]['started_at']} ===\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout.decode('utf-8', errors='replace'))
            f.write("\n\n=== STDERR ===\n")
            f.write(stderr.decode('utf-8', errors='replace'))
            f.write(f"\n\n=== Exit Code: {process.returncode} ===\n")
            f.write(f"=== Completed: {datetime.now().isoformat()} ===\n")
        
        # Update status
        if process.returncode == 0:
            TASKS_STATUS[task_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "exit_code": process.returncode,
                "output": stdout.decode('utf-8', errors='replace')
            })
        else:
            TASKS_STATUS[task_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "exit_code": process.returncode,
                "error": stderr.decode('utf-8', errors='replace'),
                "output": stdout.decode('utf-8', errors='replace')
            })
        
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": TASKS_STATUS[task_id]["status"],
            "exit_code": process.returncode,
            "output": stdout.decode('utf-8', errors='replace'),
            "error": stderr.decode('utf-8', errors='replace') if process.returncode != 0 else None
        }
        
    except Exception as e:
        # Log error
        with open(log_file, "w") as f:
            f.write(f"=== Agent: {agent_id} ===\n")
            f.write(f"=== Task ID: {task_id} ===\n")
            f.write(f"=== Command: {task.command} ===\n")
            f.write(f"=== Error: {str(e)} ===\n")
            f.write(f"=== Failed: {datetime.now().isoformat()} ===\n")
        
        TASKS_STATUS[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })
        
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": "failed",
            "error": str(e)
        }

@app.post("/run-agents")
async def run_agents(request: RunAgentsRequest):
    """
    Run multiple agents in parallel.
    Each agent gets isolated config directory and unique log file.
    """
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")
    
    # Generate unique task IDs for each task
    task_ids = [str(uuid.uuid4()) for _ in request.tasks]
    
    # Create coroutines for all agents
    coroutines = [
        run_agent(task, task_id) 
        for task, task_id in zip(request.tasks, task_ids)
    ]
    
    # Execute all agents in parallel
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Handle any exceptions that occurred during gathering
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            task_id = task_ids[i]
            TASKS_STATUS[task_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(result)
            })
            final_results.append({
                "task_id": task_id,
                "status": "failed",
                "error": str(result)
            })
        else:
            final_results.append(result)
    
    return {
        "total_tasks": len(request.tasks),
        "results": final_results
    }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a specific task"""
    if task_id not in TASKS_STATUS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS_STATUS[task_id]

@app.get("/status")
async def get_all_status():
    """Get status of all tasks"""
    return {
        "total_tasks": len(TASKS_STATUS),
        "tasks": TASKS_STATUS
    }

@app.get("/logs/{agent_id}/{task_id}")
async def get_log(agent_id: str, task_id: str):
    """Retrieve log file for a specific agent task"""
    log_file = LOGS_DIR / f"{agent_id}_{task_id}.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return FileResponse(log_file)

@app.get("/logs")
async def list_logs():
    """List all log files"""
    logs = [f.name for f in LOGS_DIR.iterdir() if f.is_file()]
    return {"logs": logs}

# Legacy endpoint for backward compatibility
@app.post("/claude")
async def run_claude(command: str, session_id: Optional[str] = None):
    """Legacy single-agent endpoint (runs synchronously)"""
    task_id = str(uuid.uuid4())
    task = Task(command=command, agent_id=session_id)
    result = await run_agent(task, task_id)
    
    # Save session data if session_id is provided
    if session_id:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        session_data = {
            "command": command,
            "output": result.get("output", ""),
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id
        }
        
        # Load existing session or create new one
        if session_file.exists():
            with open(session_file, "r") as f:
                sessions = json.load(f)
        else:
            sessions = []
        
        sessions.append(session_data)
        
        with open(session_file, "w") as f:
            json.dump(sessions, f, indent=2)
    
    return {"output": result.get("output", ""), "task_id": task_id}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and save a file to the persistent /data/files directory"""
    try:
        file_path = FILES_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"message": f"File {file.filename} saved successfully", "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{filename}")
async def get_file(filename: str):
    """Retrieve a file from the persistent /data/files directory"""
    file_path = FILES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve a session's history"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_file, "r") as f:
        return json.load(f)

@app.get("/files")
async def list_files():
    """List all files in the persistent storage"""
    files = [f.name for f in FILES_DIR.iterdir() if f.is_file()]
    return {"files": files}


# ==================== SLACK INTEGRATION ====================

def verify_slack_signature(timestamp: str, body: str, signature: str) -> bool:
    """Verify that the request is from Slack using the signing secret"""
    if not SLACK_SIGNING_SECRET:
        return False

    # Check timestamp to prevent replay attacks (5 min window)
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    sig_basestring = f"v0:{timestamp}:{body}".encode()
    computed = hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(f"v0={computed}", signature)


async def send_slack_message(channel: str, text: str, thread_ts: Optional[str] = None):
    """Send a message to a Slack channel"""
    if not SLACK_BOT_TOKEN:
        print("SLACK_BOT_TOKEN not configured")
        return None

    async with httpx.AsyncClient() as client:
        payload = {
            "channel": channel,
            "text": text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json=payload
        )
        return response.json()


async def run_agent_and_notify(command: str, channel: str, thread_ts: Optional[str] = None):
    """Run an agent task and send the result back to Slack"""
    task_id = str(uuid.uuid4())
    task = Task(command=command)

    # Send initial acknowledgment
    await send_slack_message(
        channel,
        f"🤖 Running agent...\n`{command[:100]}{'...' if len(command) > 100 else ''}`\nTask ID: `{task_id}`",
        thread_ts
    )

    # Run the agent
    result = await run_agent(task, task_id)

    # Format the response
    status_emoji = "✅" if result.get("status") == "completed" else "❌"
    output = result.get("output", "")
    error = result.get("error", "")

    # Truncate long outputs for Slack (max 3000 chars)
    if len(output) > 3000:
        output = output[:2900] + "\n\n... (truncated, see full log)"

    if result.get("status") == "completed":
        message = f"{status_emoji} *Agent completed*\n\n```{output}```"
    else:
        message = f"{status_emoji} *Agent failed*\n\nError: ```{error}```\n\nOutput: ```{output}```"

    await send_slack_message(channel, message, thread_ts)


def extract_command_from_text(text: str, bot_user_id: Optional[str] = None) -> str:
    """Extract the command from a Slack message, removing the bot mention"""
    # Remove bot mention if present
    if bot_user_id:
        text = text.replace(f"<@{bot_user_id}>", "").strip()

    # Also handle common mention patterns
    import re
    text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

    return text


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Slack Events API requests.
    Responds to URL verification challenges and processes app_mention/message events.
    """
    # Get raw body for signature verification
    body = await request.body()
    body_str = body.decode('utf-8')

    # Verify signature (skip for URL verification)
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # Parse the JSON body
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Handle URL verification challenge (no signature check needed for this)
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}

    # Verify signature for all other requests
    if SLACK_SIGNING_SECRET and not verify_slack_signature(timestamp, body_str, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Handle event callbacks
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        event_id = data.get("event_id")

        # Deduplicate events (Slack may retry)
        if event_id in PROCESSED_EVENTS:
            return {"ok": True}
        PROCESSED_EVENTS.add(event_id)

        # Clean up old event IDs (keep last 1000)
        if len(PROCESSED_EVENTS) > 1000:
            PROCESSED_EVENTS.clear()

        event_type = event.get("type")

        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return {"ok": True}

        # Handle app_mention events (@bot-name message)
        if event_type == "app_mention":
            text = event.get("text", "")
            channel = event.get("channel")
            thread_ts = event.get("thread_ts") or event.get("ts")

            command = extract_command_from_text(text)

            if command:
                # Run agent in background to respond within 3 seconds
                background_tasks.add_task(
                    run_agent_and_notify,
                    command,
                    channel,
                    thread_ts
                )
            else:
                background_tasks.add_task(
                    send_slack_message,
                    channel,
                    "👋 Hi! Mention me with a task and I'll run a Claude agent for you.\n\nExample: `@Claude Agents analyze main.py`",
                    thread_ts
                )

        # Handle direct messages
        elif event_type == "message" and event.get("channel_type") == "im":
            text = event.get("text", "")
            channel = event.get("channel")
            thread_ts = event.get("ts")

            command = extract_command_from_text(text)

            if command:
                background_tasks.add_task(
                    run_agent_and_notify,
                    command,
                    channel,
                    thread_ts
                )

    # Must respond within 3 seconds
    return {"ok": True}


@app.get("/slack/health")
async def slack_health():
    """Health check for Slack integration"""
    return {
        "slack_bot_token_configured": bool(SLACK_BOT_TOKEN),
        "slack_signing_secret_configured": bool(SLACK_SIGNING_SECRET),
        "status": "ready" if SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET else "missing_config"
    }
