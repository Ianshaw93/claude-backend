from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

app = FastAPI()

# Persistent data directory (mounted Railway volume)
DATA_DIR = Path("/data")
SESSIONS_DIR = DATA_DIR / "sessions"
FILES_DIR = DATA_DIR / "files"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)

@app.post("/claude")
async def run_claude(command: str, session_id: Optional[str] = None):
    # This executes 'claude' commands on your Railway server
    try:
        result = subprocess.run(
            ["claude", "-p", command], 
            capture_output=True, text=True, check=True
        )
        
        # Save session data if session_id is provided
        if session_id:
            session_file = SESSIONS_DIR / f"{session_id}.json"
            session_data = {
                "command": command,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat()
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
        
        return {"output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

