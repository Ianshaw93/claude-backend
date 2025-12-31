from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post("/claude")
async def run_claude(command: str):
    # This executes 'claude' commands on your Railway server
    try:
        result = subprocess.run(
            ["claude", "-p", command], 
            capture_output=True, text=True, check=True
        )
        return {"output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

