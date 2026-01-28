FROM python:3.11-slim

# Install Node.js, Git, and Ripgrep (required for Claude Code)
RUN apt-get update && apt-get install -y curl git ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install Claude Code CLI and FastAPI dependencies
RUN npm install -g @anthropic-ai/claude-code
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

# Create agent directories structure in /data (Railway volume)
RUN mkdir -p /data/agents /data/logs /data/sessions /data/files

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

