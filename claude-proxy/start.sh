#!/bin/bash

# Claude Proxy Server Startup Script
# One-click startup with auto-reload on file changes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Claude Proxy Server ==="

# Kill existing server processes
echo "Stopping existing server processes..."
pkill -f "uvicorn.*server:app.*8000" 2>/dev/null
pkill -f "python.*server\.py" 2>/dev/null
sleep 1

# Create logs directory if not exists
mkdir -p logs

# Start server with auto-reload
echo "Starting server on http://0.0.0.0:8000"
echo "Test page: http://localhost:8000/test"
echo "Auto-reload enabled - changes will take effect automatically"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
