#!/bin/bash

# Start script for Better Cursor IDE

echo "Starting Better Cursor IDE..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from env.example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "Please edit .env and add your OPENROUTER_API_KEY"
        echo "Press Enter to continue anyway, or Ctrl+C to exit..."
        read
    else
        echo "Error: env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY not set in .env file"
fi

# Start backend in background
echo "Starting backend..."
python -m backend.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 2

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "Better Cursor IDE is starting!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=========================================="

# Wait for user interrupt
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for processes
wait

