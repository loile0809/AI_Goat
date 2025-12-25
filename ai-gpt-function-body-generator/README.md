# AI GPT – Generate Function Body

Generate Python function bodies using CodeT5 via a local FastAPI backend.

## Features
- Generate function body from signature + docstring
- Fast / Quality modes
- Shows score, warnings, timing

## Usage
1. Start the FastAPI server at http://127.0.0.1:8000
2. Open a Python file
3. Place cursor inside a function
4. Run: Generate Function Body

## Configuration
- aiGpt.apiBaseUrl
- aiGpt.mode
- aiGpt.requestTimeoutMs
