# Contributing to RA.Aid

Welcome to the RA.Aid community! We're thrilled you're interested in contributing. This project thrives thanks to contributors like you, and we're excited to have you join us on this journey.

## Ways to Contribute

There are many valuable ways to contribute to RA.Aid:

### 1. Join Our Community
- Join our Discord community to connect with other users and contributors
- Help answer questions from other users
- Share your experiences and use cases
- Provide feedback and suggestions

### 2. Report Issues
- Found a bug? Open an issue on our [GitHub repository](https://github.com/ai-christianson/RA.Aid/issues)
- Before creating a new issue, please check if it already exists
- Include as much detail as possible:
  - Steps to reproduce
  - Expected vs actual behavior
  - Your environment (OS, Python version, etc.)
  - Any relevant error messages

### 3. Contribute to Documentation
- Our documentation lives in the `docs/` folder
- Found a typo? Have an idea for better explanations? Open a PR!
- You can use RA.Aid itself to help draft documentation changes
- Even small improvements are welcome

### 4. Code Contributions
- Look for issues labeled "help wanted" or "good first issue" on our GitHub
- Feel free to pick up any open issue - don't be shy!
- **You can even use RA.Aid to help understand the codebase and make changes**
- Before starting work on larger changes, please open an issue to discuss

## Making Your First Contribution

1. Fork the repository
2. Create a branch for your changes
3. Make your changes
4. Write/update tests if needed
5. Submit a Pull Request
6. Wait for review and address any feedback

Don't hesitate to ask questions if you're unsure about anything. Remember: every expert was once a beginner!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/ai-christianson/RA.Aid.git
cd RA.Aid
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install in dev mode:

```bash
pip install -e .
```

4. Run RA.Aid:

```bash
ra-aid -m "Your task or query here"
```

5. Frontend miscellany:

The backend must always be run using `ra-aid --server` separately. `yarn dev` is used to run the frontend development server.

```bash
# run development server on port 5173
cd frontend/
yarn dev

# run development bundle for other ports
# (the prebuilt bundle from uvicorn always serves both frontend and backend on --server-port argument)
VITE_FRONTEND_PORT=5555 yarn dev  # hosts web app on 5555 targeting backend on 1818
VITE_FRONTEND_PORT=2221 VITE_BACKEND_PORT=9191 yarn dev  # hosts web app on 2221 targeting backend on 9191
VITE_BACKEND_PORT=4002 yarn dev  # hosts web app on 5173 (vite default port) targeting backend on 4002
yarn dev  # hosts web app on 5173 (default) targeting backend on 1818
```

## This is Your Project Too

RA.Aid is a community project that grows stronger with each contribution. Whether it's fixing a typo in documentation, reporting a bug, or adding a new feature - every contribution matters and is valued.

Don't feel like you need to make massive changes to contribute. Small, focused contributions are often the best way to start. Use what you know, and learn as you go!
