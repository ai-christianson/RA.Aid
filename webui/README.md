# RA.Aid Web UI

A Streamlit-based web interface for RA.Aid.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Structure

- `app.py`: Main Streamlit application entry point
- `pages/`: Individual pages for the Streamlit app
- `components/`: Reusable UI components
- `assets/`: Static assets like images and styles 