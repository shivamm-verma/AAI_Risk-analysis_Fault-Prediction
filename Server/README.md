# AAI Risk Analysis API

## Run locally
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## Notes
- Place models in Model/ as .pkl files (e.g., pickle_exmaple.pkl)
- Endpoints:
  - GET /health
  - GET /models
  - POST /predict/{model_name}
