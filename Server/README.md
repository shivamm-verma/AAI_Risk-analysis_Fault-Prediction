# AAI Risk Analysis API

## Run locally
1. Reach the directory
    ```sh
    cd ./Server
    ```
1. Create virtual env & install requirements.txt
    ```sh
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

1. Then to finally run the reloading backend
    ```sh
    uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
    ```
    - No reloading version is this
    ```sh
    uvicorn app.main:app --host 0.0.0.0 --port 5000
    ```

## Notes
- Place models in Model/ as .pkl files (e.g., pickle_exmaple.pkl)
- Endpoints:
  - GET /health
  - GET /models
  - POST /predict/{model_name}
