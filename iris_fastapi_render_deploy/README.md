# Iris FastAPI on Render

Educational demo. Train in Colab; serve the saved model with FastAPI.

Build: `pip install -r requirements.txt`

Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Choose Python runtime and Free instance. Health check: `/health`.
Open `/docs` to test `POST /predict`.
