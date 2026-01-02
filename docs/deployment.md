# Deployment Guide: Hosting the Fuzzy LSTM Dashboard

This document describes how to deploy the Streamlit-based dashboard for the fuzzy LSTM model on:

- The OpenBB platform (if/when they provide managed hosting for custom apps or containerized workloads).
- Any container service (e.g., AWS ECS/Fargate, Google Cloud Run, Azure Container Apps, Fly.io, Render, Heroku-like platforms).

The dashboard provides:
- Curated blue chips across USA, India, China, and Singapore.
- Portfolio creation and persistence (up to 6 tickers).
- Model parameter configuration (eps, min_samples, epochs, lr, training date range).
- Forecasts t+1..t+13, actual vs predicted visualization for t-13..t.
- Forecasted MACD (FMACD) signals and equal-weight portfolio returns.

> Note: Data can be sourced via OpenBB SDK or yfinance. The UI is built with Streamlit for rapid and reliable web-hosting.


---

## 1) Architecture Overview

- Web UI: Streamlit app (`src/tamira_fin_ui/dashboard_app.py`)
- Model + Utils: Fuzzy LSTM (PyTorch, scikit-learn), MACD, portfolio storage (Pydantic)
- Persistence: Local JSON file (default `~/.tamira_fin_ui/portfolios.json`) configurable via `TAMIRA_STORAGE_PATH`
- Containerization: Dockerfile provided to build a production-ready image
- Port: `8501` (Streamlit default)

You can host this as:
- A container on an OpenBB-managed platform (if available)
- A container on your preferred cloud service
- A local process for internal demo or development


---

## 2) Prerequisites

- Python 3.12 (for local runs) or Docker (for containerized deployment)
- Basic understanding of container hosting (push to registry, deploy, configure env)
- Optional: OpenBB SDK credentials or provider keys (e.g., Polygon/Tiingo) if you prefer to source data via OpenBB instead of yfinance


---

## 3) Local Development (without containers)

This approach is ideal for quick testing and iteration.

1. Create a virtual environment and install deps (with `uv`):
   ```
   uv venv
   source .venv/bin/activate
   uv sync
   uv sync --extra dev
   ```

2. Run the dashboard:
   ```
   streamlit run src/tamira_fin_ui/dashboard_app.py
   ```
   Or, using project scripts:
   ```
   uv run dashboard
   ```

3. Open your browser at:
   ```
   http://localhost:8501
   ```


---

## 4) Container Build & Run (Docker)

A `Dockerfile` is provided at the project root. It installs runtime dependencies, copies `src/`, and starts Streamlit.

### 4.1 Build the image

From the project root:
```
docker build -t your-registry/tamira-fin-ui:latest .
```

### 4.2 Run locally

Map a local volume for persistent portfolio storage:
```
docker run --rm -it \
  -p 8501:8501 \
  -e TAMIRA_STORAGE_PATH=/data/portfolios.json \
  -v $(pwd)/.data:/data \
  your-registry/tamira-fin-ui:latest
```

Open your browser at:
```
http://localhost:8501
```

### 4.3 Environment variables

- `TAMIRA_STORAGE_PATH` (string): Path to the JSON file for portfolio persistence (default `~/.tamira_fin_ui/portfolios.json`). In containers, set this to an external volume path, e.g., `/data/portfolios.json`.

- Data provider keys (optional):
  - If you integrate OpenBB and specific providers, set provider-specific environment variables (e.g., `POLYGON_API_KEY`, `TIINGO_API_KEY`) according to the provider’s requirements and your data adapter.

### 4.4 Health check

The container defines a HEALTHCHECK targeting:
```
http://localhost:8501/
```

Use your platform’s native health monitoring. Expect a 200 OK from Streamlit once the app is alive.

### 4.5 Resource notes

- CPU-only is sufficient. If you later add GPU acceleration (PyTorch CUDA), ensure the base image and platform support it.
- Streamlit apps are typically stateless beyond storage; ensure the storage volume is mounted on every replica if you scale horizontally.


---

## 5) Deploying to the OpenBB Platform

If the OpenBB platform supports hosting custom dashboards or containerized applications:

1. Build and push your container image to a registry accessible by OpenBB:
   ```
   docker build -t your-registry/tamira-fin-ui:latest .
   docker push your-registry/tamira-fin-ui:latest
   ```

2. Create a new app/service in OpenBB’s platform (follow their console or CLI workflow).
   - Specify the image: `your-registry/tamira-fin-ui:latest`
   - Set the exposed port: `8501`
   - Add environment variables:
     - `TAMIRA_STORAGE_PATH=/data/portfolios.json`
     - Any OpenBB/provider keys you use (e.g., `POLYGON_API_KEY`)
   - Attach a persistent volume and mount it at `/data` (or change `TAMIRA_STORAGE_PATH` accordingly).

3. Configure networking (public URL / domain, TLS/HTTPS, etc.).
4. Deploy and validate:
   - Check health status
   - Navigate to the app URL
   - Verify portfolio persistence (create a portfolio, refresh, ensure it remains available)

> If OpenBB exposes a native “App Hub” or templating system for Streamlit apps, you can adapt this project accordingly:
> - Keep `src/tamira_fin_ui/dashboard_app.py` as your entrypoint
> - Provide configuration via `.streamlit/config.toml` and environment variables
> - Confirm the platform’s guidance on how to register Streamlit apps


---

## 6) Deploying to Common Container Services

### 6.1 Google Cloud Run (serverless containers)

1. Build & push:
   ```
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/tamira-fin-ui
   ```

2. Deploy:
   ```
   gcloud run deploy tamira-fin-ui \
     --image gcr.io/<PROJECT_ID>/tamira-fin-ui \
     --platform managed \
     --region <REGION> \
     --allow-unauthenticated \
     --port 8501 \
     --set-env-vars TAMIRA_STORAGE_PATH=/data/portfolios.json
   ```

3. Add a Cloud Storage or Filestore-backed volume if persistent storage is needed (Cloud Run’s persistent volumes are limited; alternatively store portfolios in a managed database).

### 6.2 AWS ECS/Fargate

- Create a Task Definition using your image.
- Define container port `8501`.
- Add environment variables and mount a persistent volume (EFS for multi-task access).
- Expose via ALB with health checks targeting `/`.

### 6.3 Azure Container Apps

- Create a Container App from your image.
- Configure environment variables and persistent storage (Azure Files).
- Expose port `8501`.

### 6.4 Render / Fly.io

- Create a new service/app pointing to your image or repository.
- Set port to `8501`, configure env vars and volume for persistence.
- Deploy and verify the public URL works as expected.

> Note: Some platforms don’t support persistent volumes seamlessly. If portfolio persistence is critical in such environments, consider storing portfolios in a managed DB (e.g., SQLite with cloud disk, or Postgres hosted service). The code can be adapted to use Pydantic models backed by a simple DB layer.


---

## 7) Streamlit Configuration

A `.streamlit/config.toml` is included with production defaults:

- Headless mode
- Bind to `0.0.0.0`
- Port `8501`
- CORS/XSRF protections enabled
- Poll-based file watching for container environments

You can modify the theme or security settings if your platform requires specific configurations.

Example (already in the repo):
```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = true
enableXsrfProtection = true
fileWatcherType = "poll"

[theme]
primaryColor = "#2E86C1"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F7FA"
textColor = "#1C2833"
font = "sans serif"
```


---

## 8) Security & Observability

- TLS termination: prefer running behind a managed reverse proxy or load balancer (HTTPS). Streamlit listens in HTTP by default.
- Authentication: Streamlit does not include built-in auth; front external auth (e.g., single sign-on via your platform’s gateway).
- Logs:
  - Container logs should be visible in your platform’s console.
  - Use the platform’s native logging/monitoring integrations (e.g., Cloud Logging, CloudWatch, Azure Monitor).
- Rate limiting:
  - Data provider APIs may rate-limit; consider caching or scheduling background computations if needed.


---

## 9) Customizing Data Sources

- Default: yfinance is used in the model utilities.
- OpenBB SDK integration:
  - Swap calls to `yf.download` with an adapter that queries `openbb` SDK endpoints (equity historical price).
  - Add provider selection and API key inputs via environment variables or a small UI settings panel.
- Fallback strategy:
  - Try OpenBB first; fallback to yfinance on failure.


---

## 10) Troubleshooting

- Port not reachable:
  - Ensure the platform maps external port to `8501`.
  - Verify `address = "0.0.0.0"` in Streamlit config.
- Persistent storage not working:
  - Confirm volume mounts and environment variable `TAMIRA_STORAGE_PATH` are set.
  - Inspect container logs for file write errors.
- Slow forecasts:
  - Reduce `epochs` and/or `max_horizon` in the UI.
  - Pre-compute forecasts for popular tickers and cache results.
- Data provider errors:
  - Check rate limits and credentials.
  - Reduce frequency of requests or add caching.


---

## 11) Summary

- Build and run locally using `uv` or `streamlit run`.
- Use the provided `Dockerfile` to build a production image.
- Deploy to OpenBB platform (if supported) or any container hosting service.
- Configure environment variables and persistent volumes for portfolio storage.
- Optionally integrate OpenBB SDK for provider-agnostic data retrieval.
- Streamlit provides a clean, fast UI layer with minimal configuration overhead.

If you want a fully managed OpenBB-hosted app workflow, share the platform’s hosting specifics and I’ll tailor this guide and the code (provider adapters, auth, telemetry) to their deployment standard.
