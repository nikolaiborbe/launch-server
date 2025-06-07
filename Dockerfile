# 1) Use official Python slim image
FROM python:3.11-slim AS base

# 2) Ensure logs show up immediately
ENV PYTHONUNBUFFERED=1

# 3) Create a non-root user for safety (optional but recommended)
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# 4) Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy our application code
COPY main.py .

# 6) Switch to non-root user
USER appuser

# 7) Expose the port Uvicorn will run on
EXPOSE 8000

# 8) Define entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]