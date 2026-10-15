FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY . .
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
CMD ["sh","-c","uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
