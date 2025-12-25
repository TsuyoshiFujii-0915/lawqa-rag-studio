FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir uv==0.4.20
COPY pyproject.toml uv.lock ./
COPY . .
RUN uv pip install --system --no-cache .
CMD ["lawqa-rag-studio", "serve", "--config", "config/config.yaml"]
