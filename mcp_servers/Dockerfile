FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mcp==1.27.1 stripe==15.2.0

COPY stripe_support.py .

# FastMCP reads host/port from these env vars when transport=sse
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8001
EXPOSE 8001

CMD ["python", "stripe_support.py", "--transport", "sse"]
