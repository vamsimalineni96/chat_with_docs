#!/bin/bash
# Start the Stripe MCP container using the key from the project .env file.
# Usage: ./mcp_servers/run.sh
# Run from the repo root.

set -e

ENV_FILE="$(dirname "$0")/../.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  exit 1
fi

STRIPE_KEY=$(grep -E '^STRIPE_SECRET_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")

if [ -z "$STRIPE_KEY" ]; then
  echo "ERROR: STRIPE_SECRET_KEY not set in .env"
  exit 1
fi

echo "Starting stripe-mcp on port 8001..."
docker rm -f stripe-mcp 2>/dev/null || true
docker run --rm -p 8001:8001 \
  -e STRIPE_SECRET_KEY="$STRIPE_KEY" \
  --name stripe-mcp \
  stripe-mcp
