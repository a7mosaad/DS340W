#!/bin/bash
set -euo pipefail

echo "Starting Tableau MCP server setup..."

# Proxy configuration
PROXY_URL="${PROXY_URL:-http://naappdproxylttppopers.wlb.nam.nsroot.net:9999}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,::1,citigroup.net,nsroot.net,nsrootdev.net}"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

# Node configuration
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion"

# Node.js CA certificates
export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-/app/CitiInternalCAChain_Combined.pem}"

# Extract Tableau PAT credentials from vault
VAULT_FOUND=false
VAULT_FILE="${VAULT_FILE:-/var/vault-data/tableau}"

if [ -f "$VAULT_FILE" ]; then
    PAT_NAME=$(grep -i '^name=' "$VAULT_FILE" | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    PAT_VALUE=$(grep -i '^token=' "$VAULT_FILE" | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    export PAT_NAME
    export PAT_VALUE

    VAULT_FOUND=true
    echo "Loaded Tableau PAT credentials and URL from vault"
    echo "PAT_NAME resolved to: '$PAT_NAME'"
else
    echo "Warning: Vault file $VAULT_FILE not found - PAT_NAME, PAT_VALUE, and URL must be set manually"
fi

# Tableau MCP configuration
export SERVER="${SERVER:-https://uat.cisocsfctableau.citigroup.net}"
export TRANSPORT="${TRANSPORT:-http}"
export AUTH="${AUTH:-pat}"
export PORT="${PORT:-8001}"
export DANGEROUSLY_DISABLE_OAUTH="${DANGEROUSLY_DISABLE_OAUTH:-true}"
export DEFAULT_LOG_LEVEL="${DEFAULT_LOG_LEVEL:-debug}"
export NODE_TLS_REJECT_UNAUTHORIZED="${NODE_TLS_REJECT_UNAUTHORIZED:-0}"
export TELEMETRY_PROVIDER="${TELEMETRY_PROVIDER:-noop}"
export API_VERSION="${API_VERSION:-3.19}"
export TABLEAU_SSL_VERIFY="${TABLEAU_SSL_VERIFY:-false}"
export TABLEAU_REQUEST_TIMEOUT="${TABLEAU_REQUEST_TIMEOUT:-30000}"
export FORCE_NEW_SESSION="${FORCE_NEW_SESSION:-true}"

HOST="${HOST:-127.0.0.1}"
MCP_PATH="${MCP_PATH:-/tableau-mcp}"
CONFIG_FILE="${CONFIG_FILE:-/app/tableau-mcp/config.http.json}"
REPO_DIR="${REPO_DIR:-/app/tableau-mcp}"

# Clone the Tableau MCP repo if not already present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Tableau MCP repository..."
    git clone https://github.com/tableau/tableau-mcp.git "$REPO_DIR"
else
    echo "Tableau MCP repository already exists. Skipping clone."
fi

# Write .env file only if vault credentials were found
if [ "$VAULT_FOUND" = true ]; then
    echo "Writing .env file..."
    cat > "$REPO_DIR/.env" <<EOF_ENV
TRANSPORT=${TRANSPORT}
DANGEROUSLY_DISABLE_OAUTH=${DANGEROUSLY_DISABLE_OAUTH}
PORT=${PORT}
AUTH=${AUTH}
SERVER=${SERVER}
PAT_NAME=${PAT_NAME}
PAT_VALUE=${PAT_VALUE}
DEFAULT_LOG_LEVEL=${DEFAULT_LOG_LEVEL}
NODE_TLS_REJECT_UNAUTHORIZED=${NODE_TLS_REJECT_UNAUTHORIZED}
TELEMETRY_PROVIDER=${TELEMETRY_PROVIDER}
API_VERSION=${API_VERSION}
TABLEAU_SSL_VERIFY=${TABLEAU_SSL_VERIFY}
TABLEAU_REQUEST_TIMEOUT=${TABLEAU_REQUEST_TIMEOUT}
FORCE_NEW_SESSION=${FORCE_NEW_SESSION}
EOF_ENV
    echo ".env file written successfully."
fi

# Patch config.http.json so future Inspector runs use the same server port
python3 - <<PY
import json
from pathlib import Path

config_path = Path("${CONFIG_FILE}")
url = f"http://${HOST}:${PORT}${MCP_PATH}"

if config_path.exists():
    data = json.loads(config_path.read_text())
    patched = False

    if "mcpServers" in data and "tableau" in data["mcpServers"]:
        server = data["mcpServers"]["tableau"]
        if isinstance(server, dict):
            if "url" in server:
                server["url"] = url
                patched = True
            elif "transport" in server and isinstance(server["transport"], dict):
                server["transport"]["url"] = url
                patched = True

    if patched:
        config_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Patched {config_path} -> {url}")
    else:
        print(f"Skipped patching {config_path}: unexpected tableau config shape")
else:
    print(f"Skipped patching {config_path}: file not found")
PY

# Install and build
cd "$REPO_DIR"

echo "Running npm install"
npm install

echo "Building Tableau MCP server..."
npm run build

echo "Starting Tableau MCP HTTP server on http://${HOST}:${PORT}${MCP_PATH}"
exec npm run start:http
