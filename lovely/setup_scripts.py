DATA_CLIENT_CODE = open('fuse_client.py').read()
DATA_SERVER_CODE = open('fuse_server.py').read()

DATA_CLIENT_RUN = f'''
#!/bin/bash
# Exit on any error
set -e

# Function to log messages
log() {{
    echo "$1" >&2
}}

# Function to check if command exists
command_exists() {{
    command -v "$1" >/dev/null 2>&1
}}

# Function to cleanup on failure
cleanup_on_failure() {{
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Script failed with exit code $exit_code"
        log "Cleaning up partial installation..."
        if [ -d "$VENV_DIR" ] && [ "$CLEANUP_ON_FAILURE" = "true" ]; then
            rm -rf "$VENV_DIR"
            log "Removed partial virtual environment"
        fi
    fi
    exit $exit_code
}}

# Set up error handling
trap cleanup_on_failure EXIT

# Configuration
LOVELY_DIR="${{LOVELY_DIR:-$HOME/.lovely}}"
VENV_DIR="$LOVELY_DIR/venv"
PYTHON_CMD="${{PYTHON_CMD:-python3}}"
CLEANUP_ON_FAILURE="${{CLEANUP_ON_FAILURE:-true}}"

log "Starting robust setup process..."

# Check prerequisites
log "Checking prerequisites..."

if ! command_exists "$PYTHON_CMD"; then
    if command_exists "python"; then
        PYTHON_CMD="python"
        log "Using 'python' command instead of 'python3'"
    else
        log "ERROR: Python not found. Please install Python first."
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\\.[0-9]+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 6 ]); then
    log "ERROR: Python 3.6+ required, found $PYTHON_VERSION"
    exit 1
fi

log "Using Python $PYTHON_VERSION"

# Check if venv module is available
if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    log "ERROR: Python venv module not available. Please install python3-venv package."
    exit 1
fi

# Create directory with proper error handling
log "Creating directory: $LOVELY_DIR"
if ! mkdir -p "$LOVELY_DIR"; then
    log "ERROR: Failed to create directory $LOVELY_DIR"
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    log "WARNING: Virtual environment already exists at $VENV_DIR"
    if [ "${{FORCE_RECREATE:-false}}" = "true" ]; then
        log "FORCE_RECREATE=true, removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log "Using existing virtual environment..."
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    if ! $PYTHON_CMD -m venv "$VENV_DIR"; then
        log "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
log "Activating virtual environment..."
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

# Handle Windows case
if [ ! -f "$ACTIVATE_SCRIPT" ] && [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
fi

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    log "ERROR: Activation script not found"
    exit 1
fi

# Source the activation script
if ! source "$ACTIVATE_SCRIPT"; then
    log "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    log "ERROR: Virtual environment activation failed"
    exit 1
fi

log "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip first
log "Upgrading pip..."
if ! python -m pip install --upgrade pip; then
    log "WARNING: Failed to upgrade pip, continuing anyway..."
fi

# Install packages with retry logic
install_package() {{
    local package=$1
    local max_retries=${{2:-3}}
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log "Installing $package (attempt $((retry_count + 1))/$max_retries)..."
        
        if python -m pip install "$package"; then
            log "Successfully installed $package"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                log "Installation failed, retrying in 2 seconds..."
                sleep 2
            fi
        fi
    done
    
    log "ERROR: Failed to install $package after $max_retries attempts"
    return 1
}}

# Install required packages
install_package "fusepy" || exit 1

# Verify installation
log "Verifying installation..."
if ! python -c "import fuse; print('fusepy successfully imported')"; then
    log "ERROR: fusepy installation verification failed"
    exit 1
fi

# Run the embedded client code
log "Running embedded client code..."
if ! python << 'EOF'
<MARK:CLIENT_CODE>
EOF
then
    log "ERROR: Client code execution failed"
    exit 1
fi

log "Setup completed successfully!"
log "To activate this environment in the future, run:"
log "  source $ACTIVATE_SCRIPT"

# Disable error trap for successful completion
trap - EXIT
'''.replace("<MARK:CLIENT_CODE>", DATA_CLIENT_CODE)
