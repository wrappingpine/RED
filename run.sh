#!/bin/bash
# Air Mouse Run Script
# Starts the air mouse application with proper environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Find project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$HOME/.airmouse-venv"

# Check virtual environment
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    print_error "Virtual environment not found at $VENV_DIR"
    print_info "Run ./install.sh first"
    exit 1
fi

# Check uinput
if [[ ! -c /dev/uinput ]]; then
    print_warning "uinput device not found, attempting to load..."
    sudo modprobe uinput || {
        print_error "Failed to load uinput module"
        exit 1
    }
fi

# Check permissions
if [[ ! -w /dev/uinput ]]; then
    print_warning "No write permission to /dev/uinput"
    print_info "Make sure your user is in the 'input' group:"
    print_info "  sudo usermod -aG input \$USER"
    print_info "Then log out and back in"
fi

# Check camera
print_info "Checking cameras..."
if command -v v4l2-ctl &> /dev/null; then
    v4l2-ctl --list-devices 2>/dev/null | head -20
else
    ls /dev/video* 2>/dev/null || print_warning "No video devices found"
fi

# Activate venv and run
print_info "Starting Air Mouse..."
cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" -m airmouse.ui.main_window "$@"