#!/bin/bash
# Air Mouse Installation Script
# Installs dependencies and sets up the virtual mouse on Linux (Pop!_OS/Ubuntu/Debian)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    print_error "Do not run this script as root. Run as regular user."
    exit 1
fi

print_info "Air Mouse Installation Script"
print_info "=============================="

# Detect distribution
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
    print_info "Detected: $PRETTY_NAME"
else
    print_warning "Could not detect distribution, assuming Ubuntu/Debian"
    DISTRO="ubuntu"
fi

# Update package list
print_info "Updating package list..."
sudo apt update

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    v4l-utils \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    libnss3 \
    libatspi2.0-0 \
    libdrm2 \
    libgbm1 \
    libasound2

# Load uinput kernel module
print_info "Loading uinput kernel module..."
sudo modprobe uinput

# Make uinput persistent across reboots
if ! grep -q "^uinput$" /etc/modules 2>/dev/null; then
    print_info "Adding uinput to /etc/modules for persistence..."
    echo "uinput" | sudo tee -a /etc/modules > /dev/null
fi

# Set up uinput permissions
print_info "Setting up uinput permissions..."
sudo groupadd -f input
sudo usermod -aG input $USER

# Create udev rule for uinput access
UDEV_RULE='/etc/udev/rules.d/99-uinput.rules'
if [[ ! -f "$UDEV_RULE" ]]; then
    print_info "Creating udev rule for uinput..."
    echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee "$UDEV_RULE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
fi

# Create virtual environment
VENV_DIR="$HOME/.airmouse-venv"
print_info "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate venv and install Python dependencies
print_info "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# Install the air mouse package in development mode
print_info "Installing air mouse package..."
"$VENV_DIR/bin/pip" install -e .

# Create desktop entry
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"

print_info "Creating desktop entry..."
cat > "$DESKTOP_DIR/airmouse.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Air Mouse
Comment=Control your mouse with hand gestures via webcam
Exec=$VENV_DIR/bin/python -m airmouse.ui.main_window
Icon=webcam
Terminal=false
Categories=Utility;Accessibility;
StartupNotify=true
Keywords=mouse;gesture;hand;tracking;webcam;accessibility;
EOF

# Create run script
RUN_SCRIPT="$HOME/.local/bin/airmouse"
mkdir -p "$HOME/.local/bin"

print_info "Creating run script at $RUN_SCRIPT..."
cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
# Air Mouse Run Script

VENV_DIR="$HOME/.airmouse-venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Check uinput
if [[ ! -c /dev/uinput ]]; then
    echo "uinput device not found. Loading kernel module..."
    sudo modprobe uinput
fi

# Check user is in input group
if ! groups | grep -q '\binput\b'; then
    echo "Warning: User not in 'input' group. Virtual mouse may not work."
    echo "Run: sudo usermod -aG input \$USER"
    echo "Then log out and back in."
fi

# Run the application
cd "$(dirname "$0")/../airmouse" 2>/dev/null || cd "$HOME/airmouse" 2>/dev/null || true
exec "$VENV_DIR/bin/python" -m airmouse.ui.main_window "$@"
EOF

chmod +x "$RUN_SCRIPT"

# Add ~/.local/bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    print_info "Adding ~/.local/bin to PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

print_success "Installation complete!"
echo
print_info "Next steps:"
echo "  1. Log out and log back in (or run 'newgrp input') for group changes to take effect"
echo "  2. Run 'airmouse' from terminal or find 'Air Mouse' in your application menu"
echo "  3. Grant camera permission when prompted"
echo
print_warning "Note: You may need to reboot for uinput permissions to fully apply."
print_info "To uninstall, run: $HOME/.airmouse-venv/bin/pip uninstall airmouse"