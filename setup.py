#!/usr/bin/env python3
"""Air Mouse - Setup script for packaging."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

requirements = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "mediapipe>=0.10.0",
    "PySide6>=6.5.0",
    "tomli>=2.0.0; python_version < '3.11'",
    "tomli-w>=1.0.0",
]

setup(
    name="airmouse",
    version="1.0.0",
    description="Control your mouse with hand gestures via webcam on Linux",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Air Mouse Contributors",
    author_email="",
    url="https://github.com/yourusername/airmouse",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "airmouse=airmouse.ui.main_window:run_gui",
            "airmouse-diagnose=airmouse.main:diagnose",
            "airmouse-test-camera=airmouse.main:test_camera",
            "airmouse-test-hand=airmouse.main:test_hand",
            "airmouse-test-mouse=airmouse.main:test_mouse",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Desktop Environment",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    keywords="mouse gesture hand tracking webcam accessibility linux uinput",
)