#!/usr/bin/env python3
"""
Validate that openweights is properly installed and configured.

Usage:
    python validate_setup.py
"""

import os
import sys
import subprocess


def check_installation():
    """Check if openweights is installed."""
    try:
        result = subprocess.run(
            ["python", "-c", "import openweights; print(openweights.__version__)"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ openweights installed (version {result.stdout.strip()})")
            return True
        else:
            print("❌ openweights not installed")
            print("   Install with: pip install openweights")
            return False
    except Exception as e:
        print(f"❌ Error checking openweights: {e}")
        return False


def check_api_key():
    """Check if OPENWEIGHTS_API_KEY is set."""
    api_key = os.getenv("OPENWEIGHTS_API_KEY")
    if api_key:
        print(f"✅ OPENWEIGHTS_API_KEY is set")
        return True
    else:
        print("❌ OPENWEIGHTS_API_KEY not set")
        print("   Set with: export OPENWEIGHTS_API_KEY=your_key_here")
        print("   Or create one with: ow signup")
        return False


def check_optional_keys():
    """Check optional environment variables."""
    print("\nOptional configuration:")

    optional_vars = {
        "RUNPOD_API_KEY": "Required for self-managed clusters",
        "HF_TOKEN": "Required for pushing models to HuggingFace",
        "HF_USER": "HuggingFace username",
        "HF_ORG": "HuggingFace organization",
    }

    for var, description in optional_vars.items():
        if os.getenv(var):
            print(f"  ✅ {var} is set")
        else:
            print(f"  ⚠️  {var} not set - {description}")


def main():
    print("🔍 Validating OpenWeights Setup\n")

    checks = [
        check_installation(),
        check_api_key(),
    ]

    check_optional_keys()

    print("\n" + "="*50)
    if all(checks):
        print("✅ Setup complete! Ready to use openweights.")
        return 0
    else:
        print("❌ Setup incomplete. Fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
