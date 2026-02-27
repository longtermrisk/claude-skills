#!/usr/bin/env python3
"""
Build script for the openweights skill.

This script generates the skill's reference documentation and assets
from the openweights repository, making it easy to regenerate when
openweights is updated.

Usage:
    python build_skill.py /path/to/openweights/repo
"""

import os
import shutil
import sys
from pathlib import Path


def copy_file(src, dst):
    """Copy a file, creating parent directories if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  ✓ {dst.relative_to(skill_dir)}")


def copy_tree(src, dst):
    """Copy a directory tree."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  ✓ {dst.relative_to(skill_dir)}")


def extract_section(content, start_marker, end_marker=None):
    """Extract a section from markdown content."""
    lines = content.split('\n')
    in_section = False
    result = []

    for line in lines:
        if start_marker in line:
            in_section = True
            result.append(line)
        elif end_marker and end_marker in line:
            break
        elif in_section:
            result.append(line)

    return '\n'.join(result)


def build_references(repo_path):
    """Build reference documentation from repo files."""
    print("\n📚 Building references...")

    refs_dir = skill_dir / "references"
    refs_dir.mkdir(exist_ok=True)

    # Copy main README for overview
    readme = repo_path / "README.md"
    with open(readme) as f:
        readme_content = f.read()

    # Split README into sections
    with open(refs_dir / "01_overview.md", "w") as f:
        f.write("# OpenWeights Overview\n\n")
        f.write(readme_content.split("## Core Concepts")[0])
    print(f"  ✓ references/01_overview.md")

    # Copy cookbook README as workflow guide
    cookbook_readme = repo_path / "cookbook" / "README.md"
    if cookbook_readme.exists():
        with open(cookbook_readme) as f:
            cookbook_content = f.read()
        with open(refs_dir / "02_workflows.md", "w") as f:
            f.write("# OpenWeights Workflows\n\n")
            f.write(cookbook_content)
        print(f"  ✓ references/02_workflows.md")

    # Copy CLAUDE.md (architecture) if it exists
    claude_md = repo_path / "CLAUDE.md"
    if claude_md.exists():
        copy_file(claude_md, refs_dir / "03_architecture.md")

    # Copy .env.worker.example as reference
    env_example = repo_path / ".env.worker.example"
    if env_example.exists():
        copy_file(env_example, refs_dir / "env.worker.example")


def build_scripts(repo_path):
    """Build utility scripts."""
    print("\n🔧 Building scripts...")

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Remove example script
    example_script = scripts_dir / "example.py"
    if example_script.exists():
        example_script.unlink()

    # Create setup validation script
    setup_script = scripts_dir / "validate_setup.py"
    with open(setup_script, "w") as f:
        f.write('''#!/usr/bin/env python3
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
    print("\\nOptional configuration:")

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
    print("🔍 Validating OpenWeights Setup\\n")

    checks = [
        check_installation(),
        check_api_key(),
    ]

    check_optional_keys()

    print("\\n" + "="*50)
    if all(checks):
        print("✅ Setup complete! Ready to use openweights.")
        return 0
    else:
        print("❌ Setup incomplete. Fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
''')
    setup_script.chmod(0o755)
    print(f"  ✓ scripts/validate_setup.py")

    # Create quick setup script
    quickstart_script = scripts_dir / "quickstart.sh"
    with open(quickstart_script, "w") as f:
        f.write('''#!/bin/bash
# Quick setup script for openweights

set -e

echo "🚀 OpenWeights Quick Setup"
echo "=========================="
echo ""

# Check if openweights is installed
if ! python -c "import openweights" 2>/dev/null; then
    echo "📦 Installing openweights..."
    pip install openweights
else
    echo "✅ openweights already installed"
fi

# Check for API key
if [ -z "$OPENWEIGHTS_API_KEY" ]; then
    echo ""
    echo "⚠️  OPENWEIGHTS_API_KEY not set"
    echo ""
    echo "To get an API key, run:"
    echo "  ow signup"
    echo ""
    echo "Then set it with:"
    echo "  export OPENWEIGHTS_API_KEY=your_key_here"
    echo ""
    echo "Or add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "  echo 'export OPENWEIGHTS_API_KEY=your_key_here' >> ~/.bashrc"
    echo ""
else
    echo "✅ OPENWEIGHTS_API_KEY is set"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Validate setup: python scripts/validate_setup.py"
echo "  2. Try an example: python assets/examples/sft/lora_qwen3_4b.py"
''')
    quickstart_script.chmod(0o755)
    print(f"  ✓ scripts/quickstart.sh")


def build_assets(repo_path):
    """Build example assets from cookbook."""
    print("\n📦 Building assets...")

    assets_dir = skill_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Remove example asset
    example_asset = assets_dir / "example_asset.txt"
    if example_asset.exists():
        example_asset.unlink()

    # Copy entire cookbook as examples
    cookbook_src = repo_path / "cookbook"
    cookbook_dst = assets_dir / "examples"
    if cookbook_src.exists():
        copy_tree(cookbook_src, cookbook_dst)

    # Create data format templates
    templates_dir = assets_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Conversations template
    with open(templates_dir / "conversations.jsonl", "w") as f:
        f.write('''{
    "messages": [
        {
            "role": "user",
            "content": "This is a user message"
        },
        {
            "role": "assistant",
            "content": "This is the assistant response"
        }
    ]
}
''')
    print(f"  ✓ assets/templates/conversations.jsonl")

    # Preferences template
    with open(templates_dir / "preferences.jsonl", "w") as f:
        f.write('''{
    "prompt": [
        {
            "role": "user",
            "content": "Would you use the openweights library to finetune LLMs?"
        }
    ],
    "chosen": [
        {
            "role": "assistant",
            "content": "Absolutely, it's a great library"
        }
    ],
    "rejected": [
        {
            "role": "assistant",
            "content": "No, I would use something else"
        }
    ]
}
''')
    print(f"  ✓ assets/templates/preferences.jsonl")

    # Custom job template
    custom_job_template = templates_dir / "custom_job_template.py"
    with open(custom_job_template, "w") as f:
        f.write('''"""
Template for creating a custom OpenWeights job.

Customize this template for your specific needs.
"""

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel, Field
import json
import os

ow = OpenWeights()


class MyJobParams(BaseModel):
    """Parameters for my custom job."""

    # Add your parameters here
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(default=10, description="Description of param2")


@register("my_custom_job")
class MyCustomJob(Jobs):
    """Custom job implementation."""

    # Mount local files/directories that will be available in the worker
    mount = {
        os.path.join(os.path.dirname(__file__), "worker_script.py"): "worker_script.py"
        # Add more files/directories to mount
    }

    # Define parameter validation
    params = MyJobParams

    # VRAM requirements (GB)
    requires_vram_gb = 24

    # Optional: specify base Docker image
    # base_image = 'nielsrolf/ow-default'

    def get_entrypoint(self, validated_params: MyJobParams) -> str:
        """Create the command to run the job."""
        params_json = json.dumps(validated_params.model_dump())
        return f"python worker_script.py '{params_json}'"


def main():
    """Submit the custom job."""

    # Create job with your parameters
    job = ow.my_custom_job.create(
        param1="value1",
        param2=20
    )

    print(f"Created job: {job.id}")
    print(f"Status: {job.status}")

    # Optional: wait for completion
    import time
    while job.refresh().status in ["pending", "in_progress"]:
        print(f"Status: {job.status}")
        time.sleep(5)

    if job.status == "completed":
        print(f"Job completed: {job.outputs}")
    else:
        print(f"Job failed: {job}")


if __name__ == "__main__":
    main()
''')
    print(f"  ✓ assets/templates/custom_job_template.py")


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_skill.py /path/to/openweights/repo")
        sys.exit(1)

    global skill_dir
    repo_path = Path(sys.argv[1]).resolve()
    skill_dir = Path(__file__).parent.resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / "README.md").exists():
        print(f"Error: Not an openweights repository (README.md not found)")
        sys.exit(1)

    print(f"🏗️  Building openweights skill from: {repo_path}")
    print(f"   Target: {skill_dir}")

    build_references(repo_path)
    build_scripts(repo_path)
    build_assets(repo_path)

    print("\n✅ Skill built successfully!")
    print("\nNext steps:")
    print("  1. Review and edit SKILL.md")
    print("  2. Test the skill")
    print("  3. Package with: python scripts/package_skill.py")


if __name__ == "__main__":
    main()
