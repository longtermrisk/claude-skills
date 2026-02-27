"""
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
