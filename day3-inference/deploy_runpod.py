#!/usr/bin/env python3
"""
Deploy Day 3 bootcamp pods on RunPod.

Creates GPU pods, uploads exercise files via Jupyter API, and installs deps.
Participants connect via Jupyter (browser) — no SSH keys needed.

Usage:
    python deploy_runpod.py                    # Deploy 1 pod
    python deploy_runpod.py --count 15         # Deploy 15 pods
    python deploy_runpod.py --list             # List running pods
    python deploy_runpod.py --stop-all         # Terminate all bootcamp pods
    python deploy_runpod.py --stop <pod_id>    # Terminate one pod
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from dotenv import dotenv_values

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

env_values = dotenv_values(DOTENV_PATH)
if env_values["RUNPOD_API_KEY"]:
    os.environ["RUNPOD_API_KEY"] = env_values["RUNPOD_API_KEY"]

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
API_URL = "https://api.runpod.io/graphql"

POD_NAME_PREFIX = "day3-bootcamp"
GPU_TYPE_IDS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
    "NVIDIA A40",
]
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 50
VOLUME_GB = 50
VOLUME_MOUNT = "/workspace"
PORTS = "8888/http,22/tcp"
JUPYTER_PASSWORD = "day3bootcamp"

# SSH public key injected at pod startup so we can run commands remotely
SSH_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILeUXQNeYJfPonOG/rG2XM1qbiv5VkyRtBOuxADhw2zz claude@000ddb9416eb"

# Files to upload to each pod
EXERCISE_FILES = [
    "poc_distillation.py",
    "poc_guardrails.py",
    "day3_solution.py",
    "day3_test.py",
    "day3_instructions.md",
    "DAY3_PLAN.md",
    "setup_pod.sh",
    "run_exercise.sh",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# GraphQL helpers
# ─────────────────────────────────────────────────────────────────────────────

def graphql(query: str, variables: dict | None = None) -> dict:
    import subprocess, tempfile

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    url = f"{API_URL}?api_key={API_KEY}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name

    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp}"],
            capture_output=True, text=True, timeout=30,
        )
        resp = json.loads(result.stdout)
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.unlink(tmp)

    if "errors" in resp:
        print(f"GraphQL errors: {json.dumps(resp['errors'], indent=2)}", file=sys.stderr)
        sys.exit(1)

    return resp["data"]


# ─────────────────────────────────────────────────────────────────────────────
# Jupyter API helpers (upload files to running pods)
# ─────────────────────────────────────────────────────────────────────────────

def _curl_request(url: str, data: str, method: str = "PUT") -> tuple[int, str]:
    """Make an HTTP request using curl (bypasses Cloudflare bot detection)."""
    import subprocess, tempfile
    # Write data to a temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        result = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "-X", method, url,
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: token {JUPYTER_PASSWORD}",
             "-d", f"@{tmp}"],
            capture_output=True, text=True, timeout=60,
        )
        lines = result.stdout.strip().rsplit("\n", 1)
        body = lines[0] if len(lines) > 1 else ""
        code = int(lines[-1]) if lines[-1].isdigit() else 0
        return code, body
    except Exception as e:
        return 0, str(e)
    finally:
        os.unlink(tmp)


def jupyter_upload_file(pod_id: str, remote_path: str, content: str) -> bool:
    """Upload a file to a running pod via the Jupyter Contents API."""
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"
    url = f"{base_url}/api/contents/{remote_path}"

    body = json.dumps({"type": "file", "format": "text", "content": content})
    code, _ = _curl_request(url, body)
    return code in (200, 201)


def jupyter_create_dir(pod_id: str, dir_path: str) -> bool:
    """Create a directory on the pod via Jupyter API."""
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"
    url = f"{base_url}/api/contents/{dir_path}"

    body = json.dumps({"type": "directory"})
    code, _ = _curl_request(url, body)
    return code in (200, 201)


def jupyter_run_command(pod_id: str, command: str) -> str | None:
    """Execute a command on the pod via Jupyter's terminal API.

    This creates a terminal, sends the command, and returns.
    For long-running commands, the command runs asynchronously.
    """
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"

    # Create a terminal
    url = f"{base_url}/api/terminals"
    req = urllib.request.Request(url, data=b'{}', method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"token {JUPYTER_PASSWORD}",
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            terminal = json.loads(resp.read().decode())
            term_name = terminal["name"]
    except Exception as e:
        print(f"    Could not create terminal: {e}", file=sys.stderr)
        return None

    # Send command via the terminal websocket API isn't easy with urllib,
    # so instead we'll upload a script and run it via the Jupyter kernel.
    # For our use case, uploading a script + instructing the user works fine.
    return term_name


def wait_for_jupyter(pod_id: str, timeout: int = 180) -> bool:
    """Wait for Jupyter to be accessible on the pod."""
    import subprocess
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"
    url = f"{base_url}/api/contents"

    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 url, "-H", f"Authorization: token {JUPYTER_PASSWORD}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip() == "200":
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def upload_exercises_to_pod(pod_id: str) -> bool:
    """Upload all exercise files to the pod."""
    print(f"  Uploading exercise files to pod {pod_id}...")

    # Jupyter root is /, so workspace is at /workspace
    jupyter_create_dir(pod_id, "workspace/exercises")

    success = True
    for filename in EXERCISE_FILES:
        filepath = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"    Skipping {filename} (not found)")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        remote_path = f"workspace/exercises/{filename}"
        if jupyter_upload_file(pod_id, remote_path, content):
            print(f"    Uploaded {filename}")
        else:
            print(f"    FAILED to upload {filename}")
            success = False

    # Upload a quick-start notebook for easy iteration
    notebook = create_quickstart_notebook()
    if jupyter_upload_file(pod_id, "workspace/exercises/quickstart.ipynb", notebook):
        print(f"    Uploaded quickstart.ipynb")

    return success


def create_quickstart_notebook() -> str:
    """Create a Jupyter notebook that helps participants get started."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Day 3: LLM Inference Security - Quick Start\n",
                    "\n",
                    "**Run the first two cells in order.** Cell 1 installs dependencies (~1 min), Cell 2 downloads models (~5 min).\n",
                    "After that, you can run any exercise cell."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 1: Install dependencies and check GPU\n",
                    "!pip install -q transformers accelerate scikit-learn tqdm ipywidgets\n",
                    "\n",
                    "import torch\n",
                    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU: {torch.cuda.get_device_name()}')\n",
                    "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 2: Download models (run once, ~5 min)\n",
                    "import os\n",
                    "os.environ['HF_HOME'] = '/workspace/model-cache'\n",
                    "os.environ['TRANSFORMERS_CACHE'] = '/workspace/model-cache'\n",
                    "\n",
                    "from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel\n",
                    "import torch\n",
                    "\n",
                    "CACHE = '/workspace/model-cache'\n",
                    "\n",
                    "# Tokenizers only\n",
                    "for name in ['NousResearch/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B',\n",
                    "             'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'unsloth/gemma-2-2b-it']:\n",
                    "    print(f'Downloading tokenizer: {name}')\n",
                    "    AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)\n",
                    "\n",
                    "# Full models\n",
                    "for name in ['google/gemma-4-E4B-it', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B']:\n",
                    "    print(f'Downloading model: {name}')\n",
                    "    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, cache_dir=CACHE, trust_remote_code=True)\n",
                    "\n",
                    "print('Downloading GPT-2...')\n",
                    "GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=CACHE)\n",
                    "GPT2LMHeadModel.from_pretrained('openai-community/gpt2', cache_dir=CACHE)\n",
                    "\n",
                    "print('All models downloaded!')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Exercise 4: Guardrails\n",
                    "Run individual levels to see the progression:"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Run a specific guardrail level (0-5)\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 0"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Level 1: String filtering\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 1"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Level 2: Input classifier\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 2"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Level 3: Output classifier\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 3"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Level 4: Thinking classifier\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 4"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Level 5 (Stretch): Linear probes\n",
                    "!cd /workspace/exercises && python3 poc_guardrails.py --level 5"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Exercise 3.3: Distillation Attack\n",
                    "Shows how forbidden knowledge leaks through soft labels:"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "!cd /workspace/exercises && python3 poc_distillation.py"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Iterate on exercises\n",
                    "\n",
                    "To edit and re-run code:\n",
                    "1. Open the `.py` file in Jupyter's file browser (left sidebar)\n",
                    "2. Edit the code\n",
                    "3. Re-run the cell above, or use the terminal:\n",
                    "   ```\n",
                    "   cd /workspace/exercises && python3 <script>.py\n",
                    "   ```"
                ]
            }
        ]
    }
    return json.dumps(notebook)


# ─────────────────────────────────────────────────────────────────────────────
# Pod operations
# ─────────────────────────────────────────────────────────────────────────────

def create_pod(name: str, gpu_type_id: str) -> dict:
    mutation = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        imageName
        machine { podHostId }
      }
    }
    """
    variables = {
        "input": {
            "cloudType": "ALL",
            "gpuCount": 1,
            "volumeInGb": VOLUME_GB,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": DOCKER_IMAGE,
            "ports": PORTS,
            "volumeMountPath": VOLUME_MOUNT,
            "env": [
                {"key": "JUPYTER_PASSWORD", "value": JUPYTER_PASSWORD},
                {"key": "JUPYTER_TOKEN", "value": JUPYTER_PASSWORD},
                {"key": "PUBLIC_KEY", "value": SSH_PUBLIC_KEY},
                {"key": "HF_HOME", "value": "/workspace/model-cache"},
                {"key": "TRANSFORMERS_CACHE", "value": "/workspace/model-cache"},
            ],
        }
    }
    data = graphql(mutation, variables)
    return data["podFindAndDeployOnDemand"]


def list_pods() -> list[dict]:
    query = """
    query {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime {
            uptimeInSeconds
            ports { ip isIpPublic privatePort publicPort type }
          }
        }
      }
    }
    """
    return graphql(query)["myself"]["pods"]


def stop_pod(pod_id: str) -> None:
    mutation = """
    mutation terminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    graphql(mutation, {"input": {"podId": pod_id}})
    print(f"  Terminated pod {pod_id}")


def get_pod(pod_id: str) -> dict:
    query = """
    query getPod($input: PodFilter!) {
      pod(input: $input) {
        id
        name
        desiredStatus
        runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
        }
      }
    }
    """
    return graphql(query, {"input": {"podId": pod_id}})["pod"]


def get_ssh_info(pod_id: str) -> tuple[str, int] | None:
    """Get (ip, port) for SSH from a pod's runtime info."""
    pod = get_pod(pod_id)
    if not pod or not pod.get("runtime"):
        return None
    for p in pod["runtime"].get("ports", []):
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], p["publicPort"]
    return None


def pod_exec(pod_id: str, command: str, timeout: int = 120) -> tuple[int, str, str]:
    """Execute a command on a pod via SSH. Returns (returncode, stdout, stderr)."""
    import subprocess
    info = get_ssh_info(pod_id)
    if not info:
        return -1, "", "Could not find SSH connection info"
    ip, port = info
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-o", "PasswordAuthentication=no",
         f"root@{ip}", "-p", str(port), command],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def wait_for_pod(pod_id: str, timeout: int = 300) -> dict:
    print(f"  Waiting for pod {pod_id}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        if pod and pod.get("runtime") and pod["runtime"].get("ports"):
            if any(p["privatePort"] == 8888 for p in pod["runtime"]["ports"]):
                print(" running!")
                return pod
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timeout!")
    return get_pod(pod_id)


def print_connection_info(pod: dict) -> None:
    pod_id = pod["id"]
    ports = pod.get("runtime", {}).get("ports", [])

    jupyter_url = f"https://{pod_id}-8888.proxy.runpod.net"
    print(f"\n  Pod: {pod['name']} ({pod_id})")
    print(f"  Status: {pod.get('desiredStatus', 'unknown')}")
    print(f"  Jupyter:  {jupyter_url}")
    print(f"  Password: {JUPYTER_PASSWORD}")
    print(f"  Terminal: https://www.runpod.io/console/pods/{pod_id}/terminal")

    for p in ports:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            print(f"  SSH:      ssh root@{p['ip']} -p {p['publicPort']}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy Day 3 bootcamp pods on RunPod")
    parser.add_argument("--count", type=int, default=1, help="Number of pods to create")
    parser.add_argument("--list", action="store_true", help="List all running pods")
    parser.add_argument("--stop-all", action="store_true", help="Stop all bootcamp pods")
    parser.add_argument("--stop", type=str, help="Stop a specific pod by ID")
    parser.add_argument("--gpu", type=str, default=GPU_TYPE_IDS[0], help="GPU type")
    parser.add_argument("--api-key", type=str, help="RunPod API key (or RUNPOD_API_KEY env)")
    parser.add_argument("--skip-upload", action="store_true", help="Skip file upload")
    args = parser.parse_args()

    global API_KEY
    if args.api_key:
        API_KEY = args.api_key
    if not API_KEY:
        print("Error: Set RUNPOD_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    # ── List ──
    if args.list:
        pods = list_pods()
        bootcamp_pods = [p for p in pods if p["name"].startswith(POD_NAME_PREFIX)]
        if not bootcamp_pods:
            print("No bootcamp pods found.")
            return
        print(f"\n{'='*70}")
        print(f"  Bootcamp Pods ({len(bootcamp_pods)})")
        print(f"{'='*70}")
        for pod in bootcamp_pods:
            print_connection_info(pod)
        return

    # ── Stop ──
    if args.stop:
        stop_pod(args.stop)
        return

    if args.stop_all:
        pods = list_pods()
        bootcamp_pods = [p for p in pods if p["name"].startswith(POD_NAME_PREFIX)]
        if not bootcamp_pods:
            print("No bootcamp pods to stop.")
            return
        for pod in bootcamp_pods:
            stop_pod(pod["id"])
        print(f"Stopped {len(bootcamp_pods)} pods.")
        return

    # ── Create ──
    print(f"\n{'='*70}")
    print(f"  Deploying {args.count} pod(s)")
    print(f"  Image: {DOCKER_IMAGE}")
    print(f"  Jupyter password: {JUPYTER_PASSWORD}")
    print(f"{'='*70}\n")

    created_pods = []
    for i in range(args.count):
        name = f"{POD_NAME_PREFIX}-{i+1:02d}" if args.count > 1 else POD_NAME_PREFIX
        print(f"  Creating pod '{name}'...")

        pod = None
        for gpu in ([args.gpu] if args.gpu != GPU_TYPE_IDS[0] else GPU_TYPE_IDS):
            try:
                pod = create_pod(name, gpu)
                print(f"    Got {gpu}: {pod['id']}")
                break
            except SystemExit:
                print(f"    {gpu} unavailable, trying next...", file=sys.stderr)
                continue

        if pod is None:
            print(f"    FAILED -- no GPU available", file=sys.stderr)
            continue
        created_pods.append(pod)

    if not created_pods:
        print("\nNo pods created.", file=sys.stderr)
        sys.exit(1)

    # Wait for pods, upload files, print info
    print(f"\n{'='*70}")
    print(f"  Waiting for {len(created_pods)} pod(s)...")
    print(f"{'='*70}")

    for pod in created_pods:
        ready_pod = wait_for_pod(pod["id"])
        print_connection_info(ready_pod)

        if not args.skip_upload:
            # Wait a bit for Jupyter to fully start
            print(f"\n  Waiting for Jupyter API on {pod['id']}...", end="", flush=True)
            if wait_for_jupyter(pod["id"], timeout=120):
                print(" ready!")
                upload_exercises_to_pod(pod["id"])
            else:
                print(" timed out. Upload manually via Jupyter web UI.")

    print(f"""
{'='*70}
  PARTICIPANT INSTRUCTIONS
{'='*70}

  1. Open Jupyter in your browser (URL above)
  2. Password: {JUPYTER_PASSWORD}
  3. Open exercises/quickstart.ipynb for a guided walkthrough
  4. Or open a terminal in Jupyter and run:
       cd /workspace/exercises
       pip install transformers accelerate scikit-learn tqdm
       python3 poc_guardrails.py --level 0

  To iterate: edit .py files in Jupyter, then re-run from terminal or notebook.

  FIRST TIME: run the setup to download models (~5 min):
    cd /workspace/exercises && bash setup_pod.sh
{'='*70}
""")


if __name__ == "__main__":
    main()
