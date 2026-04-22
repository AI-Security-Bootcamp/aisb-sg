#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

uv init .

uv venv

uv pip install -r requirements.txt