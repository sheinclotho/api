#!/bin/bash
set -e

echo "强制同步项目代码，忽略本地修改..."
git fetch --all
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)

echo "同步依赖..."
uv sync

echo "激活虚拟环境并启动服务..."
source .venv/bin/activate
python web.py
