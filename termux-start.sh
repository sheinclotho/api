#!/bin/bash
echo "强制同步项目代码，忽略本地修改..."
git fetch --all
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)
uv add -r requirements-termux.txt
source .venv/bin/activate
pm2 restart web 2>/dev/null || pm2 start .venv/bin/python --name web -- web.py
pm2 save
echo "✅ 服务已通过 pm2 启动，运行 'pm2 logs web' 查看日志"
