#!/bin/bash
set -e

echo "强制同步项目代码，忽略本地修改..."
git fetch --all
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)

echo "同步依赖..."
uv sync

echo "重置面板密码和API密码为默认值 (pwd)..."
source .venv/bin/activate
python - << 'PYEOF'
import asyncio, sys
sys.path.insert(0, '.')
async def reset_passwords():
    try:
        from src.storage_adapter import get_storage_adapter
        adapter = await get_storage_adapter()
        await adapter.delete_config('api_password')
        await adapter.delete_config('panel_password')
        print('✅ 密码已重置为默认值 pwd')
    except Exception as e:
        print(f'密码重置失败 (非致命): {e}')
asyncio.run(reset_passwords())
PYEOF

echo "启动服务..."
python web.py
