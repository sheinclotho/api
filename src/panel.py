"""
Panel Router - Web control panel for credential management and system status
提供凭证管理、系统状态查看和OAuth认证的控制面板路由
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import (
    get_credentials_dir,
    get_panel_password,
    reload_config,
    ENV_MAPPINGS,
    get_config_value,
)
from log import log
from src.credential_manager import credential_manager
from src.models import (
    AuthCallbackUrlRequest,
    AuthStartRequest,
    ConfigSaveRequest,
    CredFileActionRequest,
    CredFileBatchActionRequest,
    LoginRequest,
)
from src.storage_adapter import get_storage_adapter
from src.utils import verify_panel_token

router = APIRouter()

# ====================== Login ======================

@router.post("/panel/login")
async def panel_login(request: LoginRequest):
    """控制面板登录"""
    password = await get_panel_password()
    if request.password != password:
        raise HTTPException(status_code=401, detail="密码错误")
    return {"success": True, "token": request.password}


# ====================== System Status ======================

@router.get("/panel/status")
async def panel_status(token: str = Depends(verify_panel_token)):
    """获取系统状态"""
    try:
        cred_manager = await credential_manager._get_or_create()
        geminicli_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")

        return JSONResponse({
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "credentials": {
                "geminicli": len(geminicli_creds),
            },
        })
    except Exception as e:
        log.error(f"[PANEL] 获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Credential Management ======================

@router.get("/panel/credentials")
async def list_credentials(token: str = Depends(verify_panel_token)):
    """列出所有凭证"""
    try:
        cred_manager = await credential_manager._get_or_create()
        all_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")
        result = []
        for cred_name in all_creds:
            try:
                cred_data = await cred_manager._storage_adapter.get_credential(
                    cred_name, mode="geminicli"
                ) or {}
                cred_state = await cred_manager._storage_adapter.get_credential_state(
                    cred_name, mode="geminicli"
                ) or {}
                status_data = {
                    "disabled": cred_state.get("disabled", cred_data.get("disabled", False)),
                    "error_count": cred_state.get("error_count", cred_data.get("error_count", 0)),
                    "last_success": cred_state.get("last_success", cred_data.get("last_success")),
                    "project_id": cred_data.get("project_id", ""),
                    "preview": cred_data.get("preview", False),
                    "cooldown_until": cred_state.get("cooldown_until"),
                }
                result.append({
                    "filename": cred_name,
                    "project_id": cred_data.get("project_id", ""),
                    "status": status_data,
                })
            except Exception as e:
                result.append({
                    "filename": cred_name,
                    "error": str(e),
                    "status": {},
                })
        return JSONResponse({"credentials": result})
    except Exception as e:
        log.error(f"[PANEL] 列出凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/action")
async def credential_action(
    request: CredFileActionRequest,
    token: str = Depends(verify_panel_token),
):
    """对凭证执行操作（启用/禁用/删除）"""
    try:
        cred_manager = await credential_manager._get_or_create()
        action = request.action.lower()
        filename = request.filename

        if action == "enable":
            await cred_manager.set_cred_disabled(filename, False, mode="geminicli")
            return JSONResponse({"success": True, "message": f"凭证 {filename} 已启用"})
        elif action == "disable":
            await cred_manager.set_cred_disabled(filename, True, mode="geminicli")
            return JSONResponse({"success": True, "message": f"凭证 {filename} 已禁用"})
        elif action == "delete":
            await cred_manager.remove_credential(filename, mode="geminicli")
            return JSONResponse({"success": True, "message": f"凭证 {filename} 已删除"})
        else:
            raise HTTPException(status_code=400, detail=f"未知操作: {action}")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 凭证操作失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/batch")
async def batch_credential_action(
    request: CredFileBatchActionRequest,
    token: str = Depends(verify_panel_token),
):
    """批量对凭证执行操作"""
    try:
        cred_manager = await credential_manager._get_or_create()
        action = request.action.lower()
        results = []

        for filename in request.filenames:
            try:
                if action == "enable":
                    await cred_manager.set_cred_disabled(filename, False, mode="geminicli")
                    results.append({"filename": filename, "success": True})
                elif action == "disable":
                    await cred_manager.set_cred_disabled(filename, True, mode="geminicli")
                    results.append({"filename": filename, "success": True})
                elif action == "delete":
                    await cred_manager.remove_credential(filename, mode="geminicli")
                    results.append({"filename": filename, "success": True})
                else:
                    results.append({"filename": filename, "success": False, "error": f"未知操作: {action}"})
            except Exception as e:
                results.append({"filename": filename, "success": False, "error": str(e)})

        return JSONResponse({"results": results})
    except Exception as e:
        log.error(f"[PANEL] 批量凭证操作失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== OAuth Authentication ======================

@router.post("/panel/auth/start")
async def auth_start(
    request: AuthStartRequest,
    token: str = Depends(verify_panel_token),
):
    """启动OAuth认证流程"""
    try:
        from src.google_oauth_api import Flow
        from src.utils import CLIENT_ID, CLIENT_SECRET, SCOPES

        mode = request.mode or "geminicli"

        auth = Flow(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
        )

        state = f"panel_{mode}_{datetime.now(timezone.utc).timestamp()}"
        auth_url = auth.get_auth_url(state=state)

        return JSONResponse({
            "success": True,
            "auth_url": auth_url,
            "state": state,
        })
    except Exception as e:
        log.error(f"[PANEL] 启动OAuth认证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/auth/callback")
async def auth_callback(
    request: AuthCallbackUrlRequest,
    token: str = Depends(verify_panel_token),
):
    """处理OAuth回调"""
    try:
        from urllib.parse import parse_qs, urlparse

        from src.google_oauth_api import Flow
        from src.utils import CLIENT_ID, CLIENT_SECRET, SCOPES

        mode = request.mode or "geminicli"
        callback_url = request.callback_url

        # 从回调URL提取code
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]

        if not code:
            raise HTTPException(status_code=400, detail="回调URL中没有找到授权码")

        auth = Flow(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
        )

        # 交换token
        credentials = await auth.exchange_code(code=code)
        if not credentials:
            raise HTTPException(status_code=400, detail="交换授权码失败")

        # 获取项目ID
        project_id = request.project_id or credentials.project_id or ""

        # 获取用户邮箱
        email = ""
        try:
            from src.google_oauth_api import get_user_email
            email = await get_user_email(credentials) or ""
        except Exception:
            pass

        # 保存凭证
        cred_manager = await credential_manager._get_or_create()
        cred_data = {
            "token": credentials.access_token,
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "project_id": project_id,
            "email": email,
            "mode": mode,
        }

        filename = f"{email or 'credential'}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        await cred_manager._storage_adapter.store_credential(filename, cred_data, mode=mode)

        log.info(f"[PANEL] 新凭证已保存: {filename} (mode={mode})")

        return JSONResponse({
            "success": True,
            "filename": filename,
            "email": email,
            "project_id": project_id,
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] OAuth回调处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Configuration ======================

@router.get("/panel/config")
async def get_config(token: str = Depends(verify_panel_token)):
    """获取当前配置"""
    try:
        config_items = []
        for env_key, db_key in ENV_MAPPINGS.items():
            env_val = os.environ.get(env_key)
            db_val = await get_config_value(db_key, None)
            config_items.append({
                "key": db_key,
                "value": env_val if env_val is not None else (db_val if db_val is not None else ""),
                "env_locked": env_val is not None,
                "env_var": env_key,
            })
        return JSONResponse({"config": config_items})
    except Exception as e:
        log.error(f"[PANEL] 获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/config/save")
async def save_config(
    request: ConfigSaveRequest,
    token: str = Depends(verify_panel_token),
):
    """保存配置"""
    try:
        for key, value in request.config.items():
            adapter = await get_storage_adapter()
            await adapter.set_config(key, value)

        # 重载配置缓存
        await reload_config()

        return JSONResponse({"success": True, "message": "配置已保存"})
    except Exception as e:
        log.error(f"[PANEL] 保存配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Logs ======================

@router.get("/panel/logs")
async def get_logs(
    lines: int = 100,
    token: str = Depends(verify_panel_token),
):
    """获取最近的日志"""
    try:
        from log import get_recent_logs
        logs = get_recent_logs(lines)
        return JSONResponse({"logs": logs})
    except AttributeError:
        # log module may not have get_recent_logs
        return JSONResponse({"logs": [], "message": "日志获取功能不可用"})
    except Exception as e:
        log.error(f"[PANEL] 获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Simple HTML panel ======================

@router.get("/", response_class=HTMLResponse)
async def panel_index():
    """控制面板首页"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCLI2API 控制面板</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .badge-green { background: #d4edda; color: #155724; }
        .badge-blue { background: #cce5ff; color: #004085; }
    </style>
</head>
<body>
    <h1>GCLI2API 控制面板</h1>
    <p>API 代理服务运行中</p>
    <h2>可用端点</h2>
    <div class="endpoint">
        <span class="badge badge-green">GeminiCLI</span>
        <code>/v1/chat/completions</code> - OpenAI格式
    </div>
    <div class="endpoint">
        <span class="badge badge-green">GeminiCLI</span>
        <code>/v1/messages</code> - Claude格式
    </div>
    <div class="endpoint">
        <span class="badge badge-green">GeminiCLI</span>
        <code>/v1beta/models/{model}:generateContent</code> - Gemini格式
    </div>
    <div class="endpoint">
        <span class="badge badge-blue">Vertex AI</span>
        <code>/vertex/v1/chat/completions</code> - OpenAI格式
    </div>
    <div class="endpoint">
        <span class="badge badge-blue">Vertex AI</span>
        <code>/vertex/v1/messages</code> - Claude格式
    </div>
    <div class="endpoint">
        <span class="badge badge-blue">Vertex AI</span>
        <code>/vertex/v1/models/{model}:generateContent</code> - Gemini格式
    </div>
    <h2>管理API</h2>
    <p><a href="/docs">API文档</a> | <a href="/panel/status">系统状态</a></p>
</body>
</html>"""
    return HTMLResponse(content=html)
