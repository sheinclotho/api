"""
Panel Router - Web control panel for credential management and system status
提供凭证管理、系统状态查看和OAuth认证的控制面板路由
"""

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from config import (
    get_credentials_dir,
    get_panel_password,
    reload_config,
    ENV_MAPPINGS,
    get_config_value,
)
from log import log, get_recent_logs
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

# OOB redirect URI used for the OAuth "copy-paste" flow
_OOB_REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob"

# Active WebSocket connections for log streaming
_log_ws_clients: List[WebSocket] = []


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

        # 统计启用/禁用数量
        enabled_count = 0
        disabled_count = 0
        for cred_name in geminicli_creds:
            try:
                state = await cred_manager._storage_adapter.get_credential_state(cred_name, mode="geminicli") or {}
                if state.get("disabled", False):
                    disabled_count += 1
                else:
                    enabled_count += 1
            except Exception:
                enabled_count += 1

        # 获取存储后端信息
        backend_info = {}
        try:
            backend_info = await cred_manager._storage_adapter.get_backend_info()
        except Exception:
            pass

        return JSONResponse({
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "credentials": {
                "geminicli": len(geminicli_creds),
                "enabled": enabled_count,
                "disabled": disabled_count,
            },
            "backend": backend_info,
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
            redirect_uri=_OOB_REDIRECT_URI,
        )

        state = f"panel_{mode}_{datetime.now(timezone.utc).timestamp()}"
        auth_url = auth.get_auth_url(state=state)

        return JSONResponse({
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": _OOB_REDIRECT_URI,
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


@router.post("/panel/auth/exchange")
async def auth_exchange(
    request: Request,
    token: str = Depends(verify_panel_token),
):
    """用授权码直接换取凭证（OOB流程）"""
    try:
        from src.google_oauth_api import Flow
        from src.utils import CLIENT_ID, CLIENT_SECRET, SCOPES

        body = await request.json()
        code = body.get("code", "").strip()
        mode = body.get("mode", "geminicli")
        project_id = body.get("project_id", "")

        if not code:
            raise HTTPException(status_code=400, detail="未提供授权码")

        auth = Flow(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
            redirect_uri=_OOB_REDIRECT_URI,
        )

        credentials = await auth.exchange_code(code=code)
        if not credentials:
            raise HTTPException(status_code=400, detail="授权码兑换失败")

        if not project_id:
            project_id = credentials.project_id or ""

        email = ""
        try:
            from src.google_oauth_api import get_user_email
            email = await get_user_email(credentials) or ""
        except Exception:
            pass

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
        log.info(f"[PANEL] 新凭证已保存（OOB）: {filename} (mode={mode})")

        return JSONResponse({
            "success": True,
            "filename": filename,
            "email": email,
            "project_id": project_id,
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] OOB授权码兑换失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Credential Upload / Download ======================

@router.post("/panel/credentials/upload")
async def upload_credentials(
    files: List[UploadFile] = File(...),
    token: str = Depends(verify_panel_token),
):
    """上传凭证文件（支持JSON单文件或ZIP批量上传）"""
    try:
        from src.utils import CLIENT_ID, CLIENT_SECRET

        cred_manager = await credential_manager._get_or_create()
        results = []

        for upload in files:
            raw = await upload.read()
            fname = upload.filename or "credential.json"

            # ZIP批量上传
            if fname.lower().endswith(".zip") or upload.content_type in ("application/zip", "application/x-zip-compressed"):
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        for name in zf.namelist():
                            if not name.lower().endswith(".json"):
                                continue
                            try:
                                cred_bytes = zf.read(name)
                                cred_data = json.loads(cred_bytes.decode("utf-8"))
                                base_name = os.path.basename(name)
                                # 补全缺失的 client_id/client_secret
                                if not cred_data.get("client_id"):
                                    cred_data["client_id"] = CLIENT_ID
                                if not cred_data.get("client_secret"):
                                    cred_data["client_secret"] = CLIENT_SECRET
                                await cred_manager._storage_adapter.store_credential(base_name, cred_data, mode="geminicli")
                                results.append({"filename": base_name, "success": True})
                            except Exception as ex:
                                results.append({"filename": name, "success": False, "error": str(ex)})
                except Exception as ex:
                    results.append({"filename": fname, "success": False, "error": f"ZIP解析失败: {ex}"})
            else:
                # 单个JSON文件
                try:
                    cred_data = json.loads(raw.decode("utf-8"))
                    if not cred_data.get("client_id"):
                        cred_data["client_id"] = CLIENT_ID
                    if not cred_data.get("client_secret"):
                        cred_data["client_secret"] = CLIENT_SECRET
                    await cred_manager._storage_adapter.store_credential(fname, cred_data, mode="geminicli")
                    results.append({"filename": fname, "success": True})
                except Exception as ex:
                    results.append({"filename": fname, "success": False, "error": str(ex)})

        log.info(f"[PANEL] 上传凭证: {len([r for r in results if r['success']])} 成功, {len([r for r in results if not r['success']])} 失败")
        return JSONResponse({"results": results})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 上传凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/credentials/{filename}/download")
async def download_credential(
    filename: str,
    token: str = Depends(verify_panel_token),
):
    """下载凭证文件"""
    try:
        cred_manager = await credential_manager._get_or_create()
        cred_data = await cred_manager._storage_adapter.get_credential(filename, mode="geminicli")
        if cred_data is None:
            raise HTTPException(status_code=404, detail=f"凭证 {filename} 不存在")

        content = json.dumps(cred_data, ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 下载凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== Credential Email ======================

@router.post("/panel/credentials/email")
async def get_credential_email(
    request: Request,
    token: str = Depends(verify_panel_token),
):
    """获取单个凭证的邮箱"""
    try:
        body = await request.json()
        filename = body.get("filename", "")
        if not filename:
            raise HTTPException(status_code=400, detail="未提供 filename")

        cred_manager = await credential_manager._get_or_create()
        email = await cred_manager.get_or_fetch_user_email(filename, mode="geminicli")
        return JSONResponse({"filename": filename, "email": email or ""})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 获取邮箱失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/email/batch")
async def batch_get_credential_emails(
    request: Request,
    token: str = Depends(verify_panel_token),
):
    """批量获取凭证邮箱"""
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        cred_manager = await credential_manager._get_or_create()
        results = []
        for fname in filenames:
            try:
                email = await cred_manager.get_or_fetch_user_email(fname, mode="geminicli")
                results.append({"filename": fname, "email": email or "", "success": True})
            except Exception as ex:
                results.append({"filename": fname, "email": "", "success": False, "error": str(ex)})
        return JSONResponse({"results": results})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 批量获取邮箱失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        logs = get_recent_logs(lines)
        return JSONResponse({"logs": logs})
    except Exception as e:
        log.error(f"[PANEL] 获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/panel/logs")
async def clear_logs(token: str = Depends(verify_panel_token)):
    """清空日志文件和内存缓冲区"""
    try:
        from log import _log_buffer, _log_buffer_lock
        import threading
        with _log_buffer_lock:
            _log_buffer.clear()
        log_file = os.getenv("LOG_FILE", "log.txt")
        if os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                pass
        log.info("[PANEL] 日志已清空")
        return JSONResponse({"success": True, "message": "日志已清空"})
    except Exception as e:
        log.error(f"[PANEL] 清空日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/logs/download")
async def download_logs(token: str = Depends(verify_panel_token)):
    """下载日志文件"""
    try:
        log_file = os.getenv("LOG_FILE", "log.txt")
        if os.path.exists(log_file):
            return FileResponse(log_file, media_type="text/plain", filename="gcli2api.log")
        # 如果文件不存在，返回内存中的日志
        content = "\n".join(get_recent_logs(1000))
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="gcli2api.log"'},
        )
    except Exception as e:
        log.error(f"[PANEL] 下载日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/panel/ws/logs")
async def ws_logs(websocket: WebSocket, token: str = ""):
    """WebSocket 实时日志流"""
    import asyncio

    # 验证 token（从 query 参数获取）
    password = await get_panel_password()
    if token != password:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _log_ws_clients.append(websocket)

    # 发送当前缓存日志
    current_logs = get_recent_logs(200)
    if current_logs:
        try:
            await websocket.send_json({"type": "history", "logs": current_logs})
        except Exception:
            pass

    # 实时推送新日志（轮询内存缓冲区）
    last_count = len(get_recent_logs(10000))
    try:
        while True:
            await asyncio.sleep(1)
            all_logs = get_recent_logs(10000)
            current_count = len(all_logs)
            if current_count > last_count:
                new_entries = all_logs[last_count:]
                await websocket.send_json({"type": "new", "logs": new_entries})
                last_count = current_count
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _log_ws_clients:
            _log_ws_clients.remove(websocket)


# ====================== Web Panel ======================

@router.get("/", response_class=HTMLResponse)
async def panel_index():
    """控制面板首页 - 重定向到前端界面"""
    return RedirectResponse(url="/front/index.html")
