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
    """启动OAuth认证流程（本地回调服务器模式）"""
    try:
        from src.auth import create_auth_url

        mode = request.mode or "geminicli"
        project_id = getattr(request, "project_id", None) or None

        result = await create_auth_url(project_id=project_id, mode=mode)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"[PANEL] 启动OAuth认证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/auth/callback")
async def auth_callback(
    request: Request,
    token: str = Depends(verify_panel_token),
):
    """处理OAuth回调 - 交换token并保存凭证（支持自动模式和手动URL模式）

    Expected JSON body fields:
      - mode: str (default "geminicli")
      - state: str | null  — OAuth state from /panel/auth/start
      - project_id: str | null — optional override
      - callback_url: str | null — if provided, use URL-based flow; otherwise wait for
        the local callback server to receive the code (auto mode).

    Using a raw Request instead of a typed model allows callback_url to be truly
    optional. A typed model with callback_url: str would cause a 422 validation error
    when the frontend omits the field in auto mode, which would surface as
    "[object Object]" in the UI because d.detail is an array, not a string.
    """
    try:
        body = await request.json()
        mode = body.get("mode", "geminicli")
        state = body.get("state") or None
        project_id = body.get("project_id") or None
        callback_url = body.get("callback_url", "")

        # If callback_url provided, use the URL-based flow
        if callback_url:
            from src.auth import complete_auth_flow_from_callback_url
            result = await complete_auth_flow_from_callback_url(
                callback_url=callback_url,
                project_id=project_id,
                mode=mode,
            )
        else:
            from src.auth import asyncio_complete_auth_flow
            user_session = token if token else None
            # 30 s timeout: the user lands on step-3 only after the poll confirms
            # the local callback server already received the code, so the wait is
            # normally instant.  Extra headroom covers slow token-exchange round-trips.
            result = await asyncio_complete_auth_flow(
                project_id=project_id,
                user_session=user_session,
                state=state,
                timeout=30,
                mode=mode,
            )

        if result.get("requires_project_selection"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": result.get("error", "需要选择项目"),
                    "requires_project_selection": True,
                    "available_projects": result.get("available_projects", []),
                },
            )
        if result.get("requires_manual_project_id"):
            return JSONResponse(
                status_code=400,
                content={"error": result.get("error", "需要手动输入项目ID"), "requires_manual_project_id": True},
            )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "认证失败"))

        return JSONResponse({
            "credentials": result.get("credentials", {}),
            "file_path": result.get("file_path", ""),
            "message": "认证成功，凭证已保存",
            "auto_detected_project": result.get("auto_detected_project", False),
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] OAuth回调处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/auth/status/{state}")
async def auth_status(state: str, token: str = Depends(verify_panel_token)):
    """查询认证流程状态"""
    from src.auth import get_auth_status
    return JSONResponse(get_auth_status(state))


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


# ====================== Additional Credential Endpoints ======================

@router.get("/panel/credentials/status")
async def get_credentials_status_paginated(
    token: str = Depends(verify_panel_token),
    offset: int = 0,
    limit: int = 50,
    status_filter: str = "all",
    error_code_filter: str = "all",
    cooldown_filter: str = "all",
    preview_filter: str = "all",
):
    """分页获取凭证状态列表，支持多维度过滤"""
    try:
        cred_manager = await credential_manager._get_or_create()
        all_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")
        items = []
        now = datetime.now(timezone.utc).timestamp()

        for cred_name in all_creds:
            try:
                cred_data = await cred_manager._storage_adapter.get_credential(cred_name, mode="geminicli") or {}
                cred_state = await cred_manager._storage_adapter.get_credential_state(cred_name, mode="geminicli") or {}
                disabled = cred_state.get("disabled", cred_data.get("disabled", False))
                error_codes = cred_state.get("error_codes", [])
                cooldown_until = cred_state.get("cooldown_until")
                in_cooldown = bool(cooldown_until and cooldown_until > now)
                preview = cred_data.get("preview", False)

                # Apply filters
                if status_filter == "enabled" and disabled:
                    continue
                if status_filter == "disabled" and not disabled:
                    continue
                if error_code_filter != "all" and error_code_filter not in [str(c) for c in error_codes]:
                    continue
                if cooldown_filter == "in_cooldown" and not in_cooldown:
                    continue
                if cooldown_filter == "no_cooldown" and in_cooldown:
                    continue
                if preview_filter == "preview" and not preview:
                    continue
                if preview_filter == "no_preview" and preview:
                    continue

                items.append({
                    "filename": cred_name,
                    "project_id": cred_data.get("project_id", ""),
                    "email": cred_data.get("email", ""),
                    "disabled": disabled,
                    "error_count": cred_state.get("error_count", 0),
                    "error_codes": error_codes,
                    "last_success": cred_state.get("last_success"),
                    "cooldown_until": cooldown_until,
                    "in_cooldown": in_cooldown,
                    "preview": preview,
                })
            except Exception as e:
                items.append({"filename": cred_name, "error": str(e)})

        total = len(items)
        page_items = items[offset: offset + limit]
        enabled = sum(1 for i in items if not i.get("disabled", False))
        disabled_count = total - enabled

        return JSONResponse({
            "items": page_items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total,
            "stats": {"total": total, "enabled": enabled, "disabled": disabled_count},
        })
    except Exception as e:
        log.error(f"[PANEL] 获取凭证状态分页失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/credentials/{filename}/detail")
async def get_credential_detail(filename: str, token: str = Depends(verify_panel_token)):
    """获取凭证详情及状态"""
    try:
        cred_manager = await credential_manager._get_or_create()
        cred_data = await cred_manager._storage_adapter.get_credential(filename, mode="geminicli")
        if cred_data is None:
            raise HTTPException(status_code=404, detail=f"凭证 {filename} 不存在")
        cred_state = await cred_manager._storage_adapter.get_credential_state(filename, mode="geminicli") or {}
        return JSONResponse({"filename": filename, "data": cred_data, "state": cred_state})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] 获取凭证详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/credentials/download-all")
async def download_all_credentials(token: str = Depends(verify_panel_token)):
    """将所有凭证打包为 ZIP 下载"""
    try:
        cred_manager = await credential_manager._get_or_create()
        all_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for cred_name in all_creds:
                try:
                    cred_data = await cred_manager._storage_adapter.get_credential(cred_name, mode="geminicli")
                    if cred_data:
                        zf.writestr(cred_name, json.dumps(cred_data, ensure_ascii=False, indent=2))
                except Exception:
                    pass
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="credentials.zip"'},
        )
    except Exception as e:
        log.error(f"[PANEL] 下载全部凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/{filename}/verify-project")
async def verify_credential_project(filename: str, token: str = Depends(verify_panel_token)):
    """重新获取凭证的 project_id，并清除错误状态"""
    try:
        from src.google_oauth_api import fetch_project_id
        from src.utils import GEMINICLI_USER_AGENT
        from config import get_code_assist_endpoint

        cred_manager = await credential_manager._get_or_create()
        cred_data = await cred_manager._storage_adapter.get_credential(filename, mode="geminicli")
        if cred_data is None:
            raise HTTPException(status_code=404, detail=f"凭证 {filename} 不存在")

        # Refresh access token if needed
        from src.google_oauth_api import Credentials as GCreds
        creds = GCreds(
            access_token=cred_data.get("token") or cred_data.get("access_token", ""),
            refresh_token=cred_data.get("refresh_token", ""),
            client_id=cred_data.get("client_id", ""),
            client_secret=cred_data.get("client_secret", ""),
        )
        await creds.refresh_if_needed()

        api_base_url = await get_code_assist_endpoint()
        project_id = await fetch_project_id(
            access_token=creds.access_token,
            user_agent=GEMINICLI_USER_AGENT,
            api_base_url=api_base_url,
        )

        # Update credential data
        cred_data["project_id"] = project_id
        cred_data["token"] = creds.access_token
        cred_data["access_token"] = creds.access_token
        await cred_manager._storage_adapter.store_credential(filename, cred_data, mode="geminicli")

        # Reset state
        await cred_manager._storage_adapter.update_credential_state(
            filename,
            {"disabled": False, "error_codes": [], "error_count": 0, "cooldown_until": None},
            mode="geminicli",
        )

        log.info(f"[PANEL] 凭证 {filename} project_id 已更新: {project_id}")
        return JSONResponse({"success": True, "filename": filename, "project_id": project_id, "message": "project_id 已更新，错误状态已清除"})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] verify-project 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/{filename}/configure-preview")
async def configure_credential_preview(filename: str, request: Request, token: str = Depends(verify_panel_token)):
    """切换凭证的 preview 标志"""
    try:
        body = await request.json()
        preview = bool(body.get("preview", False))

        cred_manager = await credential_manager._get_or_create()
        cred_data = await cred_manager._storage_adapter.get_credential(filename, mode="geminicli")
        if cred_data is None:
            raise HTTPException(status_code=404, detail=f"凭证 {filename} 不存在")

        cred_data["preview"] = preview
        await cred_manager._storage_adapter.store_credential(filename, cred_data, mode="geminicli")

        log.info(f"[PANEL] 凭证 {filename} preview={preview}")
        return JSONResponse({"success": True, "filename": filename, "preview": preview, "message": f"preview 已设置为 {preview}"})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] configure-preview 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/refresh-all-emails")
async def refresh_all_emails(token: str = Depends(verify_panel_token)):
    """刷新所有缺少邮箱的凭证的 email 字段"""
    try:
        cred_manager = await credential_manager._get_or_create()
        all_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")
        results = []
        for cred_name in all_creds:
            try:
                cred_data = await cred_manager._storage_adapter.get_credential(cred_name, mode="geminicli") or {}
                if cred_data.get("email"):
                    results.append({"filename": cred_name, "skipped": True, "email": cred_data["email"]})
                    continue
                email = await cred_manager.get_or_fetch_user_email(cred_name, mode="geminicli")
                results.append({"filename": cred_name, "success": True, "email": email or ""})
            except Exception as ex:
                results.append({"filename": cred_name, "success": False, "error": str(ex)})
        return JSONResponse({"results": results})
    except Exception as e:
        log.error(f"[PANEL] refresh-all-emails 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/deduplicate-by-email")
async def deduplicate_credentials_by_email(token: str = Depends(verify_panel_token)):
    """按邮箱去重凭证（相同邮箱保留最新一个）"""
    try:
        cred_manager = await credential_manager._get_or_create()
        all_creds = await cred_manager._storage_adapter.list_credentials(mode="geminicli")
        email_map: Dict[str, List[str]] = {}
        for cred_name in all_creds:
            try:
                cred_data = await cred_manager._storage_adapter.get_credential(cred_name, mode="geminicli") or {}
                email = cred_data.get("email", "")
                if email:
                    email_map.setdefault(email, []).append(cred_name)
            except Exception:
                pass

        removed = []
        for email, names in email_map.items():
            if len(names) <= 1:
                continue
            # Keep the last (most recent) file, remove the rest
            to_remove = sorted(names)[:-1]
            for fname in to_remove:
                try:
                    await cred_manager.remove_credential(fname, mode="geminicli")
                    removed.append(fname)
                except Exception as ex:
                    log.warning(f"[PANEL] deduplicate: 删除 {fname} 失败: {ex}")

        log.info(f"[PANEL] 去重完成，删除 {len(removed)} 个重复凭证")
        return JSONResponse({"success": True, "removed": removed, "count": len(removed)})
    except Exception as e:
        log.error(f"[PANEL] deduplicate-by-email 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/panel/credentials/test/{filename}")
async def test_credential(filename: str, token: str = Depends(verify_panel_token)):
    """测试凭证：发送一次简单请求验证可用性"""
    try:
        from src.google_oauth_api import Credentials as GCreds
        from src.api.geminicli import prepare_request_headers_and_payload
        from src.httpx_client import post_async
        from config import get_code_assist_endpoint

        cred_manager = await credential_manager._get_or_create()
        cred_data = await cred_manager._storage_adapter.get_credential(filename, mode="geminicli")
        if cred_data is None:
            raise HTTPException(status_code=404, detail=f"凭证 {filename} 不存在")

        creds = GCreds(
            access_token=cred_data.get("token") or cred_data.get("access_token", ""),
            refresh_token=cred_data.get("refresh_token", ""),
            client_id=cred_data.get("client_id", ""),
            client_secret=cred_data.get("client_secret", ""),
        )
        await creds.refresh_if_needed()

        api_base_url = await get_code_assist_endpoint()
        test_body = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        cred_dict = {**cred_data, "token": creds.access_token, "access_token": creds.access_token}
        auth_headers, payload, target_url = await prepare_request_headers_and_payload(
            test_body, cred_dict, f"{api_base_url}/v1internal:generateContent"
        )

        resp = await post_async(target_url, headers=auth_headers, json=payload, timeout=15)
        if resp.status_code != 200:
            return JSONResponse({
                "success": False,
                "filename": filename,
                "message": f"HTTP {resp.status_code}: {resp.text[:200]}",
            })

        resp_json = resp.json()
        inner = resp_json.get("response", resp_json)
        model_version = inner.get("modelVersion", "")
        response_text = ""
        for cand in inner.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                response_text += part.get("text", "")

        return JSONResponse({
            "success": True,
            "filename": filename,
            "message": "凭证测试成功",
            "model_version": model_version,
            "response_preview": response_text[:200],
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[PANEL] test credential 失败: {e}")
        return JSONResponse({"success": False, "filename": filename, "message": str(e)})


# ====================== Version Info ======================

@router.get("/panel/version")
async def get_version(
    check_update: bool = False,
    token: str = Depends(verify_panel_token),
):
    """获取版本信息"""
    try:
        version = "unknown"
        version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")
        if os.path.exists(version_file):
            with open(version_file, "r", encoding="utf-8") as f:
                version = f.read().strip()

        result: Dict[str, Any] = {"version": version}

        if check_update:
            try:
                from src.httpx_client import get_async
                resp = await get_async(
                    "https://api.github.com/repos/su-kaka/gcli2api/releases/latest",
                    headers={"User-Agent": "gcli2api-panel"},
                    timeout=5,
                )
                if resp and resp.status_code == 200:
                    latest = resp.json().get("tag_name", "")
                    result["latest_version"] = latest
                    result["update_available"] = latest != version and bool(latest)
            except Exception:
                pass

        return JSONResponse(result)
    except Exception as e:
        log.error(f"[PANEL] 获取版本信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panel/config")
async def get_config(token: str = Depends(verify_panel_token)):
    """获取当前配置"""
    try:
        config_items = []
        env_locked = []
        for env_key, db_key in ENV_MAPPINGS.items():
            env_val = os.environ.get(env_key)
            db_val = await get_config_value(db_key, None)
            is_locked = env_val is not None
            if is_locked:
                env_locked.append(db_key)
            config_items.append({
                "key": db_key,
                "value": env_val if env_val is not None else (db_val if db_val is not None else ""),
                "env_locked": is_locked,
                "env_var": env_key,
            })
        return JSONResponse({"config": config_items, "env_locked": env_locked})
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
        _keepalive_keys = {"keepalive_url", "keepalive_interval"}
        keepalive_changed = any(k in _keepalive_keys for k in request.config)

        for key, value in request.config.items():
            adapter = await get_storage_adapter()
            await adapter.set_config(key, value)

        # 重载配置缓存
        await reload_config()

        # 如果 keepalive 相关配置变更，重启 keepalive 服务
        if keepalive_changed:
            try:
                from src.keeplive import keepalive_service
                await keepalive_service.restart()
            except Exception as ke:
                log.warning(f"[PANEL] keepalive restart failed: {ke}")

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


@router.get("/mobile", response_class=HTMLResponse)
async def panel_mobile():
    """移动端控制面板"""
    return RedirectResponse(url="/front/control_panel_mobile.html")
