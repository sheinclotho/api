"""
认证模块 - 支持本地 HTTP 回调服务器的 OAuth 自动认证流程
"""

import asyncio
import json
import socket
import threading
import time
import uuid
from datetime import timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from config import get_code_assist_endpoint, get_config_value
from log import log

from .google_oauth_api import (
    Credentials,
    Flow,
    enable_required_apis,
    fetch_project_id,
    get_user_projects,
)
from .storage_adapter import get_storage_adapter
from .utils import (
    CALLBACK_HOST,
    CLIENT_ID,
    CLIENT_SECRET,
    SCOPES,
    TOKEN_URL,
)


async def get_callback_port() -> int:
    """获取 OAuth 回调端口"""
    return int(await get_config_value("oauth_callback_port", "11451", "OAUTH_CALLBACK_PORT"))


def _prepare_credentials_data(credentials: Credentials, project_id: str) -> Dict[str, Any]:
    """准备凭证数据字典"""
    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": credentials.access_token,
        "refresh_token": credentials.refresh_token,
        "scopes": SCOPES,
        "token_uri": TOKEN_URL,
        "project_id": project_id,
    }
    if credentials.expires_at:
        expiry_utc = (
            credentials.expires_at.replace(tzinfo=timezone.utc)
            if credentials.expires_at.tzinfo is None
            else credentials.expires_at
        )
        creds_data["expiry"] = expiry_utc.isoformat()
    return creds_data


# ---------------------------------------------------------------------------
# 全局认证流程状态
# ---------------------------------------------------------------------------
auth_flows: Dict[str, Any] = {}
MAX_AUTH_FLOWS = 20


def cleanup_expired_flows():
    """清理超过 30 分钟的过期认证流程"""
    now = time.time()
    expired = [s for s, d in auth_flows.items() if now - d.get("created_at", now) > 1800]
    for state in expired:
        _cleanup_auth_flow_server(state)


def _cleanup_auth_flow_server(state: str):
    """清理认证流程的服务器资源"""
    if state not in auth_flows:
        return
    flow_data = auth_flows.pop(state)
    server = flow_data.get("server")
    if server:
        try:
            server.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP 回调处理器
# ---------------------------------------------------------------------------

class AuthCallbackHandler(BaseHTTPRequestHandler):
    """OAuth 回调处理器"""

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        code = query.get("code", [None])[0]
        state = query.get("state", [None])[0]

        if code and state and state in auth_flows:
            auth_flows[state]["code"] = code
            auth_flows[state]["completed"] = True
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                "<h1>✅ 授权成功！</h1>"
                "<p>已成功获取授权码，请返回控制面板点击「获取凭证」按钮完成授权。</p>"
                "<p>Authorization successful! Please return to the control panel and click 'Get Credentials'.</p>"
                "<script>setTimeout(()=>window.close(),3000)</script>"
                .encode("utf-8")
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1>")

    def log_message(self, format, *args):
        pass  # 静默日志


def _find_available_port_sync(start_port: int) -> int:
    """同步查找可用端口"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def find_available_port(start_port: int = None) -> int:
    """异步查找可用端口（在线程池中运行）"""
    if start_port is None:
        start_port = await get_callback_port()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_available_port_sync, start_port)


def _create_callback_server(port: int) -> HTTPServer:
    """创建并启动 OAuth 回调 HTTP 服务器（仅监听本地回环地址）"""
    server = HTTPServer(("127.0.0.1", port), AuthCallbackHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.timeout = 1.0
    t = threading.Thread(target=server.serve_forever, daemon=True, name=f"OAuth-{port}")
    t.start()
    log.info(f"OAuth 回调服务器已启动，端口: {port}")
    return server


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

async def verify_password(password: str) -> bool:
    """验证控制面板密码"""
    from config import get_panel_password
    return password == await get_panel_password()


async def create_auth_url(
    project_id: Optional[str] = None,
    user_session: Optional[str] = None,
    mode: str = "geminicli",
) -> Dict[str, Any]:
    """创建 OAuth 授权 URL，并启动本地回调服务器"""
    try:
        callback_port = await find_available_port()
        callback_url = f"http://{CALLBACK_HOST}:{callback_port}"

        server = _create_callback_server(callback_port)

        flow = Flow(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
            redirect_uri=callback_url,
        )

        state = f"{user_session}_{uuid.uuid4()}" if user_session else str(uuid.uuid4())
        auth_url = flow.get_auth_url(state=state)

        # 限制最大并发认证流程数
        if len(auth_flows) >= MAX_AUTH_FLOWS:
            oldest = min(auth_flows, key=lambda k: auth_flows[k].get("created_at", 0))
            _cleanup_auth_flow_server(oldest)

        auth_flows[state] = {
            "flow": flow,
            "project_id": project_id,
            "user_session": user_session,
            "callback_port": callback_port,
            "callback_url": callback_url,
            "server": server,
            "code": None,
            "completed": False,
            "created_at": time.time(),
            "auto_project_detection": project_id is None,
            "mode": mode,
        }

        cleanup_expired_flows()

        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "callback_port": callback_port,
            "callback_url": callback_url,
            "auto_project_detection": project_id is None,
            "detected_project_id": project_id,
        }
    except Exception as e:
        log.error(f"创建授权 URL 失败: {e}")
        return {"success": False, "error": str(e)}


def get_auth_status(state: str) -> Dict[str, Any]:
    """查询认证流程状态"""
    if state not in auth_flows:
        return {"state": state, "found": False, "completed": False}
    flow_data = auth_flows[state]
    return {
        "state": state,
        "found": True,
        "completed": flow_data.get("completed", False),
        "has_code": bool(flow_data.get("code")),
    }


async def asyncio_complete_auth_flow(
    project_id: Optional[str] = None,
    user_session: Optional[str] = None,
    state: Optional[str] = None,
    timeout: int = 300,
    mode: str = "geminicli",
) -> Dict[str, Any]:
    """
    等待 OAuth 回调完成，然后交换 token 并保存凭证。
    state 可选：若提供则查找指定流程；否则查找最新的未完成流程。
    """
    # 查找流程
    flow_data = None
    matched_state = state

    if state and state in auth_flows:
        flow_data = auth_flows[state]
    else:
        # 查找最新且属于该 user_session 的未完成流程
        for s, d in sorted(auth_flows.items(), key=lambda x: x[1].get("created_at", 0), reverse=True):
            if not d.get("completed") or not d.get("code"):
                if user_session is None or d.get("user_session") == user_session:
                    flow_data = d
                    matched_state = s
                    break

    if not flow_data:
        return {"success": False, "error": "未找到进行中的认证流程，请先点击「开始授权」"}

    # 等待 code
    start = time.time()
    while time.time() - start < timeout:
        if flow_data.get("code"):
            break
        await asyncio.sleep(0.5)
        if matched_state in auth_flows:
            flow_data = auth_flows[matched_state]

    if not flow_data.get("code"):
        _cleanup_auth_flow_server(matched_state)
        return {"success": False, "error": f"等待 OAuth 回调超时（{timeout} 秒）"}

    return await _exchange_and_save(flow_data, matched_state, project_id, mode)


async def complete_auth_flow_from_callback_url(
    callback_url: str,
    project_id: Optional[str] = None,
    mode: str = "geminicli",
) -> Dict[str, Any]:
    """从粘贴的完整回调 URL 完成认证"""
    try:
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if not code:
            return {"success": False, "error": "回调 URL 中未找到授权码（code 参数）"}

        # 查找对应的 flow（如果有）
        flow_data = auth_flows.get(state) if state else None

        if flow_data:
            flow_data["code"] = code
            flow_data["completed"] = True
            return await _exchange_and_save(flow_data, state, project_id, mode)
        else:
            # 没有对应 flow，直接用 OOB 方式交换
            parsed_base = urlparse(callback_url)
            redirect_base = f"{parsed_base.scheme}://{parsed_base.netloc}{parsed_base.path}"
            flow = Flow(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                scopes=SCOPES,
                redirect_uri=redirect_base,
            )
            credentials = await flow.exchange_code(code=code)
            return await _save_credentials(credentials, project_id, mode)
    except Exception as e:
        log.error(f"从回调 URL 完成认证失败: {e}")
        return {"success": False, "error": str(e)}


async def _exchange_and_save(
    flow_data: Dict[str, Any],
    state: str,
    override_project_id: Optional[str],
    mode: str,
) -> Dict[str, Any]:
    """内部：交换 code → token，再保存凭证"""
    try:
        flow: Flow = flow_data["flow"]
        code: str = flow_data["code"]
        project_id = override_project_id or flow_data.get("project_id")

        credentials = await flow.exchange_code(code=code)

        # 清理服务器
        _cleanup_auth_flow_server(state)

        return await _save_credentials(credentials, project_id, mode)
    except Exception as e:
        log.error(f"Token 交换失败: {e}")
        _cleanup_auth_flow_server(state)
        return {"success": False, "error": str(e)}


async def _save_credentials(
    credentials: Credentials,
    project_id: Optional[str],
    mode: str,
) -> Dict[str, Any]:
    """获取 project_id（如未提供则自动获取）并保存凭证"""
    try:
        from .google_oauth_api import get_user_email

        # 自动获取 project_id
        if not project_id:
            try:
                api_base_url = await get_code_assist_endpoint()
                from .utils import GEMINICLI_USER_AGENT
                project_id = await fetch_project_id(
                    access_token=credentials.access_token,
                    user_agent=GEMINICLI_USER_AGENT,
                    api_base_url=api_base_url,
                )
                log.info(f"自动获取 project_id: {project_id}")
            except Exception as e:
                log.warning(f"自动获取 project_id 失败: {e}")

        if not project_id:
            # 尝试获取项目列表让用户选择
            try:
                projects = await get_user_projects(credentials)
                if projects:
                    if len(projects) == 1:
                        project_id = projects[0].get("projectId") or projects[0].get("name", "").split("/")[-1]
                    else:
                        return {
                            "success": False,
                            "error": "需要选择一个 Google Cloud 项目",
                            "requires_project_selection": True,
                            "available_projects": [
                                {"id": p.get("projectId", ""), "name": p.get("name", "")}
                                for p in projects[:20]
                            ],
                        }
            except Exception as e:
                log.warning(f"获取项目列表失败: {e}")

        if not project_id:
            return {
                "success": False,
                "error": "无法自动获取 Project ID，请手动填写",
                "requires_manual_project_id": True,
            }

        # 获取邮箱
        email = ""
        try:
            email = await get_user_email(credentials) or ""
        except Exception:
            pass

        # 保存凭证
        creds_data = _prepare_credentials_data(credentials, project_id)
        creds_data["email"] = email

        filename = f"{email or 'credential'}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"

        storage_adapter = await get_storage_adapter()
        await storage_adapter.store_credential(filename, creds_data, mode=mode)
        log.info(f"凭证已保存: {filename} (mode={mode}, project={project_id})")

        return {
            "success": True,
            "file_path": filename,
            "credentials": {
                "email": email,
                "project_id": project_id,
                "filename": filename,
            },
            "auto_detected_project": True,
        }
    except Exception as e:
        log.error(f"保存凭证失败: {e}")
        return {"success": False, "error": str(e)}
