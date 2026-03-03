"""
Vertex AI API Client - Handles all communication with Google Vertex AI API.
通过 Vertex AI 端点处理 Gemini API 请求
"""

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import Response
from config import get_auto_ban_error_codes, get_vertex_ai_location
from log import log

from src.credential_manager import credential_manager
from src.httpx_client import stream_post_async, post_async

from src.api.utils import (
    handle_error_with_retry,
    get_retry_config,
    record_api_call_success,
    record_api_call_error,
    parse_and_log_cooldown,
)
from src.utils import GEMINICLI_USER_AGENT

# 复用 geminicli 的模型版本工具（Vertex AI 响应无 response 包装，直接用）
from src.api.geminicli import _extract_model_series, _log_model_version


def _extract_vertex_model_version(resp_json: Any) -> Optional[str]:
    """
    从 Vertex AI 响应中提取 modelVersion。

    Vertex AI 使用标准 Gemini API 格式（无 response 包装层）::

        {"candidates": [...], "modelVersion": "..."}
    """
    if isinstance(resp_json, list):
        resp_json = resp_json[0] if resp_json else {}
    if not isinstance(resp_json, dict):
        return None
    return resp_json.get("modelVersion") or resp_json.get("model") or None


# ==================== 请求准备 ====================

async def prepare_vertex_request(payload: dict, credential_data: dict, location: str):
    """
    从凭证数据准备 Vertex AI 请求头和最终 payload

    Args:
        payload: 原始请求 payload（包含 model 和 request 字段）
        credential_data: 凭证数据字典
        location: Vertex AI 区域（如 us-central1）

    Returns:
        元组: (headers, final_payload, stream_url, non_stream_url)

    Raises:
        Exception: 如果凭证中缺少必要字段
    """
    token = credential_data.get("token") or credential_data.get("access_token", "")
    if not token:
        raise Exception("凭证中没有找到有效的访问令牌（token或access_token字段）")

    project_id = credential_data.get("project_id", "")
    if not project_id:
        raise Exception("项目ID不存在于凭证数据中")

    model_name = payload.get("model", "")
    source_request = payload.get("request", {})

    # Vertex AI 端点
    stream_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/{model_name}:streamGenerateContent?alt=sse"
    )
    non_stream_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/{model_name}:generateContent"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": GEMINICLI_USER_AGENT,
    }

    # Vertex AI 使用请求体直接发送（无 project 包装层）
    final_payload = source_request

    return headers, final_payload, stream_url, non_stream_url


# ==================== 流式请求 ====================

async def stream_request(
    body: Dict[str, Any],
    native: bool = False,
    headers: Optional[Dict[str, str]] = None,
):
    """
    Vertex AI 流式请求函数

    Args:
        body: 请求体
        native: 是否返回原生bytes流，False则返回str流
        headers: 额外的请求头

    Yields:
        Response对象（错误时）或 bytes流/str流（成功时）
    """
    model_name = body.get("model", "")

    # 1. 获取有效凭证
    cred_result = await credential_manager.get_valid_credential(
        mode="geminicli", model_name=model_name
    )

    if not cred_result:
        yield Response(
            content=json.dumps({"error": "当前无可用凭证"}),
            status_code=500,
            media_type="application/json"
        )
        return

    current_file, credential_data = cred_result

    # 2. 构建URL和请求头
    try:
        location = await get_vertex_ai_location()
        auth_headers, final_payload, stream_url, _ = await prepare_vertex_request(
            body, credential_data, location
        )

        if headers:
            auth_headers.update(headers)

    except Exception as e:
        log.error(f"[VERTEX STREAM] 准备请求失败: {e}")
        yield Response(
            content=json.dumps({"error": f"准备请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )
        return

    # 3. 调用stream_post_async进行请求
    retry_config = await get_retry_config()
    max_retries = retry_config["max_retries"]
    retry_interval = retry_config["retry_interval"]

    DISABLE_ERROR_CODES = await get_auto_ban_error_codes()
    last_error_response = None
    next_cred_task = None

    async def refresh_credential_fast():
        nonlocal current_file, credential_data, auth_headers, final_payload, stream_url
        try:
            new_cred_result = await credential_manager.get_valid_credential(
                mode="geminicli", model_name=model_name
            )
            if not new_cred_result:
                return False
            current_file, credential_data = new_cred_result
            token = credential_data.get("token") or credential_data.get("access_token", "")
            project_id = credential_data.get("project_id", "")
            if not token or not project_id:
                return False
            auth_headers["Authorization"] = f"Bearer {token}"
            location = await get_vertex_ai_location()
            stream_url = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
                f"/locations/{location}/publishers/google/models/{model_name}:streamGenerateContent?alt=sse"
            )
            return True
        except Exception:
            return None

    for attempt in range(max_retries + 1):
        success_recorded = False
        need_retry = False
        stream_model_version_logged = False

        try:
            async for chunk in stream_post_async(
                url=stream_url,
                body=final_payload,
                native=native,
                headers=auth_headers
            ):
                if isinstance(chunk, Response):
                    status_code = chunk.status_code
                    last_error_response = chunk

                    error_body = None
                    try:
                        error_body = chunk.body.decode('utf-8') if isinstance(chunk.body, bytes) else str(chunk.body)
                    except Exception:
                        error_body = ""

                    if status_code == 429 or status_code == 503 or status_code in DISABLE_ERROR_CODES:
                        log.warning(f"[VERTEX STREAM] 流式请求失败 (status={status_code}), 凭证: {current_file}, 响应: {error_body[:500] if error_body else '无'}")

                        cooldown_until = None
                        if (status_code == 429 or status_code == 503) and error_body:
                            try:
                                cooldown_until = await parse_and_log_cooldown(error_body, mode="vertex")
                            except Exception:
                                pass

                        if next_cred_task is None and attempt < max_retries:
                            next_cred_task = asyncio.create_task(
                                credential_manager.get_valid_credential(
                                    mode="geminicli", model_name=model_name
                                )
                            )

                        await record_api_call_error(
                            credential_manager, current_file, status_code,
                            cooldown_until, mode="geminicli", model_name=model_name,
                            error_message=error_body
                        )

                        should_retry = await handle_error_with_retry(
                            credential_manager, status_code, current_file,
                            retry_config["retry_enabled"], attempt, max_retries, retry_interval,
                            mode="geminicli"
                        )

                        if should_retry and attempt < max_retries:
                            need_retry = True
                            break
                        else:
                            log.error(f"[VERTEX STREAM] 达到最大重试次数或不应重试，返回原始错误")
                            yield chunk
                            return

                    else:
                        await record_api_call_error(
                            credential_manager, current_file, status_code,
                            None, mode="geminicli", model_name=model_name,
                            error_message=error_body
                        )
                        log.error(f"[VERTEX STREAM] 流式请求失败，非重试错误码 (status={status_code}), 凭证: {current_file}")
                        yield chunk
                        return
                else:
                    if not success_recorded:
                        await record_api_call_success(
                            credential_manager, current_file, mode="geminicli", model_name=model_name
                        )
                        success_recorded = True
                        log.debug(f"[VERTEX STREAM] 开始接收流式响应，模型: {model_name}")

                    # 从流式数据块中提取模型版本（检测模型降级）
                    if not stream_model_version_logged and isinstance(chunk, (str, bytes)):
                        try:
                            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                            for line in chunk_str.splitlines():
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_part = line[5:].strip()
                                    if data_part and data_part != "[DONE]":
                                        chunk_json = json.loads(data_part)
                                        # Vertex AI 使用标准 Gemini 格式，无包装层
                                        actual_model_version = _extract_vertex_model_version(chunk_json)
                                        if actual_model_version:
                                            _log_model_version(model_name, actual_model_version)
                                            stream_model_version_logged = True
                                            break
                        except Exception:
                            pass

                    yield chunk

            if success_recorded:
                log.debug(f"[VERTEX STREAM] 流式响应完成，模型: {model_name}")
                return

            if need_retry:
                log.info(f"[VERTEX STREAM] 重试请求 (attempt {attempt + 2}/{max_retries + 1})...")

                if next_cred_task is not None:
                    try:
                        cred_result = await next_cred_task
                        next_cred_task = None

                        if cred_result:
                            current_file, credential_data = cred_result
                            token = credential_data.get("token") or credential_data.get("access_token", "")
                            project_id = credential_data.get("project_id", "")
                            if token and project_id:
                                auth_headers["Authorization"] = f"Bearer {token}"
                                location = await get_vertex_ai_location()
                                stream_url = (
                                    f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
                                    f"/locations/{location}/publishers/google/models/{model_name}:streamGenerateContent?alt=sse"
                                )
                                await asyncio.sleep(retry_interval)
                                continue
                    except Exception as e:
                        log.warning(f"[VERTEX STREAM] 预热凭证任务失败: {e}")
                        next_cred_task = None

                await asyncio.sleep(retry_interval)

                if not await refresh_credential_fast():
                    log.error("[VERTEX STREAM] 重试时无可用凭证或刷新失败")
                    yield Response(
                        content=json.dumps({"error": "当前无可用凭证"}),
                        status_code=500,
                        media_type="application/json"
                    )
                    return
                continue

        except Exception as e:
            log.error(f"[VERTEX STREAM] 流式请求异常: {e}, 凭证: {current_file}")
            if attempt < max_retries:
                log.info(f"[VERTEX STREAM] 异常后重试 (attempt {attempt + 2}/{max_retries + 1})...")
                await asyncio.sleep(retry_interval)
                continue
            else:
                log.error(f"[VERTEX STREAM] 所有重试均失败，最后异常: {e}")
                if last_error_response:
                    yield last_error_response
                else:
                    yield Response(
                        content=json.dumps({"error": f"流式请求异常: {str(e)}"}),
                        status_code=500,
                        media_type="application/json"
                    )
                return


# ==================== 非流式请求 ====================

async def non_stream_request(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    Vertex AI 非流式请求函数

    Args:
        body: 请求体
        headers: 额外的请求头

    Returns:
        Response对象
    """
    model_name = body.get("model", "")

    # 1. 获取有效凭证
    cred_result = await credential_manager.get_valid_credential(
        mode="geminicli", model_name=model_name
    )

    if not cred_result:
        return Response(
            content=json.dumps({"error": "当前无可用凭证"}),
            status_code=500,
            media_type="application/json"
        )

    current_file, credential_data = cred_result

    # 2. 构建URL和请求头
    try:
        location = await get_vertex_ai_location()
        auth_headers, final_payload, _, non_stream_url = await prepare_vertex_request(
            body, credential_data, location
        )

        if headers:
            auth_headers.update(headers)

    except Exception as e:
        log.error(f"[VERTEX] 准备请求失败: {e}")
        return Response(
            content=json.dumps({"error": f"准备请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )

    # 3. 调用post_async进行请求
    retry_config = await get_retry_config()
    max_retries = retry_config["max_retries"]
    retry_interval = retry_config["retry_interval"]

    DISABLE_ERROR_CODES = await get_auto_ban_error_codes()
    last_error_response = None
    next_cred_task = None

    for attempt in range(max_retries + 1):
        try:
            response = await post_async(
                url=non_stream_url,
                json=final_payload,
                headers=auth_headers,
                timeout=300.0
            )

            status_code = response.status_code

            # 成功
            if status_code == 200:
                await record_api_call_success(
                    credential_manager, current_file, mode="geminicli", model_name=model_name
                )
                response_headers = dict(response.headers)
                response_headers.pop('content-encoding', None)
                response_headers.pop('content-length', None)

                # 记录实际响应的模型版本（检测模型降级）
                try:
                    resp_json = response.json()
                    # Vertex AI 使用标准 Gemini 格式，无包装层
                    actual_model_version = _extract_vertex_model_version(resp_json)
                    if actual_model_version:
                        _log_model_version(model_name, actual_model_version)
                except Exception:
                    pass

                return Response(
                    content=response.content,
                    status_code=200,
                    headers=response_headers
                )

            # 失败
            error_headers = dict(response.headers)
            error_headers.pop('content-encoding', None)
            error_headers.pop('content-length', None)

            last_error_response = Response(
                content=response.content,
                status_code=status_code,
                headers=error_headers
            )

            error_text = ""
            try:
                error_text = response.text
            except Exception:
                pass

            if status_code == 429 or status_code == 503 or status_code in DISABLE_ERROR_CODES:
                log.warning(f"[VERTEX] 非流式请求失败 (status={status_code}), 凭证: {current_file}, 响应: {error_text[:500] if error_text else '无'}")

                cooldown_until = None
                if (status_code == 429 or status_code == 503) and error_text:
                    try:
                        cooldown_until = await parse_and_log_cooldown(error_text, mode="vertex")
                    except Exception:
                        pass

                if next_cred_task is None and attempt < max_retries:
                    next_cred_task = asyncio.create_task(
                        credential_manager.get_valid_credential(
                            mode="geminicli", model_name=model_name
                        )
                    )

                await record_api_call_error(
                    credential_manager, current_file, status_code,
                    cooldown_until, mode="geminicli", model_name=model_name,
                    error_message=error_text
                )

                should_retry = await handle_error_with_retry(
                    credential_manager, status_code, current_file,
                    retry_config["retry_enabled"], attempt, max_retries, retry_interval,
                    mode="geminicli"
                )

                if should_retry and attempt < max_retries:
                    log.info(f"[VERTEX] 重试请求 (attempt {attempt + 2}/{max_retries + 1})...")

                    if next_cred_task is not None:
                        try:
                            cred_result = await next_cred_task
                            next_cred_task = None

                            if cred_result:
                                current_file, credential_data = cred_result
                                token = credential_data.get("token") or credential_data.get("access_token", "")
                                project_id = credential_data.get("project_id", "")
                                if token and project_id:
                                    auth_headers["Authorization"] = f"Bearer {token}"
                                    location = await get_vertex_ai_location()
                                    non_stream_url = (
                                        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
                                        f"/locations/{location}/publishers/google/models/{model_name}:generateContent"
                                    )
                                    await asyncio.sleep(retry_interval)
                                    continue
                        except Exception as e:
                            log.warning(f"[VERTEX] 预热凭证任务失败: {e}")
                            next_cred_task = None

                    await asyncio.sleep(retry_interval)

                    new_cred_result = await credential_manager.get_valid_credential(
                        mode="geminicli", model_name=model_name
                    )
                    if new_cred_result:
                        current_file, credential_data = new_cred_result
                        token = credential_data.get("token") or credential_data.get("access_token", "")
                        project_id = credential_data.get("project_id", "")
                        if token and project_id:
                            auth_headers["Authorization"] = f"Bearer {token}"
                            location = await get_vertex_ai_location()
                            non_stream_url = (
                                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
                                f"/locations/{location}/publishers/google/models/{model_name}:generateContent"
                            )
                            continue

                return last_error_response

            else:
                log.error(f"[VERTEX] 非流式请求失败，非重试错误码 (status={status_code}), 凭证: {current_file}")
                await record_api_call_error(
                    credential_manager, current_file, status_code,
                    None, mode="geminicli", model_name=model_name,
                    error_message=error_text
                )
                return last_error_response

        except Exception as e:
            log.error(f"[VERTEX] 非流式请求异常: {e}, 凭证: {current_file}")
            if attempt < max_retries:
                log.info(f"[VERTEX] 异常后重试 (attempt {attempt + 2}/{max_retries + 1})...")
                await asyncio.sleep(retry_interval)
                continue
            else:
                log.error(f"[VERTEX] 所有重试均失败，最后异常: {e}")
                if last_error_response:
                    return last_error_response
                return Response(
                    content=json.dumps({"error": f"请求异常: {str(e)}"}),
                    status_code=500,
                    media_type="application/json"
                )

    # 超过最大重试次数
    log.error(f"[VERTEX] 超过最大重试次数")
    if last_error_response:
        return last_error_response
    return Response(
        content=json.dumps({"error": "超过最大重试次数"}),
        status_code=500,
        media_type="application/json"
    )
