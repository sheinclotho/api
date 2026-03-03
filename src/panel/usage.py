"""
使用统计路由模块 - 处理 /usage/* 相关的HTTP请求
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.utils import verify_panel_token


router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/stats")
async def get_usage_stats(token: str = Depends(verify_panel_token)):
    """获取各凭证使用统计"""
    return JSONResponse(content={"success": True, "data": {}})


@router.get("/aggregated")
async def get_aggregated_stats(token: str = Depends(verify_panel_token)):
    """获取汇总使用统计"""
    return JSONResponse(content={
        "success": True,
        "data": {
            "total_calls_24h": 0,
            "total_files": 0,
            "avg_calls_per_file": 0,
        }
    })


@router.post("/reset")
async def reset_usage_stats(token: str = Depends(verify_panel_token)):
    """重置使用统计"""
    return JSONResponse(content={"success": True, "message": "统计已重置"})
