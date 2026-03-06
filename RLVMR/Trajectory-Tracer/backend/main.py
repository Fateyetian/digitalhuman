"""
Trajectory Viewer Backend - FastAPI Service
提供轨迹数据的 REST API
支持多种轨迹数据格式
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import json
from pathlib import Path
from trajectory_adapters import TrajectoryLoader

app = FastAPI(title="Trajectory Viewer API", version="2.1.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管前端静态文件（必须在所有 API 路由注册之后挂载）
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# 全局变量存储轨迹数据
trajectory_loader = TrajectoryLoader()
processed_trajectories = []
# 记录已加载的数据源路径，用于 reload
_loaded_data_sources = []


class Message(BaseModel):
    """单条消息"""
    role: str  # 'human' 或 'agent'
    content: str
    thought: Optional[str] = None
    action: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrajectoryInfo(BaseModel):
    """轨迹基本信息"""
    id: str
    task: str
    status: str  # 'success' 或 'failed'
    steps: int
    task_type: str


class TrajectoryDetail(BaseModel):
    """轨迹详细信息"""
    id: str
    task: str
    status: str
    steps: int
    task_type: str
    messages: List[Message]
    environment: str
    metadata: Optional[Dict[str, Any]] = None




def _resolve_data_sources() -> List[Dict[str, Any]]:
    """解析数据源配置，合并 data_sources.json 和环境变量"""
    base_path = Path(__file__).parent.parent
    sources = []

    # 1. 从 data_sources.json 加载
    ds_config_path = Path(__file__).parent / "data_sources.json"
    if ds_config_path.exists():
        try:
            with open(ds_config_path, 'r') as f:
                config = json.load(f)
            for src in config.get('data_sources', []):
                if not src.get('enabled', True):
                    continue
                src_path = src.get('path', '')
                fmt = src.get('type', None)

                p = Path(src_path)
                if not p.is_absolute():
                    p = base_path / src_path

                # 如果是目录，分两种情况：
                # (a) 目录下直接有 rollout_*.jsonl → 单个实验目录
                # (b) 目录下是多个实验子目录（每个子目录含 rollout_*.jsonl）→ 父目录
                if p.is_dir():
                    direct_files = sorted(p.glob('rollout_*.jsonl'))
                    if direct_files:
                        # 单个实验目录
                        for jf in direct_files:
                            sources.append({
                                'name': f"{p.name}/{jf.name}",
                                'path': jf,
                                'type': fmt or 'webshop_rl_jsonl',
                            })
                    else:
                        # 父目录：扫描所有子目录
                        subdirs = sorted(p.iterdir())
                        found_any = False
                        for subdir in subdirs:
                            if not subdir.is_dir():
                                continue
                            sub_files = sorted(subdir.glob('rollout_*.jsonl'))
                            if sub_files:
                                found_any = True
                                exp_name = subdir.name  # e.g. "20260306_120000_M5_rebel_full_seed42"
                                for jf in sub_files:
                                    sources.append({
                                        'name': f"{exp_name}/{jf.name}",
                                        'path': jf,
                                        'type': fmt or 'webshop_rl_jsonl',
                                    })
                                print(f"Found {len(sub_files)} rollout files in {exp_name}")
                        if not found_any:
                            print(f"Info: No rollout_*.jsonl found under {p} (training not started yet)")
                else:
                    sources.append({
                        'name': src.get('name', src_path),
                        'path': p,
                        'type': fmt,
                    })
        except Exception as e:
            print(f"Warning: Failed to read data_sources.json: {e}")

    # 2. 默认内置数据源（retro_traj）
    app_path = Path("/app")
    if app_path.exists() and (app_path / "retro_traj").exists():
        sources.append({'name': 'Retrosynthesis (Docker)', 'path': app_path / "retro_traj", 'type': None})
    else:
        retro_path = base_path / "retro_traj"
        if retro_path.exists():
            sources.append({'name': 'Retrosynthesis', 'path': retro_path, 'type': None})

    # 3. 环境变量额外数据源: TRAJ_VIEWER_EXTRA_SOURCES (逗号分隔的路径)
    extra = os.environ.get('TRAJ_VIEWER_EXTRA_SOURCES', '')
    if extra:
        for ep in extra.split(','):
            ep = ep.strip()
            if ep:
                sources.append({'name': f'Extra: {Path(ep).name}', 'path': Path(ep), 'type': None})

    return sources


def _load_all_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从多个数据源加载轨迹，返回字典列表"""
    all_trajs = []
    for src in sources:
        data_path = src['path']
        fmt = src.get('type')
        if not data_path.exists():
            print(f"Warning: Data source not found: {data_path} ({src['name']})")
            continue
        try:
            trajectories = trajectory_loader.load(data_path, format_type=fmt)
            for traj in trajectories:
                all_trajs.append(traj.to_dict())
            print(f"Loaded {len(trajectories)} trajectories from {src['name']} ({data_path})")
        except Exception as e:
            print(f"Warning: Failed to load {src['name']} ({data_path}): {e}")
    return all_trajs


@app.on_event("startup")
async def load_data():
    """启动时加载数据集"""
    global processed_trajectories, _loaded_data_sources

    _loaded_data_sources = _resolve_data_sources()
    processed_trajectories = _load_all_sources(_loaded_data_sources)
    print(f"Total processed trajectories: {len(processed_trajectories)}")


@app.get("/health")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "message": "Trajectory Viewer API is running",
        "trajectories_loaded": len(processed_trajectories)
    }


@app.get("/api/trajectories", response_model=List[TrajectoryInfo])
async def get_trajectories(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None, pattern="^(success|failed|unknown)$"),
    task_type: Optional[str] = Query(None),
    min_steps: Optional[int] = Query(None, ge=0),
    max_steps: Optional[int] = Query(None, ge=0),
):
    """
    获取轨迹列表（支持分页和筛选）
    """
    if not processed_trajectories:
        return []

    # 筛选
    filtered = processed_trajectories

    if status:
        filtered = [t for t in filtered if t['status'] == status]

    if task_type:
        filtered = [t for t in filtered if t['task_type'] == task_type]

    if min_steps is not None:
        filtered = [t for t in filtered if t['steps'] >= min_steps]

    if max_steps is not None:
        filtered = [t for t in filtered if t['steps'] <= max_steps]

    # 分页
    total = len(filtered)
    results = filtered[skip:skip + limit]

    # 转换为响应模型
    return [
        TrajectoryInfo(
            id=t['id'],
            task=t['task'],
            status=t['status'],
            steps=t['steps'],
            task_type=t['task_type']
        )
        for t in results
    ]


@app.get("/api/trajectories/{trajectory_id}", response_model=TrajectoryDetail)
async def get_trajectory_detail(trajectory_id: str):
    """
    获取单条轨迹的详细信息
    """
    # 查找轨迹
    trajectory = next((t for t in processed_trajectories if t['id'] == trajectory_id), None)

    if not trajectory:
        raise HTTPException(status_code=404, detail="Trajectory not found")

    return TrajectoryDetail(
        id=trajectory['id'],
        task=trajectory['task'],
        status=trajectory['status'],
        steps=trajectory['steps'],
        task_type=trajectory['task_type'],
        messages=trajectory['messages'],
        environment=trajectory['environment'],
        metadata=trajectory.get('metadata')
    )


@app.get("/api/statistics")
async def get_statistics():
    """
    获取统计信息
    """
    if not processed_trajectories:
        return {
            "total": 0,
            "by_status": {},
            "by_task_type": {},
            "by_source": {},
            "avg_steps": 0
        }

    total = len(processed_trajectories)

    # 按状态统计
    by_status = {}
    for t in processed_trajectories:
        status = t['status']
        by_status[status] = by_status.get(status, 0) + 1

    # 按任务类型统计
    by_task_type = {}
    for t in processed_trajectories:
        task_type = t['task_type']
        by_task_type[task_type] = by_task_type.get(task_type, 0) + 1

    # 按数据源统计
    by_source = {}
    for t in processed_trajectories:
        source = t['metadata'].get('source', 'unknown')
        by_source[source] = by_source.get(source, 0) + 1

    # 平均步数
    total_steps = sum(t['steps'] for t in processed_trajectories)
    avg_steps = total_steps / total if total > 0 else 0

    return {
        "total": total,
        "by_status": by_status,
        "by_task_type": by_task_type,
        "by_source": by_source,
        "avg_steps": round(avg_steps, 2)
    }


@app.get("/api/data-sources")
async def get_data_sources():
    """
    获取已加载的数据源信息
    """
    sources = {}
    for traj in processed_trajectories:
        source = traj['metadata'].get('source', 'unknown')
        if source not in sources:
            sources[source] = {
                'count': 0,
                'format': source,
                'sample_id': traj['id']
            }
        sources[source]['count'] += 1

    return {
        'total_sources': len(sources),
        'sources': list(sources.values())
    }


@app.post("/api/reload")
async def reload_data():
    """
    重新加载所有数据源（用于实时查看新生成的标注数据）
    """
    global processed_trajectories, _loaded_data_sources

    old_count = len(processed_trajectories)
    _loaded_data_sources = _resolve_data_sources()
    processed_trajectories = _load_all_sources(_loaded_data_sources)
    new_count = len(processed_trajectories)

    return {
        'status': 'ok',
        'previous_count': old_count,
        'current_count': new_count,
        'delta': new_count - old_count
    }


@app.post("/api/add-source")
async def add_data_source(path: str = Query(..., description="数据源路径")):
    """
    动态添加一个数据源并立即加载
    """
    global processed_trajectories, _loaded_data_sources

    data_path = Path(path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    src = {'name': f'Dynamic: {data_path.name}', 'path': data_path, 'type': None}
    _loaded_data_sources.append(src)

    try:
        new_trajs = _load_all_sources([src])
        processed_trajectories.extend(new_trajs)
        return {
            'status': 'ok',
            'added': len(new_trajs),
            'total': len(processed_trajectories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trajectory-groups")
async def get_trajectory_groups(
    source: Optional[str] = Query(None, description="按数据源过滤，如 webshop_rl"),
):
    """
    按任务(group_id)分组返回轨迹，每组包含同一任务的所有rollout轨迹。
    用于分析同一任务下不同轨迹的差异和reward分布。
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for traj in processed_trajectories:
        meta = traj.get('metadata', {})
        traj_source = meta.get('source', 'unknown')
        if source and traj_source != source:
            continue

        group_id = meta.get('group_id', traj.get('task', '')[:60])

        # Extract per-step actions and rewards from messages
        actions, step_rewards = [], []
        for msg in traj.get('messages', []):
            if msg.get('role') == 'agent' and msg.get('action'):
                actions.append(msg['action'])
                step_rewards.append(msg.get('metadata', {}).get('reward', 0.0))

        groups[group_id].append({
            'id': traj['id'],
            'task': traj['task'],
            'status': traj['status'],
            'steps': traj['steps'],
            'total_reward': meta.get('total_reward', 0.0),
            'traj_index': meta.get('traj_index', 0),
            'source': traj_source,
            'exp_tag': meta.get('exp_tag', ''),
            'actions': actions,
            'step_rewards': step_rewards,
        })

    # 构建分组列表，按reward均值排序
    group_list = []
    for group_id, trajs in groups.items():
        rewards = [t['total_reward'] for t in trajs]
        group_list.append({
            'group_id': group_id,
            'task': trajs[0]['task'] if trajs else '',
            'count': len(trajs),
            'reward_mean': sum(rewards) / len(rewards) if rewards else 0.0,
            'reward_max': max(rewards) if rewards else 0.0,
            'reward_min': min(rewards) if rewards else 0.0,
            'reward_std': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5 if len(rewards) > 1 else 0.0,
            'success_count': sum(1 for t in trajs if t['status'] == 'success'),
            'trajectories': sorted(trajs, key=lambda x: x['total_reward'], reverse=True),
        })

    group_list.sort(key=lambda x: x['reward_mean'], reverse=True)
    return {'total_groups': len(group_list), 'groups': group_list}


@app.get("/api/training-metrics")
async def get_training_metrics(
    metrics_path: Optional[str] = Query(None, description="训练指标JSON文件路径"),
):
    """
    返回训练过程中的step-level指标（reward曲线、valid_action_ratio等）
    用于可视化训练动态
    """
    # 默认路径
    if not metrics_path:
        default = Path("/root/testttt/RLVMR/Trajectory-Tracer/webshop_rl_traj/rl_trajectories_metrics.json")
        if default.exists():
            metrics_path = str(default)
        else:
            return {'steps': [], 'message': 'No metrics file found'}

    p = Path(metrics_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {metrics_path}")

    with open(p, 'r') as f:
        metrics = json.load(f)

    return {'steps': metrics, 'total_steps': len(metrics)}


@app.get("/api/prompts")
async def get_prompts():
    """
    返回所有 Prompt 模板的结构化目录（ALFWorld + WebShop，含元数据分析）
    用于 Prompt 对比可视化界面
    """
    from prompt_catalog import build_prompt_catalog
    return build_prompt_catalog()


# ── 前端静态文件托管（放在所有 API 路由之后）─────────────────────────────
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """对所有非 /api 路径返回 index.html（SPA 路由支持）"""
        index = _FRONTEND_DIST / "index.html"
        return FileResponse(str(index))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
