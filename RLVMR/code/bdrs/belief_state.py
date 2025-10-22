from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import re
import time


@dataclass
class BeliefSnapshot:
    step_idx: int
    timestamp: float
    world_model: Dict[str, Any]
    task_progress: Dict[str, Any]
    exploration_map: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


@dataclass
class BeliefState:
    # M_t: 世界模型（对环境事实的信念）
    world_model: Dict[str, Any] = field(default_factory=lambda: {
        "visible_objects": set(),     # {'apple 1', 'fridge 1', ...}
        "current_room": None,         # 'kitchen', 'living room', ...
        "object_locations": {},       # {'apple 1': 'counter 1', ...}
        "last_observation": "",
    })
    # P_t: 任务进展（目标/子目标状态）
    task_progress: Dict[str, Any] = field(default_factory=lambda: {
        "task_description": "",
        "subgoals": [],               # [{'text': 'heat apple', 'status': 'pending|completed'}]
        "current_subgoal_idx": 0,
    })
    # E_t: 探索地图（到过哪里/见过什么）
    exploration_map: Dict[str, Any] = field(default_factory=lambda: {
        "visited_rooms": set(),
        "visited_containers": set(),
        "visited_objects": set(),
        "trajectory": [],             # [{'obs': ..., 'act': ...}]
    })
    # 反思/验证记录
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "consistency_checks": [],     # [{'ok': bool, 'reason': str}]
        "reflections": [],            # [{'type': 'verify', 'msg': str}]
        "step_idx": 0,
    })


class BeliefStateManager:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.states: Dict[int, BeliefState] = {}  # env_id -> BeliefState

    def reset(self, env_ids: List[int], task_desc: Optional[List[str]] = None):
        self.states.clear()
        for i, env_id in enumerate(env_ids):
            st = BeliefState()
            if task_desc is not None and i < len(task_desc):
                st.task_progress["task_description"] = task_desc[i]
            self.states[env_id] = st

    def step_update(
        self,
        env_id: int,
        observation: str,
        action: str,
        info: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
    ):
        st = self.states[env_id]
        st.meta["step_idx"] += 1

        # 1) 更新E_t: 轨迹/访问记录
        st.exploration_map["trajectory"].append({"obs": observation, "act": action})
        self._update_exploration(st, observation)

        # 2) 更新M_t: 世界模型（解析房间/可见物体/对象位置的粗糙启发式）
        self._update_world_model(st, observation, info)

        # 3) 更新P_t: 任务进展（启发式：根据动作/观察是否完成某些子目标）
        self._update_task_progress(st, observation, action, info)

        # 4) 观测-信念一致性验证（VERIFY信念）+ 记录“反思”
        ok, reason = self._consistency_check(st, observation, info)
        st.meta["consistency_checks"].append({"ok": ok, "reason": reason})
        if not ok:
            st.meta["reflections"].append({"type": "verify", "msg": reason})

        st.world_model["last_observation"] = observation

    def snapshot(self, env_id: int) -> BeliefSnapshot:
        st = self.states[env_id]
        return BeliefSnapshot(
            step_idx=st.meta["step_idx"],
            timestamp=time.time(),
            world_model=self._safe_copy(st.world_model),
            task_progress=self._safe_copy(st.task_progress),
            exploration_map=self._safe_copy(st.exploration_map),
            notes=[r.get("msg","") for r in st.meta["reflections"][-3:]],
        )

    # ----------------- internal helpers -----------------

    def _update_exploration(self, st: BeliefState, observation: str):
        # room name heuristic
        room = self._extract_room(observation)
        if room:
            st.world_model["current_room"] = room
            st.exploration_map["visited_rooms"].add(room)
        # containers/objects heuristics
        for obj in self._extract_objects(observation):
            st.exploration_map["visited_objects"].add(obj)

    def _update_world_model(self, st: BeliefState, observation: str, info: Optional[Dict[str, Any]]):
        # Extract visible objects
        visible = set(self._extract_objects(observation))
        st.world_model["visible_objects"] = visible
        # Extract simple "obj in/on recep" patterns from observation text
        for m in re.finditer(r"(\w+(?: \d+)?) (?:in|on) (\w+(?: \s*\d+)?)", observation):
            obj, recep = m.group(1), m.group(2)
            st.world_model["object_locations"][obj] = recep

    def _update_task_progress(self, st: BeliefState, observation: str, action: str, info: Optional[Dict[str, Any]]):
        # 如果还没有子目标，先用一个默认子目标容器（后续由高层填充）
        if not st.task_progress["subgoals"]:
            st.task_progress["subgoals"] = [{"text": "auto-init", "status": "pending"}]

        # 简单启发式：若动作以 put/heat/cool/clean/take/open/close/use 开头则推进子目标
        if re.search(r"^(put|heat|cool|clean|take|open|close|use)\b", action.strip()):
            idx = st.task_progress["current_subgoal_idx"]
            if 0 <= idx < len(st.task_progress["subgoals"]):
                st.task_progress["subgoals"][idx]["status"] = "completed"
                st.task_progress["current_subgoal_idx"] = min(idx + 1, len(st.task_progress["subgoals"]) - 1)

    def _consistency_check(self, st: BeliefState, observation: str, info: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        # 简单一致性：若world_model声明当前房间，但观测明显属于另一个典型房间词，做提示
        declared = st.world_model.get("current_room")
        hint = self._extract_room(observation)
        if declared and hint and declared != hint:
            return False, f"room mismatch: belief={declared}, obs={hint}"
        # 如果visible_objects为空但观测包含典型对象词
        if not st.world_model["visible_objects"]:
            if re.search(r"\b(apple|fridge|counter|cabinet|microwave)\b", observation, re.I):
                return False, "missing visible_objects despite cues in observation"
        return True, "ok"

    @staticmethod
    def _extract_room(text: str) -> Optional[str]:
        # 粗糙房间词表/启发式
        rooms = ["kitchen", "living room", "bedroom", "bathroom", "hallway", "garage"]
        for r in rooms:
            if re.search(rf"\b{re.escape(r)}\b", text, re.I):
                return r
        return None

    @staticmethod
    def _extract_objects(text: str) -> List[str]:
        # 非严格抽取：抽取带编号的名词或常见对象名词（可逐步增强）
        candidates = set()
        for m in re.finditer(r"\b([a-zA-Z]+(?:\s*\d+)?)\b", text):
            token = m.group(1).lower().strip()
            if re.match(r"(apple|fridge|counter|cabinet|microwave|plate|bottle|pan|sink)(?:\s*\d+)?$", token):
                candidates.add(token)
        return sorted(candidates)

    @staticmethod
    def _safe_copy(x: Any) -> Any:
        # 简单深拷贝替代，避免集合在日志里不可序列化
        import copy, json
        try:
            return json.loads(json.dumps(x, default=list))
        except Exception:
            return copy.deepcopy(x)


