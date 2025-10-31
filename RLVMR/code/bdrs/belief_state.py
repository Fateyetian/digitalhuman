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
        "object_states": {},          # {'microwave 1': {'open': True, 'hot': False}, ...}
        "last_observation": "",
        "belief_conflicts": [],       # [{'type': 'location_mismatch', 'obj': 'apple 1', 'step': 3}]
        "belief_corrections": [],     # [{'type': 'location_update', 'obj': 'apple 1', 'step': 5}]
    })
    # P_t: 任务进展（目标/子目标状态）
    task_progress: Dict[str, Any] = field(default_factory=lambda: {
        "task_description": "",
        "subgoals": [],               # [{'text': 'heat apple', 'status': 'pending|completed', 'step_completed': -1}]
        "current_subgoal_idx": 0,
        "completed_count": 0,         # 已完成子目标数量
        "last_completed_step": -1,    # 最后完成子目标的步数
    })
    # E_t: 探索地图（到过哪里/见过什么）
    exploration_map: Dict[str, Any] = field(default_factory=lambda: {
        "visited_rooms": set(),
        "visited_containers": set(),
        "visited_objects": set(),
        "trajectory": [],             # [{'obs': ..., 'act': ...}]
        "room_visit_counts": {},      # {'kitchen': 3, 'living room': 1, ...}
        "new_rooms_this_step": set(), # 本步新发现的房间
        "new_objects_this_step": set(), # 本步新发现的对象
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
                # 自动解析子目标
                subgoals = self._parse_subgoals(task_desc[i])
                st.task_progress["subgoals"] = subgoals
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
        # 清空本步的新发现集合
        st.exploration_map["new_rooms_this_step"] = set()
        st.exploration_map["new_objects_this_step"] = set()

        # room name heuristic
        room = self._extract_room(observation)
        if room:
            st.world_model["current_room"] = room
            # 检查是否是新房间
            if room not in st.exploration_map["visited_rooms"]:
                st.exploration_map["new_rooms_this_step"].add(room)
                st.exploration_map["visited_rooms"].add(room)
                st.exploration_map["room_visit_counts"][room] = 1
            else:
                st.exploration_map["room_visit_counts"][room] = st.exploration_map["room_visit_counts"].get(room, 0) + 1

        # containers/objects heuristics
        for obj in self._extract_objects(observation):
            # 检查是否是新对象
            if obj not in st.exploration_map["visited_objects"]:
                st.exploration_map["new_objects_this_step"].add(obj)
                st.exploration_map["visited_objects"].add(obj)

    def _update_world_model(self, st: BeliefState, observation: str, info: Optional[Dict[str, Any]]):
        # Extract visible objects
        visible = set(self._extract_objects(observation))
        st.world_model["visible_objects"] = visible

        # Extract simple "obj in/on recep" patterns from observation text
        for m in re.finditer(r"(\w+(?: \d+)?) (?:in|on) (\w+(?: \s*\d+)?)", observation):
            obj, recep = m.group(1), m.group(2)
            # 检查是否与之前的信念冲突
            if obj in st.world_model["object_locations"]:
                prev_loc = st.world_model["object_locations"][obj]
                if prev_loc != recep:
                    # 记录冲突和修正
                    st.world_model["belief_conflicts"].append({
                        "type": "location_mismatch",
                        "obj": obj,
                        "prev_belief": prev_loc,
                        "observation": recep,
                        "step": st.meta["step_idx"]
                    })
                    st.world_model["belief_corrections"].append({
                        "type": "location_update",
                        "obj": obj,
                        "from": prev_loc,
                        "to": recep,
                        "step": st.meta["step_idx"]
                    })
            st.world_model["object_locations"][obj] = recep

        # Extract object states (open/closed, hot/cold, clean/dirty)
        for obj in visible:
            if obj not in st.world_model["object_states"]:
                st.world_model["object_states"][obj] = {}

            # Check for state keywords in observation
            if re.search(rf"\b{re.escape(obj)}.*?\b(open|opened)\b", observation, re.I):
                st.world_model["object_states"][obj]["open"] = True
            elif re.search(rf"\b{re.escape(obj)}.*?\b(closed|close)\b", observation, re.I):
                st.world_model["object_states"][obj]["open"] = False

            if re.search(rf"\b{re.escape(obj)}.*?\b(hot|heated)\b", observation, re.I):
                st.world_model["object_states"][obj]["hot"] = True
            elif re.search(rf"\b{re.escape(obj)}.*?\bcold\b", observation, re.I):
                st.world_model["object_states"][obj]["hot"] = False

            if re.search(rf"\b{re.escape(obj)}.*?\bclean\b", observation, re.I):
                st.world_model["object_states"][obj]["clean"] = True
            elif re.search(rf"\b{re.escape(obj)}.*?\bdirty\b", observation, re.I):
                st.world_model["object_states"][obj]["clean"] = False

    def _update_task_progress(self, st: BeliefState, observation: str, action: str, info: Optional[Dict[str, Any]]):
        # 如果还没有子目标，使用默认子目标
        if not st.task_progress["subgoals"]:
            st.task_progress["subgoals"] = [{"text": "complete task", "status": "pending", "step_completed": -1}]

        # 检查当前子目标是否完成
        idx = st.task_progress["current_subgoal_idx"]
        if 0 <= idx < len(st.task_progress["subgoals"]):
            current_goal = st.task_progress["subgoals"][idx]
            if current_goal["status"] == "pending":
                # 检查是否完成 - 基于关键动词和观察反馈
                goal_text = current_goal["text"].lower()
                action_lower = action.lower()
                observation_lower = observation.lower()

                # 启发式：检查动作是否匹配目标关键词
                goal_keywords = re.findall(r'\b(heat|cool|clean|put|take|open|close|use|find|go)\b', goal_text)
                action_verbs = re.findall(r'^\s*(heat|cool|clean|put|take|open|close|use|go to)\b', action_lower)

                # 如果动作包含目标关键词，且观察没有明显的失败信号
                if any(keyword in action_lower for keyword in goal_keywords):
                    # 检查观察是否表明成功（没有"nothing happens"之类的失败信号）
                    failure_signals = ["nothing happens", "can't", "cannot", "unable", "already"]
                    if not any(signal in observation_lower for signal in failure_signals):
                        # 标记为完成
                        st.task_progress["subgoals"][idx]["status"] = "completed"
                        st.task_progress["subgoals"][idx]["step_completed"] = st.meta["step_idx"]
                        st.task_progress["completed_count"] += 1
                        st.task_progress["last_completed_step"] = st.meta["step_idx"]
                        # 移动到下一个子目标
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
        # 扩展对象词表
        common_objects = [
            'apple', 'fridge', 'counter', 'cabinet', 'microwave', 'plate', 'bottle', 'pan', 'sink',
            'table', 'desk', 'shelf', 'drawer', 'box', 'bed', 'sofa', 'chair', 'lamp', 'book',
            'cup', 'mug', 'knife', 'fork', 'spoon', 'pot', 'bowl', 'kettle', 'toaster', 'tv',
            'phone', 'laptop', 'key', 'pen', 'paper', 'cloth', 'towel', 'soap', 'sponge'
        ]
        candidates = set()
        # 抽取带编号的对象
        for m in re.finditer(r"\b([a-zA-Z]+(?:\s*\d+)?)\b", text):
            token = m.group(1).lower().strip()
            # 检查是否匹配常见对象
            for obj_name in common_objects:
                if re.match(rf"{obj_name}(?:\s*\d+)?$", token):
                    candidates.add(token)
                    break
        return sorted(candidates)

    def _parse_subgoals(self, task_description: str) -> List[Dict[str, Any]]:
        """
        从任务描述中自动解析子目标
        例如: "heat apple and put in fridge" -> ["heat apple", "put apple in fridge"]
        """
        subgoals = []
        task_lower = task_description.lower().strip()

        # 尝试按 'and' 分割
        if ' and ' in task_lower:
            parts = task_lower.split(' and ')
            for part in parts:
                part = part.strip()
                if part:
                    subgoals.append({
                        "text": part,
                        "status": "pending",
                        "step_completed": -1
                    })
        # 尝试按逗号分割
        elif ',' in task_lower:
            parts = task_lower.split(',')
            for part in parts:
                part = part.strip()
                if part and part not in ['then', 'and']:
                    subgoals.append({
                        "text": part,
                        "status": "pending",
                        "step_completed": -1
                    })
        # 如果没有明显的分隔符，创建单一子目标
        else:
            subgoals.append({
                "text": task_lower,
                "status": "pending",
                "step_completed": -1
            })

        return subgoals if subgoals else [{"text": "complete task", "status": "pending", "step_completed": -1}]

    @staticmethod
    def _safe_copy(x: Any) -> Any:
        # 简单深拷贝替代，避免集合在日志里不可序列化
        import copy, json
        try:
            return json.loads(json.dumps(x, default=list))
        except Exception:
            return copy.deepcopy(x)


