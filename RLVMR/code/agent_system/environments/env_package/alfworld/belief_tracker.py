"""
ReBel (Reward Belief) Framework - Belief State Tracking and Intrinsic Reward Calculation

This module implements the core components of ReBel:
1. Belief state parsing from model outputs
2. Ground truth state tracking from observation history
3. Three intrinsic rewards: consistency, progress, and exploration efficiency
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class BeliefStateParser:
    """Parse belief states from model output"""

    @staticmethod
    def parse_belief(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract belief state from model output.
        Expected format:
        <belief>
        {
          "world_model_update": {
            "found_objects": {"obj_id": "recep_id"},
            "state_changes": {"obj_id": "new_state"},
            "cleared_receptacles": ["recep1", "recep2"]
          },
          "task_progress_update": {
            "subgoal_status": "completed/in_progress",
            "evidence": "...",
            "updated_subgoal": "..."
          },
          "exploration_map_update": {
            "newly_visited": ["loc1", "loc2"],
            "next_priority": ["loc3", "loc4"]
          }
        }
        </belief>
        """
        belief_match = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL | re.IGNORECASE)
        if not belief_match:
            return None

        belief_text = belief_match.group(1).strip()

        try:
            # Try to parse as JSON first (new format)
            # Handle both single and double quotes
            belief_json = belief_text.replace("'", '"')
            belief_data = json.loads(belief_json)

            # New format: nested JSON structure
            if isinstance(belief_data, dict) and ('world_model_update' in belief_data or 'task_progress_update' in belief_data):
                return {
                    'world_model_update': belief_data.get('world_model_update', {}),
                    'task_progress_update': belief_data.get('task_progress_update', {}),
                    'exploration_map_update': belief_data.get('exploration_map_update', {})
                }

        except json.JSONDecodeError:
            # Fall back to old format parsing: M_t, P_t, E_t
            pass

        # Parse old format: M_t, P_t, E_t
        belief_state = {}

        # Parse M_t (World Model)
        m_match = re.search(r'M_t:\s*(\{.*?\})', belief_text, re.DOTALL)
        if m_match:
            try:
                m_str = m_match.group(1).replace("'", '"')
                belief_state['M_t'] = json.loads(m_str)
            except json.JSONDecodeError:
                belief_state['M_t'] = {}
        else:
            belief_state['M_t'] = {}

        # Parse P_t (Task Progress)
        p_match = re.search(r'P_t:\s*(\{.*?\})', belief_text, re.DOTALL)
        if p_match:
            try:
                p_str = p_match.group(1).replace("'", '"')
                belief_state['P_t'] = json.loads(p_str)
            except json.JSONDecodeError:
                belief_state['P_t'] = {}
        else:
            belief_state['P_t'] = {}

        # Parse E_t (Exploration Map)
        e_match = re.search(r'E_t:\s*(\{.*?\})', belief_text, re.DOTALL)
        if e_match:
            try:
                e_str = e_match.group(1).replace("'", '"')
                belief_state['E_t'] = json.loads(e_str)
            except json.JSONDecodeError:
                belief_state['E_t'] = {}
        else:
            belief_state['E_t'] = {}

        # Convert old format to new format for consistency
        if belief_state.get('M_t') or belief_state.get('P_t') or belief_state.get('E_t'):
            return {
                'world_model_update': {
                    'found_objects': belief_state.get('M_t', {}),
                    'state_changes': {},
                    'cleared_receptacles': []
                },
                'task_progress_update': belief_state.get('P_t', {}),
                'exploration_map_update': belief_state.get('E_t', {})
            }

        return None


class GroundTruthTracker:
    """Track ground truth state from observation history"""

    def __init__(self):
        self.visible_receptacles = set()  # receptacles we can see (world knowledge)
        self.visited_locations = set()    # receptacles we have actually explored/interacted with
        self.object_locations = {}  # object -> location
        self.object_states = {}     # object -> state (open/closed/clean/dirty/hot/cold)
        self.cleared_receptacles = set()  # receptacles that were checked and found empty/irrelevant
        self.interaction_history = []
        self.task_goal = None
        self.initial_location = None
        self.current_inventory = None  # track what agent is currently holding

    def update_from_observation(self, obs: str, action: str = None):
        """Update ground truth from observation text"""
        obs_lower = obs.lower()

        # Extract visible receptacles from initial room description
        # These are "seen" but not yet "visited/explored"
        loc_match = re.search(r'you are in the middle of a room\. looking quickly around you, you see (.*?)\.', obs_lower)
        if loc_match:
            location_desc = loc_match.group(1)
            # Preprocess: remove ", and " to handle last item (e.g., "..., and a drawer 2")
            location_desc = location_desc.replace(', and ', ', ')
            # Extract receptacles without article (a/an)
            # Pattern captures only the object name + number (e.g., "drawer 1" from "a drawer 1")
            receptacles = re.findall(r'(?:a |an )([\w\s]+\d+)', location_desc)
            for recep in receptacles:
                self.visible_receptacles.add(recep.strip())

        # "You are facing X" - this is visible but not visited
        facing_match = re.search(r'you are facing (.*?)\.', obs_lower)
        if facing_match:
            facing = facing_match.group(1).strip()
            self.visible_receptacles.add(facing)

        # Track what's currently in agent's inventory
        # Pattern: "You pick up X from Y" or "You are carrying X"
        pickup_match = re.search(r'you (?:pick up|take) ([\w\s]+\d+)', obs_lower)
        if pickup_match:
            self.current_inventory = pickup_match.group(1).strip()

        # Pattern: "You put X in/on Y"
        put_match = re.search(r'you put ([\w\s]+\d+) (?:in|on) ([\w\s]+\d+)', obs_lower)
        if put_match:
            obj = put_match.group(1).strip()
            location = put_match.group(2).strip()
            self.object_locations[obj] = f"in/on {location}"
            self.current_inventory = None  # no longer holding it

        # Extract object locations from observations
        # Pattern: "On the X, you see Y" or "On the X, you see nothing"
        # This means we actually examined/visited this receptacle
        on_matches = re.findall(r'on the ([\w\s]+\d+), you see (.*?)\.', obs_lower)
        for location, items_str in on_matches:
            location = location.strip()
            self.visited_locations.add(location)  # Actually visited/examined
            self.visible_receptacles.add(location)  # Also visible

            if 'nothing' in items_str:
                # This receptacle is empty
                self.cleared_receptacles.add(location)
            else:
                # Preprocess: remove "and" connector
                items_str = items_str.replace(', and ', ', ').replace(' and ', ', ')
                # Extract items without article (a/an)
                items = re.findall(r'(?:a |an )([\w\s]+\d+)', items_str)
                for item in items:
                    item = item.strip()
                    if item:
                        self.object_locations[item] = f"on {location}"

        # Pattern: "In the X, you see Y" or "In the X, you see nothing"
        # This means we actually opened and examined this receptacle
        in_matches = re.findall(r'in the ([\w\s]+\d+), you see (.*?)\.', obs_lower)
        for location, items_str in in_matches:
            location = location.strip()
            self.visited_locations.add(location)  # Actually visited/examined
            self.visible_receptacles.add(location)  # Also visible

            if 'nothing' in items_str:
                # This receptacle is empty
                self.cleared_receptacles.add(location)
            else:
                # Preprocess: remove "and" connector
                items_str = items_str.replace(', and ', ', ').replace(' and ', ', ')
                # Extract items without article (a/an)
                items = re.findall(r'(?:a |an )([\w\s]+\d+)', items_str)
                for item in items:
                    item = item.strip()
                    if item:
                        self.object_locations[item] = f"in {location}"

        # Extract object states
        # Pattern: "The X is open/closed/clean/dirty/hot/cold"
        state_matches = re.findall(r'the ([\w\s]+\d+) is (open|closed|clean|dirty|hot|cold)', obs_lower)
        for obj, state in state_matches:
            obj = obj.strip()
            self.object_states[obj] = state
            # If we opened something, mark it as visited and visible
            if state == 'open':
                self.visited_locations.add(obj)
                self.visible_receptacles.add(obj)

        # Pattern: "You heat X using Y"
        heat_match = re.search(r'you heat ([\w\s]+\d+)', obs_lower)
        if heat_match:
            obj = heat_match.group(1).strip()
            self.object_states[obj] = 'hot'

        # Pattern: "You clean X using Y"
        clean_match = re.search(r'you clean ([\w\s]+\d+)', obs_lower)
        if clean_match:
            obj = clean_match.group(1).strip()
            self.object_states[obj] = 'clean'

        # Pattern: "You cool X using Y"
        cool_match = re.search(r'you cool ([\w\s]+\d+)', obs_lower)
        if cool_match:
            obj = cool_match.group(1).strip()
            self.object_states[obj] = 'cold'

        # Track interactions
        if action:
            self.interaction_history.append(action.lower())

        # Extract task goal from initial observation
        if self.task_goal is None:
            goal_match = re.search(r'your task is to: (.*?)\.', obs_lower)
            if goal_match:
                self.task_goal = goal_match.group(1).strip()

    def get_ground_truth_state(self) -> Dict[str, Any]:
        """Return current ground truth state"""
        return {
            'visible_receptacles': list(self.visible_receptacles),  # What we can see (world knowledge)
            'visited': list(self.visited_locations),  # What we've actually explored
            'object_locations': dict(self.object_locations),
            'object_states': dict(self.object_states),
            'cleared_receptacles': list(self.cleared_receptacles),
            'current_inventory': self.current_inventory,
            'interactions': self.interaction_history.copy(),
            'task_goal': self.task_goal
        }


class RebelRewardCalculator:
    """Calculate three intrinsic rewards for ReBel framework"""

    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, delta=0.1, use_belief_reward=True,
                 belief_reward_decay_enable=False, belief_reward_decay_method='cosine',
                 belief_reward_warmup_epochs=5, belief_reward_decay_start_epoch=10,
                 belief_reward_decay_end_epoch=60, belief_reward_min_weight=0.1,
                 # V11: Adaptive decay parameters
                 belief_reward_adaptive_decay=False,
                 belief_reward_target_sr=0.90,
                 belief_reward_decay_alpha=2.0,
                 # V11: Differential component decay rates
                 belief_reward_progress_decay_rate=0.7,
                 belief_reward_consistency_decay_rate=1.0,
                 belief_reward_exploration_decay_rate=2.0):
        """
        Args:
            alpha: Weight for consistency reward (environment alignment)
            beta: Weight for progress reward (task understanding)
            gamma: Weight for exploration reward (exploration efficiency)
            delta: Weight for format validity reward (output format compliance)
            use_belief_reward: Whether to use belief-based intrinsic rewards (for ablation studies)
            belief_reward_decay_enable: Whether to enable belief reward decay (V9)
            belief_reward_decay_method: Decay method - 'cosine', 'linear', 'exponential', 'adaptive' (V9/V11)
            belief_reward_warmup_epochs: Number of warmup epochs (V9)
            belief_reward_decay_start_epoch: Epoch to start decay (V9)
            belief_reward_decay_end_epoch: Epoch to end decay (V9)
            belief_reward_min_weight: Minimum weight after decay (V9)
            belief_reward_adaptive_decay: Whether to use success-rate-based adaptive decay (V11)
            belief_reward_target_sr: Target success rate for full decay (V11)
            belief_reward_decay_alpha: Decay curve exponent (V11)
            belief_reward_progress_decay_rate: Differential decay rate for progress (V11, slow=0.7)
            belief_reward_consistency_decay_rate: Differential decay rate for consistency (V11, normal=1.0)
            belief_reward_exploration_decay_rate: Differential decay rate for exploration (V11, fast=2.0)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_belief_reward = use_belief_reward

        # V9: Belief reward decay parameters
        self.belief_reward_decay_enable = belief_reward_decay_enable
        self.belief_reward_decay_method = belief_reward_decay_method
        self.belief_reward_warmup_epochs = belief_reward_warmup_epochs
        self.belief_reward_decay_start_epoch = belief_reward_decay_start_epoch
        self.belief_reward_decay_end_epoch = belief_reward_decay_end_epoch
        self.belief_reward_min_weight = belief_reward_min_weight
        self.current_epoch = 0  # Will be updated by env_manager

        # V11: Adaptive decay parameters
        self.belief_reward_adaptive_decay = belief_reward_adaptive_decay
        self.belief_reward_target_sr = belief_reward_target_sr
        self.belief_reward_decay_alpha = belief_reward_decay_alpha
        self.current_success_rate = None  # Will be updated by env_manager

        # V11: Differential component decay rates
        self.progress_decay_rate = belief_reward_progress_decay_rate
        self.consistency_decay_rate = belief_reward_consistency_decay_rate
        self.exploration_decay_rate = belief_reward_exploration_decay_rate

        # Hyperparameters for reward calculation
        self.consistency_scale = 0.1  # Scale to match task reward magnitude
        self.progress_scale = 0.1
        self.exploration_scale = 0.05

        # Format validity rewards
        self.format_valid_reward = 0.01      # Small bonus for correct format
        self.format_invalid_penalty = -0.05  # Penalty for incorrect format
        self.action_invalid_penalty = -0.02  # Additional penalty for invalid action

    def set_current_epoch(self, epoch: int):
        """V9: Update current epoch for belief reward decay calculation"""
        self.current_epoch = epoch

    def set_success_rate(self, success_rate: float):
        """V11: Update current success rate for adaptive decay"""
        self.current_success_rate = success_rate

    def get_belief_reward_weight(self) -> float:
        """
        V9/V11: Calculate belief reward weight based on current epoch and success rate.

        V9: Fixed schedule (cosine/linear/exponential)
        V11: Adaptive schedule based on success rate + cosine floor as safety net

        Implements multi-phase schedule:
        1. Warmup phase (0 -> warmup_epochs): weight increases from 0 to 1
        2. Peak phase (warmup_epochs -> decay_start_epoch): weight stays at 1
        3. Decay phase: adaptive (SR-based) or fixed schedule
        4. Maintain phase (> decay_end_epoch): weight stays at min_weight

        Returns:
            float: Belief reward weight in [min_weight, 1.0]
        """
        import math

        if not self.belief_reward_decay_enable:
            return 1.0 if self.use_belief_reward else 0.0

        epoch = self.current_epoch
        warmup = self.belief_reward_warmup_epochs
        decay_start = self.belief_reward_decay_start_epoch
        decay_end = self.belief_reward_decay_end_epoch
        min_weight = self.belief_reward_min_weight

        # Phase 1: Warmup (0 -> warmup_epochs)
        if epoch < warmup:
            return epoch / warmup if warmup > 0 else 1.0

        # Base weight after warmup
        base_weight = 1.0

        # V11: Adaptive decay based on success rate
        if self.belief_reward_adaptive_decay and self.current_success_rate is not None and epoch >= warmup:
            sr = self.current_success_rate
            target_sr = self.belief_reward_target_sr
            alpha = self.belief_reward_decay_alpha

            if target_sr > 0:
                decay_factor = max(min_weight, 1.0 - (sr / target_sr) ** alpha)
            else:
                decay_factor = 1.0
            base_weight = base_weight * decay_factor

        # Phase 2: Peak (warmup_epochs -> decay_start_epoch) — only if no adaptive decay
        if not self.belief_reward_adaptive_decay and epoch < decay_start:
            return 1.0

        # Phase 3: Cosine floor (safety net, applies regardless of adaptive mode)
        if epoch > decay_start:
            progress = (epoch - decay_start) / (decay_end - decay_start)
            progress = min(1.0, max(0.0, progress))

            if self.belief_reward_decay_method == 'cosine' or self.belief_reward_adaptive_decay:
                cosine_weight = min_weight + (1.0 - min_weight) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.belief_reward_decay_method == 'linear':
                cosine_weight = 1.0 - progress * (1.0 - min_weight)
            elif self.belief_reward_decay_method == 'exponential':
                cosine_weight = min_weight + (1.0 - min_weight) * math.exp(-3 * progress)
            else:
                cosine_weight = min_weight + (1.0 - min_weight) * 0.5 * (1 + math.cos(math.pi * progress))

            cosine_weight = max(min_weight, cosine_weight)

            if self.belief_reward_adaptive_decay:
                # Take the more aggressive (lower) of adaptive and cosine floor
                base_weight = min(base_weight, cosine_weight)
            else:
                base_weight = cosine_weight

        return max(min_weight, base_weight)

    def compute_component_weights(self, base_weight: float) -> Dict[str, float]:
        """
        V11: Compute per-component weights with differential decay rates.

        Different reward components have different alignment with the task objective:
        - Progress (most aligned): decays slowest (rate=0.7)
        - Consistency (moderately aligned): normal decay (rate=1.0)
        - Exploration (least aligned, potentially harmful): decays fastest (rate=2.0)
        - Format: never decays (structural necessity)

        The formula: component_weight = base_weight ^ decay_rate
        When base_weight = 0.5:
          progress:    0.5^0.7 = 0.62 (retains 62%)
          consistency: 0.5^1.0 = 0.50 (retains 50%)
          exploration: 0.5^2.0 = 0.25 (retains only 25%)

        Args:
            base_weight: Overall belief reward weight from get_belief_reward_weight()

        Returns:
            Dict with keys 'progress', 'consistency', 'exploration', 'format'
        """
        if base_weight <= 0:
            return {
                'progress': 0.0,
                'consistency': 0.0,
                'exploration': 0.0,
                'format': 1.0,
            }

        # Clamp to avoid math domain errors
        bw = max(1e-8, min(1.0, base_weight))

        return {
            'progress': bw ** self.progress_decay_rate,
            'consistency': bw ** self.consistency_decay_rate,
            'exploration': bw ** self.exploration_decay_rate,
            'format': 1.0,  # Format never decays
        }

    def calculate_consistency_reward(
        self,
        belief_world: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int = -1  # V6新增: 当前步数，用于早期宽松验证
    ) -> float:
        """
        Calculate environment consistency reward (r_consistency)
        Measures alignment between world_model_update and ground truth world state

        V6改进: 当 ground_truth 信息不足时（早期阶段或解析不完整），
        使用结构化奖励而非严格验证，避免正确预测被错误惩罚

        Args:
            belief_world: world_model_update from model
                {
                    "found_objects": {"obj_id": "recep_id"},
                    "state_changes": {"obj_id": "new_state"},
                    "cleared_receptacles": ["recep1", "recep2"]
                }
            ground_truth: Actual environment state
            step: Current step number (V6新增)

        Returns:
            Consistency reward in [0, 1] range
        """
        if not belief_world:
            return 0.0

        score = 0.0
        components = []

        # V6改进: 检查 ground_truth 是否足够完整
        gt_objects = ground_truth.get('object_locations', {})
        if not isinstance(gt_objects, dict):
            gt_objects = {}

        # V6: 如果是早期阶段(step < 3)或 ground_truth 信息不足(< 3个物体)
        # 使用结构化奖励而非严格验证
        use_lenient_mode = (step >= 0 and step < 3) or len(gt_objects) < 3

        # Component 1: Found objects accuracy (40% weight)
        found_objects = belief_world.get('found_objects', {}) or {}
        # Ensure found_objects is a dict, not a string
        if not isinstance(found_objects, dict):
            found_objects = {}

        if use_lenient_mode:
            # V6: 宽松模式 - 奖励结构化的 belief 输出
            if found_objects and len(found_objects) > 0:
                # 有合理的物体预测，给予中性偏正的分数
                components.append(0.6)
            else:
                components.append(0.4)  # 没有预测也不过度惩罚
        elif found_objects:
            correct = 0
            total = 0
            false_claims = 0

            for obj, claimed_loc in found_objects.items():
                if obj is None or claimed_loc is None:
                    continue
                if not isinstance(claimed_loc, str):
                    continue
                obj_normalized = obj.lower().strip()
                total += 1

                # Check if this object exists in ground truth
                matched = False
                for gt_obj, gt_loc in gt_objects.items():
                    if not isinstance(gt_loc, str):
                        continue
                    if obj_normalized in gt_obj or gt_obj in obj_normalized:
                        matched = True
                        # Check if location matches
                        if claimed_loc.lower() in gt_loc.lower() or gt_loc.lower() in claimed_loc.lower():
                            correct += 1
                        break

                if not matched:
                    false_claims += 1

            if total > 0:
                accuracy = correct / total
                penalty = min(1.0, false_claims * 0.2)
                components.append(max(0.0, accuracy - penalty))
            else:
                components.append(0.5)  # Neutral score
        else:
            components.append(0.5)  # Neutral score if no objects claimed

        # Component 2: State changes accuracy (30% weight)
        state_changes = belief_world.get('state_changes', {}) or {}
        # Ensure state_changes is a dict, not a string
        if not isinstance(state_changes, dict):
            state_changes = {}

        if use_lenient_mode:
            # V6: 宽松模式
            if state_changes and len(state_changes) > 0:
                components.append(0.6)
            else:
                components.append(0.5)  # 中性分数
        elif state_changes:
            gt_states = ground_truth.get('object_states', {})
            if not isinstance(gt_states, dict):
                gt_states = {}

            correct_states = 0
            total_states = 0
            false_state_claims = 0

            for obj, claimed_state in state_changes.items():
                if obj is None or claimed_state is None:
                    continue
                if not isinstance(claimed_state, str):
                    continue
                obj_normalized = obj.lower().strip()
                total_states += 1

                matched = False
                for gt_obj, gt_state in gt_states.items():
                    if not isinstance(gt_state, str):
                        continue
                    if obj_normalized in gt_obj or gt_obj in obj_normalized:
                        matched = True
                        if claimed_state.lower() == gt_state.lower():
                            correct_states += 1
                        break

                if not matched:
                    false_state_claims += 1

            if total_states > 0:
                state_accuracy = correct_states / total_states
                state_penalty = min(1.0, false_state_claims * 0.2)
                components.append(max(0.0, state_accuracy - state_penalty))
            else:
                components.append(0.5)
        else:
            components.append(0.5)

        # Component 3: Cleared receptacles accuracy (30% weight)
        cleared_receptacles = belief_world.get('cleared_receptacles', []) or []
        # Ensure cleared_receptacles is a list, not a string
        if isinstance(cleared_receptacles, str):
            cleared_receptacles = [cleared_receptacles] if cleared_receptacles else []
        elif not isinstance(cleared_receptacles, list):
            cleared_receptacles = []

        if use_lenient_mode:
            # V6: 宽松模式
            if cleared_receptacles:
                components.append(0.6)
            else:
                components.append(0.5)
        elif cleared_receptacles:
            gt_cleared_raw = ground_truth.get('cleared_receptacles', [])
            if isinstance(gt_cleared_raw, str):
                gt_cleared_raw = [gt_cleared_raw] if gt_cleared_raw else []
            elif not isinstance(gt_cleared_raw, list):
                gt_cleared_raw = []
            gt_cleared = set(r.lower().strip() for r in gt_cleared_raw if r is not None and isinstance(r, str))

            claimed_cleared = set(r.lower().strip() for r in cleared_receptacles if r is not None and isinstance(r, str))

            if gt_cleared:
                # True positives
                correct_cleared = len(claimed_cleared & gt_cleared)
                # False positives (claimed empty but not actually empty)
                false_cleared = len(claimed_cleared - gt_cleared)

                precision = correct_cleared / len(claimed_cleared) if claimed_cleared else 0.0
                recall = correct_cleared / len(gt_cleared) if gt_cleared else 0.0

                # F1 score with penalty for false positives
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    cleared_score = max(0.0, f1 - false_cleared * 0.1)
                    components.append(cleared_score)
                else:
                    components.append(0.0)
            else:
                # No ground truth cleared receptacles yet
                # Small penalty for claiming cleared when none exist
                components.append(max(0.0, 0.5 - len(claimed_cleared) * 0.1))
        else:
            components.append(0.5)  # Neutral score if none claimed

        # Weighted combination: 40%, 30%, 30%
        score = 0.4 * components[0] + 0.3 * components[1] + 0.3 * components[2]

        return score * self.consistency_scale

    def calculate_progress_reward(
        self,
        belief_progress: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int,
        done: bool,
        success: bool,
        task_type: str = None,
        world_model: Dict[str, Any] = None,
        prev_world_model: Dict[str, Any] = None
    ) -> float:
        """
        Calculate task progress reward (r_progress)
        Measures accuracy of task understanding and progress tracking
        V4增强: 支持task_type和状态改变检测

        Args:
            belief_progress: task_progress_update from model
                {
                    "subgoal_status": "completed/in_progress",
                    "evidence": "...",
                    "updated_subgoal": "..."
                }
            ground_truth: Actual environment state
            step: Current step number
            done: Whether episode is done
            success: Whether task succeeded
            task_type: 任务类型 (V4新增)
            world_model: 当前world_model_update (V4新增)
            prev_world_model: 上一步world_model_update (V4新增)

        Returns:
            Progress reward in [0, 1] range
        """
        if not belief_progress:
            return 0.0

        score = 0.0

        # Component 1: Subgoal status accuracy (40% weight)
        subgoal_status_score = 0.0
        if 'subgoal_status' in belief_progress and belief_progress['subgoal_status'] is not None:
            status = belief_progress['subgoal_status']
            if not isinstance(status, str):
                status = str(status)
            status = status.lower()

            if done and success:
                # Task completed successfully
                subgoal_status_score = 1.0 if 'complete' in status else 0.0
            elif done and not success:
                # Task failed
                subgoal_status_score = 1.0 if 'incomplete' in status or 'fail' in status else 0.5
            else:
                # Task in progress
                subgoal_status_score = 1.0 if 'progress' in status or 'incomplete' in status else 0.7

        # Component 2: Evidence quality (30% weight)
        evidence_score = 0.0
        if 'evidence' in belief_progress:
            evidence = belief_progress['evidence']
            if not isinstance(evidence, str):
                evidence = str(evidence) if evidence else ""

            # Evidence should be non-empty and substantive
            if evidence and len(evidence.strip()) > 10:
                evidence_score = 0.8

                # Bonus: evidence mentions specific objects or locations
                # Extract recent observations to check if evidence refers to them
                obj_locations = ground_truth.get('object_locations', {})
                if not isinstance(obj_locations, dict):
                    obj_locations = {}
                recent_objects = list(obj_locations.keys())[-5:]
                visited_list = ground_truth.get('visited', [])
                if not isinstance(visited_list, list):
                    visited_list = []
                recent_locations = visited_list[-5:]

                for obj in recent_objects:
                    if obj.lower() in evidence.lower():
                        evidence_score = min(1.0, evidence_score + 0.1)
                        break

                for loc in recent_locations:
                    if loc.lower() in evidence.lower():
                        evidence_score = min(1.0, evidence_score + 0.1)
                        break
            elif evidence:
                evidence_score = 0.3  # Has evidence but too short
            else:
                evidence_score = 0.0  # No evidence provided

        # Component 3: Updated subgoal reasonableness (30% weight)
        subgoal_score = 0.5  # Default neutral score
        if 'updated_subgoal' in belief_progress:
            updated_subgoal = belief_progress['updated_subgoal']
            if not isinstance(updated_subgoal, str):
                updated_subgoal = str(updated_subgoal) if updated_subgoal else ""

            if updated_subgoal and updated_subgoal.lower() != 'null' and len(updated_subgoal.strip()) > 5:
                # Has a substantive subgoal
                subgoal_score = 0.7

                # Check if subgoal is related to task goal
                task_goal = ground_truth.get('task_goal', '')
                if not isinstance(task_goal, str):
                    task_goal = str(task_goal) if task_goal else ""
                if task_goal:
                    # Extract key words from task goal
                    task_keywords = set(task_goal.lower().split())
                    subgoal_keywords = set(updated_subgoal.lower().split())

                    # Check overlap
                    overlap = len(task_keywords & subgoal_keywords)
                    if overlap > 0:
                        subgoal_score = min(1.0, 0.7 + overlap * 0.1)

        # V4新增: Component 4 - 状态改变检测 (对heat/cool/clean任务关键)
        # V6改进: 增加 look_at 任务专用检测
        state_change_bonus = 0.0
        if task_type and world_model:
            task_lower = str(task_type).lower()

            # V6新增: look_at_obj_in_light 任务专用进度检测
            if 'look_at' in task_lower:
                state_change_bonus = self._calculate_look_at_progress(
                    belief_progress, world_model, step, done, success
                )
            else:
                # 原有的 heat/cool/clean 检测
                curr_state_changes = world_model.get('state_changes', {}) or {}
                prev_state_changes = (prev_world_model or {}).get('state_changes', {}) or {}

                if isinstance(curr_state_changes, dict):
                    for obj, state in curr_state_changes.items():
                        state_lower = str(state).lower()
                        prev_state = str(prev_state_changes.get(obj, '')).lower() if prev_state_changes else ''

                        # 检测是否有新的状态改变
                        if state_lower != prev_state:
                            if 'heat' in task_lower and any(x in state_lower for x in ['heated', 'hot', 'cooked']):
                                state_change_bonus = 0.3  # 完成了加热
                            elif 'cool' in task_lower and any(x in state_lower for x in ['cooled', 'cold', 'chilled']):
                                state_change_bonus = 0.3  # 完成了冷却
                            elif 'clean' in task_lower and any(x in state_lower for x in ['cleaned', 'clean', 'washed']):
                                state_change_bonus = 0.3  # 完成了清洁

        # Weighted combination: 35%, 25%, 25%, 15% (调整权重以容纳新组件)
        if state_change_bonus > 0:
            score = (
                0.35 * subgoal_status_score +
                0.25 * evidence_score +
                0.25 * subgoal_score +
                0.15 * (state_change_bonus / 0.3)  # 归一化到 [0,1]
            )
        else:
            score = (
                0.4 * subgoal_status_score +
                0.3 * evidence_score +
                0.3 * subgoal_score
            )

        return score * self.progress_scale

    def _calculate_look_at_progress(
        self,
        belief_progress: Dict[str, Any],
        world_model: Dict[str, Any],
        step: int,
        done: bool,
        success: bool
    ) -> float:
        """
        V6新增: look_at_obj_in_light 任务专用进度检测

        该任务的关键阶段:
        1. 找到目标物体并拾取 (+0.15)
        2. 找到灯光源 (+0.2)
        3. 使用灯检查物体 (+0.25)
        4. 任务完成 (+0.4)

        Args:
            belief_progress: task_progress_update
            world_model: world_model_update
            step: 当前步数
            done: 是否结束
            success: 是否成功

        Returns:
            bonus: [0, 1.0]
        """
        bonus = 0.0
        subgoal = str(belief_progress.get('updated_subgoal', '')).lower()

        # 阶段1: 拾取目标物体 (+0.15)
        # 检查 inventory 是否有物品
        inventory = world_model.get('inventory', []) or []
        if isinstance(inventory, str):
            inventory = [inventory] if inventory else []
        if inventory and len(inventory) > 0:
            bonus += 0.15

        # 阶段2: 找到灯光源 (+0.2)
        # 检查 found_objects 中是否有 lamp/light
        found_objects = world_model.get('found_objects', {}) or {}
        if isinstance(found_objects, dict):
            has_lamp = any('lamp' in str(k).lower() or 'light' in str(k).lower()
                          for k in found_objects.keys())
            # 也检查 subgoal 中是否提到找到了灯
            has_lamp = has_lamp or any(x in subgoal for x in ['found lamp', 'lamp found', 'see lamp', 'lamp is'])
            if has_lamp:
                bonus += 0.2

        # 阶段3: 使用灯检查物体 (+0.25)
        # 检查 subgoal 是否包含使用灯的关键词
        if any(x in subgoal for x in ['turn on', 'use lamp', 'examine', 'look at', 'using lamp']):
            bonus += 0.25

        # 阶段4: 任务完成 (+0.4)
        if done and success:
            bonus += 0.4

        return min(1.0, bonus)  # 限制最大值为1.0

    def calculate_exploration_reward(
        self,
        belief_exploration: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int
    ) -> float:
        """
        Calculate exploration efficiency reward (r_exploration)
        Measures quality of exploration tracking

        Args:
            belief_exploration: exploration_map_update from model
                {
                    "newly_visited": ["loc1", "loc2"],
                    "next_priority": ["loc3", "loc4"]
                }
            ground_truth: Actual environment state
            step: Current step number

        Returns:
            Exploration reward in [0, 1] range
        """
        if not belief_exploration:
            return 0.0

        score = 0.0

        # Component 1: Newly visited accuracy (70% weight)
        newly_visited_score = 0.0
        if 'newly_visited' in belief_exploration:
            newly_visited_list = belief_exploration['newly_visited'] or []
            # Ensure newly_visited_list is a list
            if isinstance(newly_visited_list, str):
                newly_visited_list = [newly_visited_list] if newly_visited_list else []
            elif not isinstance(newly_visited_list, list):
                newly_visited_list = []
            claimed_newly_visited = set(v.lower().strip() for v in newly_visited_list if v is not None and isinstance(v, str))
            visited_list = ground_truth.get('visited', []) or []
            # Ensure visited_list is a list
            if isinstance(visited_list, str):
                visited_list = [visited_list] if visited_list else []
            elif not isinstance(visited_list, list):
                visited_list = []
            actual_visited = set(v.lower().strip() for v in visited_list if v is not None and isinstance(v, str))

            if actual_visited:
                # Check if claimed newly visited are actually in the visited set
                correct_visits = len(claimed_newly_visited & actual_visited)
                false_visits = len(claimed_newly_visited - actual_visited)

                if claimed_newly_visited:
                    precision = correct_visits / len(claimed_newly_visited)
                    # Recall: how many of the actual visited locations did we claim this turn
                    # (This is approximate since we don't know exactly which were visited THIS turn)
                    recall = min(1.0, correct_visits / max(1, len(claimed_newly_visited)))

                    # F1 score
                    if precision + recall > 0:
                        newly_visited_score = 2 * (precision * recall) / (precision + recall)

                    # Penalty for false claims
                    newly_visited_score = max(0.0, newly_visited_score - false_visits * 0.15)
                else:
                    newly_visited_score = 0.5  # Neutral if nothing claimed
            else:
                # No visits yet, but claimed some - small penalty
                newly_visited_score = max(0.0, 0.5 - len(claimed_newly_visited) * 0.1)

        # Component 2: Next priority planning (30% weight)
        next_priority_score = 0.5  # Default neutral
        if 'next_priority' in belief_exploration:
            next_priority = belief_exploration['next_priority'] or []
            # Ensure next_priority is a list
            if isinstance(next_priority, str):
                next_priority = [next_priority] if next_priority else []
            elif not isinstance(next_priority, list):
                next_priority = []

            if next_priority and len(next_priority) > 0:
                # Reward for having a plan
                next_priority_score = 0.7

                # Bonus: next priority should be locations not yet visited
                visited_list = ground_truth.get('visited', []) or []
                # Ensure visited_list is a list
                if isinstance(visited_list, str):
                    visited_list = [visited_list] if visited_list else []
                elif not isinstance(visited_list, list):
                    visited_list = []
                actual_visited = set(v.lower().strip() for v in visited_list if v is not None and isinstance(v, str))
                next_priority_set = set(p.lower().strip() for p in next_priority if p is not None and isinstance(p, str))

                # Good if prioritizing unvisited locations
                unvisited_priorities = next_priority_set - actual_visited
                if unvisited_priorities:
                    ratio = len(unvisited_priorities) / len(next_priority_set)
                    next_priority_score = min(1.0, 0.7 + ratio * 0.3)
            else:
                # No next priority planning
                next_priority_score = 0.3

        # Weighted combination: 70%, 30%
        score = (
            0.7 * newly_visited_score +
            0.3 * next_priority_score
        )

        return score * self.exploration_scale

    def calculate_format_reward(
        self,
        is_format_valid: bool,
        is_action_available: bool
    ) -> float:
        """
        Calculate format validity reward (r_format)
        Encourages the model to follow the correct output format

        Args:
            is_format_valid: Whether the output format is valid (has proper <belief> and <action> tags)
            is_action_available: Whether the action is in the admissible actions list

        Returns:
            Format reward (can be positive or negative)
        """
        reward = 0.0

        # Reward/penalty for format validity
        if is_format_valid:
            reward += self.format_valid_reward
        else:
            reward += self.format_invalid_penalty

        # Additional penalty if action is not in admissible actions
        # (only matters if format was valid enough to extract an action)
        if is_format_valid and not is_action_available:
            reward += self.action_invalid_penalty

        return reward

    def calculate_total_intrinsic_reward(
        self,
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int,
        done: bool,
        success: bool,
        is_format_valid: bool = True,
        is_action_available: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total intrinsic reward from all four components

        Args:
            belief_state: Parsed belief state from model
                {
                    "world_model_update": {...},
                    "task_progress_update": {...},
                    "exploration_map_update": {...}
                }
            ground_truth: Ground truth environment state
            step: Current step number
            done: Whether episode is done
            success: Whether task succeeded
            is_format_valid: Whether output format is valid
            is_action_available: Whether action is in admissible actions

        Returns:
            total_reward: Weighted sum of four rewards
            breakdown: Dict with individual reward values
        """
        world_model = belief_state.get('world_model_update', {})
        if not isinstance(world_model, dict):
            world_model = {}
        task_progress = belief_state.get('task_progress_update', {})
        if not isinstance(task_progress, dict):
            task_progress = {}
        exploration_map = belief_state.get('exploration_map_update', {})
        if not isinstance(exploration_map, dict):
            exploration_map = {}

        # V6: 传递 step 参数给 consistency_reward
        r_consistency = self.calculate_consistency_reward(world_model, ground_truth, step)
        r_progress = self.calculate_progress_reward(task_progress, ground_truth, step, done, success)
        r_exploration = self.calculate_exploration_reward(exploration_map, ground_truth, step)
        r_format = self.calculate_format_reward(is_format_valid, is_action_available)

        # V9: Get belief reward weight (supports decay schedule)
        belief_weight = self.get_belief_reward_weight()

        # V8 Ablation: Support disabling belief rewards (backward compatible)
        if not self.use_belief_reward and not self.belief_reward_decay_enable:
            # Ablation: Only use format reward (no belief-based intrinsic rewards)
            belief_weight = 0.0

        # V11: Compute per-component weights with differential decay
        component_weights = self.compute_component_weights(belief_weight)

        # V11: Apply differential component weights
        r_consistency_weighted = r_consistency * component_weights['consistency']
        r_progress_weighted = r_progress * component_weights['progress']
        r_exploration_weighted = r_exploration * component_weights['exploration']

        # Weighted combination with differentially decayed belief rewards
        total_reward = (
            self.alpha * r_consistency_weighted +
            self.beta * r_progress_weighted +
            self.gamma * r_exploration_weighted +
            self.delta * r_format
        )

        breakdown = {
            'r_consistency': r_consistency,
            'r_progress': r_progress,
            'r_exploration': r_exploration,
            'r_format': r_format,
            'r_intrinsic_total': total_reward,
            # V9: Add belief weight tracking for monitoring
            'belief_weight': belief_weight,
            'r_consistency_weighted': r_consistency_weighted,
            'r_progress_weighted': r_progress_weighted,
            'r_exploration_weighted': r_exploration_weighted,
            'current_epoch': self.current_epoch,
            # V11: Component weights for monitoring
            'component_weight_progress': component_weights['progress'],
            'component_weight_consistency': component_weights['consistency'],
            'component_weight_exploration': component_weights['exploration'],
        }

        return total_reward, breakdown


# ============================================================================ #
# ====================== Belief Deviation Metrics ============================ #
# ============================================================================ #

class BeliefDeviationCalculator:
    """
    Calculate belief deviation metrics between model predictions and ground truth.

    This class computes various metrics to measure how well the model's belief state
    aligns with the actual environment state. These metrics are useful for:
    1. Monitoring model's world understanding during training
    2. Analyzing failure cases
    3. Paper experiments and ablation studies
    """

    @staticmethod
    def compute_object_location_deviation(
        belief_world: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute deviation in object location predictions.

        Args:
            belief_world: world_model_update from model prediction
            ground_truth: actual environment state

        Returns:
            Dict with precision, recall, f1, and deviation score
        """
        # Extract predicted and actual object locations
        predicted = belief_world.get('found_objects', {}) or {}
        if not isinstance(predicted, dict):
            predicted = {}

        actual = ground_truth.get('object_locations', {}) or {}
        if not isinstance(actual, dict):
            actual = {}

        if not predicted and not actual:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'deviation': 0.0}

        if not predicted:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'deviation': 1.0}

        if not actual:
            # Model predicted objects but none exist - penalize false positives
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'deviation': 1.0}

        # Normalize keys for comparison
        pred_normalized = {k.lower().strip(): v.lower().strip() if isinstance(v, str) else str(v)
                          for k, v in predicted.items() if k and v}
        actual_normalized = {k.lower().strip(): v.lower().strip() if isinstance(v, str) else str(v)
                           for k, v in actual.items() if k and v}

        # Calculate matches
        true_positives = 0
        for obj, loc in pred_normalized.items():
            for actual_obj, actual_loc in actual_normalized.items():
                if (obj in actual_obj or actual_obj in obj):
                    if (loc in actual_loc or actual_loc in loc):
                        true_positives += 1
                        break

        precision = true_positives / len(pred_normalized) if pred_normalized else 0.0
        recall = true_positives / len(actual_normalized) if actual_normalized else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Deviation is inverse of F1
        deviation = 1.0 - f1

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'deviation': deviation
        }

    @staticmethod
    def compute_state_deviation(
        belief_world: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute deviation in object state predictions (hot/cold/clean/dirty/open/closed).

        Args:
            belief_world: world_model_update from model prediction
            ground_truth: actual environment state

        Returns:
            Dict with accuracy and deviation score
        """
        predicted_states = belief_world.get('state_changes', {}) or {}
        if not isinstance(predicted_states, dict):
            predicted_states = {}

        actual_states = ground_truth.get('object_states', {}) or {}
        if not isinstance(actual_states, dict):
            actual_states = {}

        if not predicted_states and not actual_states:
            return {'accuracy': 1.0, 'deviation': 0.0, 'n_predictions': 0}

        if not predicted_states:
            return {'accuracy': 0.0, 'deviation': 1.0 if actual_states else 0.0, 'n_predictions': 0}

        # Check accuracy of state predictions
        correct = 0
        total = len(predicted_states)

        for obj, pred_state in predicted_states.items():
            if not isinstance(pred_state, str):
                continue
            obj_lower = obj.lower().strip()
            pred_state_lower = pred_state.lower().strip()

            for actual_obj, actual_state in actual_states.items():
                if not isinstance(actual_state, str):
                    continue
                if obj_lower in actual_obj.lower() or actual_obj.lower() in obj_lower:
                    if pred_state_lower == actual_state.lower().strip():
                        correct += 1
                    break

        accuracy = correct / total if total > 0 else 0.0
        deviation = 1.0 - accuracy

        return {
            'accuracy': accuracy,
            'deviation': deviation,
            'n_predictions': total
        }

    @staticmethod
    def compute_exploration_deviation(
        belief_exploration: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute deviation in exploration tracking (visited locations).

        Args:
            belief_exploration: exploration_map_update from model prediction
            ground_truth: actual environment state

        Returns:
            Dict with precision, recall, f1, and deviation score
        """
        # Handle case where belief_exploration is not a dict
        if not isinstance(belief_exploration, dict):
            if isinstance(belief_exploration, list):
                # If it's a list, treat it as newly_visited
                belief_exploration = {'newly_visited': belief_exploration}
            else:
                belief_exploration = {}

        predicted_visited = belief_exploration.get('newly_visited', []) or []
        if isinstance(predicted_visited, str):
            predicted_visited = [predicted_visited] if predicted_visited else []
        elif not isinstance(predicted_visited, list):
            predicted_visited = []

        actual_visited = ground_truth.get('visited', []) or []
        if isinstance(actual_visited, str):
            actual_visited = [actual_visited] if actual_visited else []
        elif not isinstance(actual_visited, list):
            actual_visited = []

        if not predicted_visited and not actual_visited:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'deviation': 0.0}

        # Normalize for comparison
        pred_set = set(v.lower().strip() for v in predicted_visited if v and isinstance(v, str))
        actual_set = set(v.lower().strip() for v in actual_visited if v and isinstance(v, str))

        if not pred_set:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'deviation': 1.0}

        if not actual_set:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'deviation': 1.0}

        # Calculate intersection
        true_positives = len(pred_set & actual_set)

        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(actual_set) if actual_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        deviation = 1.0 - f1

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'deviation': deviation
        }

    @staticmethod
    def compute_total_belief_deviation(
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Compute overall belief deviation score combining all components.

        Args:
            belief_state: Full belief state from model
                {
                    "world_model_update": {...},
                    "task_progress_update": {...},
                    "exploration_map_update": {...}
                }
            ground_truth: Actual environment state
            weights: Optional weights for each component (default: equal weights)

        Returns:
            Dict with component deviations and total weighted deviation
        """
        if weights is None:
            weights = {
                'object_location': 0.4,
                'state': 0.3,
                'exploration': 0.3
            }

        # Handle None or invalid belief_state
        if not belief_state or not isinstance(belief_state, dict):
            return {
                'object_location': {'deviation': 1.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'state': {'deviation': 1.0, 'accuracy': 0.0, 'n_predictions': 0},
                'exploration': {'deviation': 1.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'total_deviation': 1.0,
                'belief_valid': False
            }

        world_model = belief_state.get('world_model_update', {}) or {}
        if not isinstance(world_model, dict):
            world_model = {}
        exploration_map = belief_state.get('exploration_map_update', {}) or {}
        if not isinstance(exploration_map, dict):
            exploration_map = {}

        # Compute component deviations
        obj_loc_metrics = BeliefDeviationCalculator.compute_object_location_deviation(
            world_model, ground_truth
        )
        state_metrics = BeliefDeviationCalculator.compute_state_deviation(
            world_model, ground_truth
        )
        exploration_metrics = BeliefDeviationCalculator.compute_exploration_deviation(
            exploration_map, ground_truth
        )

        # Compute weighted total deviation
        total_deviation = (
            weights['object_location'] * obj_loc_metrics['deviation'] +
            weights['state'] * state_metrics['deviation'] +
            weights['exploration'] * exploration_metrics['deviation']
        )

        return {
            'object_location': obj_loc_metrics,
            'state': state_metrics,
            'exploration': exploration_metrics,
            'total_deviation': total_deviation,
            'belief_valid': True
        }


# Factory function for easy import
def create_rebel_tracker():
    """Create ReBel components"""
    parser = BeliefStateParser()
    calculator = RebelRewardCalculator(alpha=0.3, beta=0.5, gamma=0.2, delta=0.1)
    return parser, calculator


def create_belief_deviation_calculator():
    """Create BeliefDeviationCalculator instance"""
    return BeliefDeviationCalculator()
