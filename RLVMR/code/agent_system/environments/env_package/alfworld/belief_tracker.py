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
            if 'world_model_update' in belief_data or 'task_progress_update' in belief_data:
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
        self.visited_locations = set()
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

        # Extract current location
        loc_match = re.search(r'you are in the middle of a room\. looking quickly around you, you see (.*?)\.', obs_lower)
        if loc_match:
            location_desc = loc_match.group(1)
            # Extract receptacles as locations
            receptacles = re.findall(r'((?:a |an )?[\w\s]+\d+)', location_desc)
            for recep in receptacles:
                self.visited_locations.add(recep.strip())

        # Extract location from "You are facing ..."
        facing_match = re.search(r'you are facing (.*?)\.', obs_lower)
        if facing_match:
            facing = facing_match.group(1).strip()
            self.visited_locations.add(facing)

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
        on_matches = re.findall(r'on the ([\w\s]+\d+), you see (.*?)\.', obs_lower)
        for location, items_str in on_matches:
            location = location.strip()
            self.visited_locations.add(location)

            if 'nothing' in items_str:
                # This receptacle is empty
                self.cleared_receptacles.add(location)
            else:
                items = re.findall(r'((?:a |an )?[\w\s]+\d+)', items_str)
                for item in items:
                    item = item.strip()
                    if item:
                        self.object_locations[item] = f"on {location}"

        # Pattern: "In the X, you see Y" or "In the X, you see nothing"
        in_matches = re.findall(r'in the ([\w\s]+\d+), you see (.*?)\.', obs_lower)
        for location, items_str in in_matches:
            location = location.strip()
            self.visited_locations.add(location)

            if 'nothing' in items_str:
                # This receptacle is empty
                self.cleared_receptacles.add(location)
            else:
                items = re.findall(r'((?:a |an )?[\w\s]+\d+)', items_str)
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
            # If we opened something, mark it as visited
            if state == 'open':
                self.visited_locations.add(obj)

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
            'visited': list(self.visited_locations),
            'object_locations': dict(self.object_locations),
            'object_states': dict(self.object_states),
            'cleared_receptacles': list(self.cleared_receptacles),
            'current_inventory': self.current_inventory,
            'interactions': self.interaction_history.copy(),
            'task_goal': self.task_goal
        }


class RebelRewardCalculator:
    """Calculate three intrinsic rewards for ReBel framework"""

    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, delta=0.1):
        """
        Args:
            alpha: Weight for consistency reward (environment alignment)
            beta: Weight for progress reward (task understanding)
            gamma: Weight for exploration reward (exploration efficiency)
            delta: Weight for format validity reward (output format compliance)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Hyperparameters for reward calculation
        self.consistency_scale = 0.1  # Scale to match task reward magnitude
        self.progress_scale = 0.1
        self.exploration_scale = 0.05

        # Format validity rewards
        self.format_valid_reward = 0.01      # Small bonus for correct format
        self.format_invalid_penalty = -0.05  # Penalty for incorrect format
        self.action_invalid_penalty = -0.02  # Additional penalty for invalid action

    def calculate_consistency_reward(
        self,
        belief_world: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Calculate environment consistency reward (r_consistency)
        Measures alignment between world_model_update and ground truth world state

        Args:
            belief_world: world_model_update from model
                {
                    "found_objects": {"obj_id": "recep_id"},
                    "state_changes": {"obj_id": "new_state"},
                    "cleared_receptacles": ["recep1", "recep2"]
                }
            ground_truth: Actual environment state

        Returns:
            Consistency reward in [0, 1] range
        """
        if not belief_world:
            return 0.0

        score = 0.0
        components = []

        # Component 1: Found objects accuracy (40% weight)
        found_objects = belief_world.get('found_objects', {}) or {}
        # Ensure found_objects is a dict, not a string
        if not isinstance(found_objects, dict):
            found_objects = {}
        if found_objects:
            gt_objects = ground_truth.get('object_locations', {})
            if not isinstance(gt_objects, dict):
                gt_objects = {}

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
        if state_changes:
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
        if cleared_receptacles:
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
        success: bool
    ) -> float:
        """
        Calculate task progress reward (r_progress)
        Measures accuracy of task understanding and progress tracking

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

        # Weighted combination: 40%, 30%, 30%
        score = (
            0.4 * subgoal_status_score +
            0.3 * evidence_score +
            0.3 * subgoal_score
        )

        return score * self.progress_scale

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

        r_consistency = self.calculate_consistency_reward(world_model, ground_truth)
        r_progress = self.calculate_progress_reward(task_progress, ground_truth, step, done, success)
        r_exploration = self.calculate_exploration_reward(exploration_map, ground_truth, step)
        r_format = self.calculate_format_reward(is_format_valid, is_action_available)

        # Weighted combination
        total_reward = (
            self.alpha * r_consistency +
            self.beta * r_progress +
            self.gamma * r_exploration +
            self.delta * r_format
        )

        breakdown = {
            'r_consistency': r_consistency,
            'r_progress': r_progress,
            'r_exploration': r_exploration,
            'r_format': r_format,
            'r_intrinsic_total': total_reward
        }

        return total_reward, breakdown


# Factory function for easy import
def create_rebel_tracker():
    """Create ReBel components"""
    parser = BeliefStateParser()
    calculator = RebelRewardCalculator(alpha=0.3, beta=0.5, gamma=0.2, delta=0.1)
    return parser, calculator
