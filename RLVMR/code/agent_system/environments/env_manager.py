from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
import copy
from bdrs import BeliefStateManager

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        self.belief_mgr = BeliefStateManager(env_name)
        self.prev_beliefs = {}  # 保存上一步的belief快照，用于差分奖励计算
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.meta_think = (self.config is not None and
                          hasattr(self.config, 'env') and
                          hasattr(self.config.env, 'alfworld') and
                          hasattr(self.config.env.alfworld, 'meta_think') and
                          self.config.env.alfworld.meta_think)
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        # initialize belief states with tasks
        env_ids = list(range(len(text_obs)))
        try:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=self.tasks)
        except Exception:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=None)

        # 清空prev_beliefs（开始新的episode）
        self.prev_beliefs.clear()

        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        actions, valids, plannings, action_available = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['action_available'] = to_numpy(action_available[i])

            # 保存当前belief作为prev_belief（在更新之前）
            prev_belief_snap = None
            if i in self.prev_beliefs:
                prev_belief_snap = self.prev_beliefs[i]

            # update belief state per env and attach a light snapshot for debugging/analysis
            try:
                anchor_obs = text_obs[i]
                self.belief_mgr.step_update(
                    env_id=i,
                    observation=anchor_obs,
                    action=actions[i],
                    info=info,
                    reward=float(rewards[i]),
                )
                curr_belief_snap = self.belief_mgr.snapshot(i)

                # 将完整的exploration_map传递（包括差分信息）
                info['belief'] = {
                    'step_idx': curr_belief_snap.step_idx,
                    'world_model': curr_belief_snap.world_model,
                    'task_progress': curr_belief_snap.task_progress,
                    'exploration_map': curr_belief_snap.exploration_map,  # 传递完整的exploration_map
                    'notes': curr_belief_snap.notes,
                }

                # 传递prev_belief用于差分奖励计算
                info['prev_belief'] = prev_belief_snap

                # 保存当前belief作为下一步的prev_belief
                self.prev_beliefs[i] = {
                    'step_idx': curr_belief_snap.step_idx,
                    'world_model': copy.deepcopy(curr_belief_snap.world_model),
                    'task_progress': copy.deepcopy(curr_belief_snap.task_progress),
                    'exploration_map': copy.deepcopy(curr_belief_snap.exploration_map),
                    'notes': curr_belief_snap.notes[:],
                }
            except Exception as e:
                # 出错时也要确保有默认值
                info['belief'] = {}
                info['prev_belief'] = None
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')

            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        use_bdrs_template = (
            self.config is not None and
            hasattr(self.config, 'algorithm') and
            hasattr(self.config.algorithm, 'bdrs') and
            getattr(self.config.algorithm.bdrs, 'enable', False)
        )

        # DEBUG: 诊断template选择
        print(f"\n{'='*80}")
        print(f"[DEBUG build_text_obs] Template选择诊断:")
        print(f"  self.config is not None: {self.config is not None}")
        if self.config is not None:
            print(f"  hasattr(config, 'algorithm'): {hasattr(self.config, 'algorithm')}")
            if hasattr(self.config, 'algorithm'):
                print(f"  hasattr(algorithm, 'bdrs'): {hasattr(self.config.algorithm, 'bdrs')}")
                if hasattr(self.config.algorithm, 'bdrs'):
                    print(f"  config.algorithm.bdrs.enable: {getattr(self.config.algorithm.bdrs, 'enable', False)}")
        print(f"  => use_bdrs_template = {use_bdrs_template}")
        print(f"  self.meta_think = {self.meta_think}")
        if self.config and hasattr(self.config.env, 'alfworld') and hasattr(self.config.env.alfworld, 'action_only'):
            print(f"  self.config.env.alfworld.action_only = {self.config.env.alfworld.action_only}")

        if self.meta_think and not use_bdrs_template:
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_MC
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_MC
            print(f"  => 选择分支: meta_think (MC)")
        elif self.config is not None and self.config.env.alfworld.action_only and not use_bdrs_template:
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_NOTHINK
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_NOTHINK
            print(f"  => 选择分支: action_only (NOTHINK)")
        else:
            if use_bdrs_template:
                _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_BDRS
                _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_BDRS
                print(f"  => 选择分支: BDRS ✓✓✓")
            else:
                _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS
                _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE
                print(f"  => 选择分支: 默认 (THINK)")
        print(f"{'='*80}\n")

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or history_length <= 0:
                if self.meta_think:
                    obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
                else:
                    obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
            else:
                # with all the history
                history_length = len(self.buffers[i])

                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Observation {step_number}: {record['text_obs']}, output: '{record['full_output']}']\n"

                if self.meta_think:
                    obs = _ALFWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                        planning=self.plannings[i]
                    )
                else:
                    obs = _ALFWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions, plannings=[]):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
        for i in range(len(plannings)):
            if plannings[i] is not None:
                self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def _set_meta_think(self, type: bool):
        self.meta_think = type


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        self.belief_mgr = BeliefStateManager(env_name)
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        text_obs, infos = self.envs.reset()

        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)
        # initialize belief states with tasks
        env_ids = list(range(len(text_obs)))
        tasks = getattr(self, 'tasks', None)
        try:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=tasks)
        except Exception:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=None)
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos])

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)
            # update belief state and attach snapshot
            try:
                anchor_obs = text_obs[i]
                self.belief_mgr.step_update(
                    env_id=i,
                    observation=anchor_obs,
                    action=actions[i],
                    info=info,
                    reward=float(rewards[i]),
                )
                snap = self.belief_mgr.snapshot(i)
                info['belief'] = {
                    'step_idx': snap.step_idx,
                    'world_model': snap.world_model,
                    'task_progress': snap.task_progress,
                    'exploration_map': {
                        'visited_rooms': list(snap.exploration_map.get('visited_rooms', [])),
                        'visited_objects': list(snap.exploration_map.get('visited_objects', [])),
                    },
                    'notes': snap.notes,
                }
            except Exception:
                pass

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.config is not None and hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') and self.config.env.sciworld.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        planning=self.plannings[i],
                        available_actions=available_actions[i]
                    )
                else:
                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=available_actions[i]
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        self.belief_mgr = BeliefStateManager(env_name)
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        # initialize belief states with tasks
        env_ids = list(range(len(infos)))
        try:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=self.tasks)
        except Exception:
            self.belief_mgr.reset(env_ids=env_ids, task_desc=None)
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            # update belief state and attach snapshot
            try:
                anchor_obs = self.pre_text_obs[i]
                self.belief_mgr.step_update(
                    env_id=i,
                    observation=anchor_obs,
                    action=actions[i],
                    info=info,
                    reward=float(rewards[i]),
                )
                snap = self.belief_mgr.snapshot(i)
                info['belief'] = {
                    'step_idx': snap.step_idx,
                    'world_model': snap.world_model,
                    'task_progress': snap.task_progress,
                    'exploration_map': {
                        'visited_rooms': list(snap.exploration_map.get('visited_rooms', [])),
                        'visited_objects': list(snap.exploration_map.get('visited_objects', [])),
                    },
                    'notes': snap.notes,
                }
            except Exception:
                pass

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], "full_output": ""})
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return


def make_envs(config):
    """
    Create enviroments
    """
    print("[DEBUG make_envs] Starting make_envs()")
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    print(f"[DEBUG make_envs] group_n={group_n}, env_name={config.env.env_name}")

    if "alfworld" in config.env.env_name.lower():
        print("[DEBUG make_envs] Creating ALFWorld environments")
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection, alfworld_projection_rlvmr, alfworld_projection_bdrs

        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        print(f"[DEBUG make_envs] alf_config_path={alf_config_path}")
        print(f"[DEBUG make_envs] generalization_level={config.env.alfworld.generalization_level}")

        if config.env.alfworld.generalization_level == 2:
            alf_train_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_ood.yaml')
            alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_test_ood.yaml')
            print(f"[DEBUG make_envs] Building training envs (level 2)...")
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            val_env_num = group_n
            _envs = build_alfworld_envs(alf_train_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Building validation envs (level 2)...")
            _val_envs = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 1:
            print(f"[DEBUG make_envs] Building training envs (level 1)...")
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            val_env_num = group_n
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Building validation envs (level 1)...")
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 0:
            print(f"[DEBUG make_envs] Building training envs (level 0)...")
            # Use env.rollout.n for validation-only mode when total_epochs=0
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            val_env_num = group_n  # Use env.rollout.n for validation
            print(f"[DEBUG make_envs] train_env_num={train_env_num}, val_env_num={val_env_num}")
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Training envs built successfully")
            print(f"[DEBUG make_envs] Building validation envs (level 0)...")
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False)
            print(f"[DEBUG make_envs] Validation envs built successfully")

        print(f"[DEBUG make_envs] Setting up projection function...")
        # Check if BDRS is enabled
        use_bdrs = (hasattr(config, 'algorithm') and
                   hasattr(config.algorithm, 'bdrs') and
                   getattr(config.algorithm.bdrs, 'enable', False))

        if use_bdrs:
            print("[DEBUG make_envs] Using BDRS projection (PLAN/EXECUTE/EXPLORE/VERIFY)")
            projection_f = partial(alfworld_projection_bdrs)
        elif config.env.alfworld.meta_think:
            print("[DEBUG make_envs] Using RLVMR projection (planning/explore/reflection/monitor)")
            projection_f = partial(alfworld_projection_rlvmr)
        else:
            print("[DEBUG make_envs] Using default projection (think/action)")
            projection_f = partial(alfworld_projection)

        print(f"[DEBUG make_envs] Creating AlfWorldEnvironmentManager for training...")
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        print(f"[DEBUG make_envs] Creating AlfWorldEnvironmentManager for validation...")
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)
        print(f"[DEBUG make_envs] Environment managers created successfully")
        return envs, val_envs
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        import json
        generalization_level = config.env.sciworld['generalization_level']

        if generalization_level == 2:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L2_idx.json'
        elif generalization_level == 1:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L1_idx.json'
        elif generalization_level == 0:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'

        with open(variation_path, 'r') as f:
            variations_idx = json.load(f)

        simplifications_preset = config.env.sciworld.get('simplifications_preset', "easy")
        env_step_limit = config.env.sciworld.get('env_step_limit', 100)
        jar_path = config.env.sciworld.get('jar_path', None)

        _envs = build_sciworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['train']
        )

        _val_envs = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['test']
        )

        # Create projection function
        projection_f = partial(sciworld_projection)

        # Create environment managers
        envs = SciWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs = SciWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)

        # Give some time for environments to initialize
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)

        return envs, val_envs

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)