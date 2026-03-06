import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Tuple, Any, Optional, Union

# Teacher planner for ReBel - uses high-capability model for initial planning
from agent_system.multi_turn_rollout.teacher_planner import get_teacher_planner

from rlvmr import core_rlvmr
from bdrs.bdrs_rewards import BDRSRewardCalculator

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def _save_trajectories_if_enabled(
        self,
        total_batch_list: List[List[Dict]],
        total_infos: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        is_train: bool = True,
        envs=None,
    ):
        """Save trajectories to disk if enabled in config.

        Saves both training and validation rollouts. Training saves are limited
        to max_save_per_call trajectories per call to avoid excessive I/O.
        """
        if not getattr(self.config.trainer, 'save_trajectories', False):
            return

        import json
        import os
        import re
        from datetime import datetime

        save_dir = getattr(self.config.trainer, 'trajectory_save_dir', '/tmp/trajectories')
        os.makedirs(save_dir, exist_ok=True)
        mode = "train" if is_train else "val"

        # Limit training saves to avoid disk overflow (val saves all)
        max_save = 128 if is_train else len(total_batch_list)
        indices = list(range(min(max_save, len(total_batch_list))))

        # Derive experiment name from save_dir (last component) for group_id prefix
        exp_tag = os.path.basename(save_dir)  # e.g. "20260306_M5_rebel_full_seed42"

        # Get task list from env if available
        env_tasks = []
        if envs is not None and hasattr(envs, 'tasks'):
            env_tasks = list(envs.tasks)

        trajectories = []
        for env_idx in indices:
            traj = total_batch_list[env_idx]
            infos_seq = total_infos[env_idx] if env_idx < len(total_infos) else []
            task = env_tasks[env_idx] if env_idx < len(env_tasks) else ''

            traj_data = []
            for step_idx, (step, info) in enumerate(zip(traj, infos_seq)):
                # Get action text (decoded model response)
                action_text = info.get('_action_text', '')
                next_obs_text = info.get('_next_obs_text', '')

                # Extract action tag content
                action_match = re.search(r'<action>\s*(.*?)\s*</action>', action_text, re.DOTALL)
                action = action_match.group(1).strip() if action_match else action_text[:100]

                step_reward = step.get('rewards', 0.0)
                if hasattr(step_reward, 'item'):
                    step_reward = float(step_reward.item())
                elif hasattr(step_reward, '__float__'):
                    step_reward = float(step_reward)
                else:
                    try:
                        step_reward = float(step_reward)
                    except:
                        step_reward = 0.0

                traj_data.append({
                    "step": step_idx + 1,
                    "obs": next_obs_text[:500] if next_obs_text else '',
                    "response": action_text[:2000] if action_text else '',
                    "action": action,
                    "reward": step_reward,
                    "is_action_valid": bool(info.get('is_action_valid', True)),
                    "won": bool(info.get('won', False)),
                })

            # Derive task from first step if not available from env
            if not task and traj_data:
                for step_d in traj_data[:1]:
                    r = step_d.get('response', '')
                    if 'Task:' in r:
                        t_start = r.find('Task:') + 5
                        t_end = r.find('\n', t_start)
                        task = r[t_start:t_end].strip() if t_end > t_start else r[t_start:t_start+100]

            total_reward = float(episode_rewards[env_idx]) if env_idx < len(episode_rewards) else 0.0
            group_id = task[:60] if task else f'group_{env_idx}'
            # done=True means the env terminated (agent clicked Buy Now)
            # won=True means the purchase actually matched the task (reward >= threshold)
            episode_done = any(step.get('won', False) for step in traj_data)
            episode_bought = any(step.get('action', '').lower().strip() == 'click[buy now]' or
                                 'buy now' in step.get('action', '').lower()
                                 for step in traj_data)

            trajectories.append({
                "task": task,
                "done": "True" if episode_bought else "False",
                "won": episode_done,
                "total_reward": total_reward,
                "group_id": group_id,
                "exp_tag": exp_tag,
                "traj_index": env_idx,
                "data": traj_data,
            })

        if not trajectories:
            return

        # Append to JSONL file (one line per trajectory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"rollout_{mode}_{timestamp}.jsonl"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + '\n')

        success_count = sum(1 for t in trajectories if t.get('won', False))
        print(f"[TrajectoryCollector] Saved {len(trajectories)} {mode} trajectories "
              f"({success_count} success) to {filepath}")

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # 防御性检查：确保obs_text是字符串或None
        if obs_text is not None and not isinstance(obs_text, str):
            raise TypeError(
                f"obs_text must be str or None, got {type(obs_text)}. "
                f"item={item}, obs_texts type={type(obs_texts)}"
            )

        # Build chat structure with system prompt
        obs_content = raw_prompt[0]['content']
        if '<image>' in obs_content:
            obs_content = obs_content.replace('<image>', '')

        if obs_text is not None:
            obs_content += obs_text

        # Environment-aware system prompt
        env_name = getattr(self.config.env, 'env_name', '').lower()
        if 'webshop' in env_name:
            # WebShop SFT was trained without system prompt (user-only chat template).
            # Match that format for RL rollout to avoid distribution shift.
            chat = np.array([
                {
                    "content": obs_content,
                    "role": "user",
                }
            ])
        else:
            system_prompt = (
                "You are an expert embodied agent operating in the ALFRED environment. "
                "Your goal is to complete household tasks by navigating, interacting with objects, "
                "and maintaining accurate beliefs about the world state. "
                "Always output your response in the required format."
            )
            chat = np.array([
                {
                    "content": system_prompt,
                    "role": "system",
                },
                {
                    "content": obs_content,
                    "role": "user",
                }
            ])

        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )

        # Initialize return dict
        row_dict = {}

        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template

        # 防御性检查：确保prompt_with_chat_template是字符串
        if not isinstance(prompt_with_chat_template, str):
            raise TypeError(
                f"prompt_with_chat_template must be str, got {type(prompt_with_chat_template)}. "
                f"obs_content={repr(obs_content[:100]) if obs_content else None}, "
                f"obs_text={repr(obs_text[:100]) if obs_text else None}"
            )

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation='error')
        
        
    
        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []

        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)

        # Aggregate batch data
        batch = collate_fn(processed_samples)

        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        traj_length = []

        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            valid_step = 0
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    valid_step += 1
                    effective_batch.append(data)
            traj_length.append(valid_step)  

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        gen_batch_output.meta_info["traj_length"] = traj_length
        traj_length_stat = {
            "traj_length_mean": np.mean(traj_length),
            "traj_length_min": np.min(traj_length),
            "traj_length_max": np.max(traj_length),
        }
        gen_batch_output.meta_info["traj_length_stat"] = traj_length_stat

        return gen_batch_output

    def _run_planning_step(
            self,
            planning_prompts: List[str],
            gen_batch: DataProto,
            actor_rollout_wg,
    ) -> List[str]:
        """
        Run planning prompts through the model to generate task plans.

        Args:
            planning_prompts: List of planning prompt strings
            gen_batch: Original batch (used as template for creating planning batch)
            actor_rollout_wg: Actor model workers

        Returns:
            List of model output strings (should be JSON plans)
        """
        # Create a batch for planning prompts
        batch_size = len(planning_prompts)
        planning_samples = []

        for i, prompt in enumerate(planning_prompts):
            # Apply chat template
            chat = [{"content": prompt, "role": "user"}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )

            # Tokenize
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='error'
            )

            # Compute position IDs
            position_ids = compute_position_id_with_mask(attention_mask)

            planning_samples.append({
                'input_ids': input_ids[0],
                'attention_mask': attention_mask[0],
                'position_ids': position_ids[0],
                'raw_prompt_ids': self.tokenizer.encode(prompt_with_template, add_special_tokens=False),
            })

        # Collate into batch
        from verl.utils.dataset.rl_dataset import collate_fn as _collate_fn
        planning_batch_data = _collate_fn(planning_samples)

        # Create DataProto for generation
        planning_batch = DataProto.from_single_dict(
            data=planning_batch_data,
            meta_info=gen_batch.meta_info.copy() if hasattr(gen_batch, 'meta_info') else {}
        )

        # Pop the necessary keys for generation
        batch_input = planning_batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=['raw_prompt_ids'],
        )
        batch_input.meta_info = gen_batch.meta_info

        # Generate sequences
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        # Decode responses
        responses = batch_output.batch.get('responses', batch_output.batch.get('response_ids', None))
        if responses is None:
            return ["" for _ in range(batch_size)]

        planning_outputs = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        return planning_outputs

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances

        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        # Initial observations from the environment
        obs, infos = envs.reset()

        # ═══════════════════════════════════════════════════════════════════════════════
        # NEW: ReBel Planning Step - Run BEFORE main interaction loop
        # Uses Teacher Model (Claude Opus) for high-quality initial plans
        # ═══════════════════════════════════════════════════════════════════════════════
        if hasattr(envs, 'use_rebel') and envs.use_rebel:
            try:
                planning_prompts = envs.get_planning_prompts()
                if planning_prompts:
                    # Check if teacher planner is configured (default: True)
                    use_teacher = getattr(self.config.env, 'use_teacher_planner', True)

                    if use_teacher:
                        # Use high-capability teacher model for planning
                        teacher_config = getattr(self.config.env, 'teacher_planner', None)
                        if teacher_config:
                            teacher_model = getattr(teacher_config, 'model', "claude-sonnet-4-6-cc")
                            teacher_api_base = getattr(teacher_config, 'api_base', "https://www.dmxapi.cn")
                            teacher_api_key = getattr(teacher_config, 'api_key', "sk-4n74DJ6yiFNurN9Yq3JpzrxDvuUzN0vQQmh8s6kIl6IHrLEh")
                        else:
                            # Default values
                            teacher_model = "claude-sonnet-4-6-cc"
                            teacher_api_base = "https://www.dmxapi.cn"
                            teacher_api_key = "sk-4n74DJ6yiFNurN9Yq3JpzrxDvuUzN0vQQmh8s6kIl6IHrLEh"

                        teacher_planner = get_teacher_planner(
                            model=teacher_model,
                            api_base=teacher_api_base,
                            api_key=teacher_api_key,
                        )

                        # Generate plans using teacher model
                        plans = teacher_planner.generate_plans(planning_prompts)
                        envs.set_task_plans(plans)

                        # Log planning results
                        valid_plans = sum(1 for p in plans if p is not None)
                        print(f"[ReBel Planning] Teacher model generated {valid_plans}/{len(plans)} valid task plans")
                    else:
                        # Use the training model for planning (original behavior)
                        planning_outputs = self._run_planning_step(
                            planning_prompts=planning_prompts,
                            gen_batch=gen_batch,
                            actor_rollout_wg=actor_rollout_wg,
                        )

                        # Parse and store the plans
                        plans = envs.parse_planning_output(planning_outputs)
                        envs.set_task_plans(plans)

                        # Log planning results
                        valid_plans = sum(1 for p in plans if p is not None)
                        print(f"[ReBel Planning] Training model generated {valid_plans}/{len(plans)} valid task plans")
            except Exception as e:
                print(f"[ReBel Planning] Warning: Planning step failed: {e}")
                import traceback
                traceback.print_exc()

        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None

        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                batch_input = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                batch_input = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            batch_input.meta_info = gen_batch.meta_info

            batch_output = actor_rollout_wg.generate_sequences(batch_input)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            batch.non_tensor_batch['full_output'] = text_actions

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            batch.non_tensor_batch['action_available'] = np.array([info['action_available'] for info in infos], dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                # Attach decoded action text and next obs text for trajectory saving
                infos[i]['_action_text'] = text_actions[i] if i < len(text_actions) else ''
                anchor = next_obs.get('anchor') if isinstance(next_obs, dict) else None
                infos[i]['_next_obs_text'] = anchor[i] if isinstance(anchor, (list, np.ndarray)) and i < len(anchor) else ''
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)

            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # Save trajectories if enabled
        self._save_trajectories_if_enabled(
            total_batch_list=total_batch_list,
            total_infos=total_infos,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            is_train=True,
            envs=envs,
        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, total_infos

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_infos_list = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, infos_list = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_infos_list += infos_list

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos_list

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)

        # add rlvmr rewards into total_batch_list
        if self.config.algorithm.rlvmr.enable:
            total_batch_list = core_rlvmr.process_trajectory_rlvmr_rewards(
                trajectory_list=total_batch_list, config=self.config, episode_rewards=total_episode_rewards
            )
        # Helper function for statistics (used by BDRS and ReBel)
        def _safe_stat(arr):
            import numpy as _np
            if len(arr) == 0:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            return {
                "mean": float(_np.mean(arr)),
                "min": float(_np.min(arr)),
                "max": float(_np.max(arr)),
                "std": float(_np.std(arr))
            }

        # BDRS: 从 infos 中读取 belief 和 prev_belief，计算差分奖励，写回 step 字段
        if hasattr(self.config.algorithm, 'bdrs') and getattr(self.config.algorithm.bdrs, 'enable', False):
            # 从config读取细粒度奖励参数
            bdrs_config = self.config.algorithm.bdrs
            calc = BDRSRewardCalculator(
                world_w=float(getattr(bdrs_config, 'world_consistency_weight', 1.0)),
                progress_w=float(getattr(bdrs_config, 'task_progress_weight', 1.0)),
                explore_w=float(getattr(bdrs_config, 'exploration_efficiency_weight', 1.0)),
                reward_correct_belief=float(getattr(bdrs_config, 'reward_correct_belief', 0.2)),
                reward_new_conflict=float(getattr(bdrs_config, 'reward_new_conflict', -0.1)),
                reward_subgoal_complete=float(getattr(bdrs_config, 'reward_subgoal_complete', 0.5)),
                reward_new_entity=float(getattr(bdrs_config, 'reward_new_entity', 0.05)),
                reward_new_location=float(getattr(bdrs_config, 'reward_new_location', 0.1)),
                penalty_revisit=float(getattr(bdrs_config, 'penalty_revisit', -0.02)),
            )

            bdrs_world, bdrs_progress, bdrs_explore, bdrs_total = [], [], [], []

            for env_idx in range(len(total_batch_list)):
                traj = total_batch_list[env_idx]
                infos_seq = total_infos[env_idx]
                for step_idx in range(len(traj)):
                    step = traj[step_idx]
                    if not step.get('active_masks', False):
                        continue

                    info = infos_seq[step_idx] if step_idx < len(infos_seq) else {}
                    curr_belief = info.get('belief', {})
                    prev_belief = info.get('prev_belief', None)

                    # 计算差分奖励
                    bdrs = calc.step_reward(
                        prev_belief=prev_belief,
                        curr_belief=curr_belief,
                        info=info
                    )

                    step['bdrs_step_reward'] = torch.tensor(bdrs['total'])
                    step['bdrs_components'] = bdrs

                    bdrs_world.append(bdrs['world_consistency'])
                    bdrs_progress.append(bdrs['task_progress'])
                    bdrs_explore.append(bdrs['exploration_efficiency'])
                    bdrs_total.append(bdrs['total'])

            # 不能直接给config添加属性（struct mode），先保存到临时变量
            meta_info_bdrs = {
                "world_consistency": _safe_stat(bdrs_world),
                "task_progress": _safe_stat(bdrs_progress),
                "exploration_efficiency": _safe_stat(bdrs_explore),
                "total_reward": _safe_stat(bdrs_total),
                "n_steps": len(bdrs_total),
            }

        # ReBel: 从 infos 中读取 belief_state 和 rebel_intrinsic_reward，写回 step 字段
        if hasattr(self.config.algorithm, 'rebel') and getattr(self.config.algorithm.rebel, 'enable', False):
            rebel_intrinsic_rewards = []

            # Determine environment type for env-agnostic dispatching
            env_name = getattr(self.config.env, 'env_name', '').lower()
            is_webshop = 'webshop' in env_name

            # V7: Import BeliefDeviationCalculator for belief deviation metrics (ALFWorld only)
            belief_deviation_calc = None
            if not is_webshop:
                try:
                    from agent_system.environments.env_package.alfworld import BeliefDeviationCalculator
                    belief_deviation_calc = BeliefDeviationCalculator()
                except ImportError:
                    pass

            # V11: Import the appropriate semantic_belief_abstract for HiBO grouping
            if is_webshop:
                from rebel.hibo_grouping import webshop_semantic_belief_abstract as semantic_belief_abstract
            else:
                from rebel.hibo_grouping import semantic_belief_abstract

            # V7: Collect belief deviation metrics
            belief_deviations = {
                'total': [],
                'object_location': [],
                'state': [],
                'exploration': [],
                'belief_valid_ratio': []
            }

            for env_idx in range(len(total_batch_list)):
                traj = total_batch_list[env_idx]
                infos_seq = total_infos[env_idx]
                for step_idx in range(len(traj)):
                    step = traj[step_idx]
                    if not step.get('active_masks', False):
                        continue

                    info = infos_seq[step_idx] if step_idx < len(infos_seq) else {}

                    # Store parsed belief_state from env
                    belief_state = info.get('belief_state', {})
                    step['belief_state'] = belief_state

                    # V11: Compute and store belief abstract for HiBO grouping
                    step['belief_abstract'] = semantic_belief_abstract(belief_state)

                    # Store ReBel intrinsic reward (already computed in env)
                    rebel_intrinsic = info.get('rebel_intrinsic_reward', 0.0)
                    step['rebel_intrinsic_reward'] = torch.tensor(rebel_intrinsic)

                    # V2: Store task_type for task-aware grouping
                    task_type = info.get('task_type', 'unknown')
                    step['task_type'] = task_type

                    # V6: Store gamefile (full task name) and success status
                    gamefile = info.get('extra.gamefile', '')
                    step['gamefile'] = gamefile
                    step['won'] = info.get('won', False)

                    rebel_intrinsic_rewards.append(rebel_intrinsic)

                    # V7: Compute belief deviation metrics (ALFWorld only)
                    ground_truth = info.get('ground_truth_state', {})
                    if ground_truth and belief_deviation_calc is not None:
                        deviation_metrics = belief_deviation_calc.compute_total_belief_deviation(
                            belief_state=belief_state,
                            ground_truth=ground_truth
                        )
                        belief_deviations['total'].append(deviation_metrics['total_deviation'])
                        belief_deviations['object_location'].append(deviation_metrics['object_location']['deviation'])
                        belief_deviations['state'].append(deviation_metrics['state']['deviation'])
                        belief_deviations['exploration'].append(deviation_metrics['exploration']['deviation'])
                        belief_deviations['belief_valid_ratio'].append(1.0 if deviation_metrics['belief_valid'] else 0.0)

                        # Store deviation in step for detailed analysis
                        step['belief_deviation'] = deviation_metrics

            # Statistics for logging
            meta_info_rebel = {
                "intrinsic_reward": _safe_stat(rebel_intrinsic_rewards),
                "n_steps": len(rebel_intrinsic_rewards),
            }

            # V7: Add belief deviation statistics
            if belief_deviations['total']:
                meta_info_rebel["belief_deviation"] = {
                    "total": _safe_stat(belief_deviations['total']),
                    "object_location": _safe_stat(belief_deviations['object_location']),
                    "state": _safe_stat(belief_deviations['state']),
                    "exploration": _safe_stat(belief_deviations['exploration']),
                    "belief_valid_ratio": _safe_stat(belief_deviations['belief_valid_ratio']),
                }

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        assert len(envs.buffers[0]) == len(total_batch_list[0]), "envs.buffers[0] and total_batch_list[0] should have the same length"
        if self.config.algorithm.rlvmr.enable:
            gen_batch_output.meta_info["rlvmr_step_advantage_w"] = (self.config.algorithm.rlvmr.step_advantage_w)
            gen_batch_output.meta_info["rlvmr_mode"] = self.config.algorithm.rlvmr.mode
        if hasattr(self.config.algorithm, 'bdrs') and getattr(self.config.algorithm.bdrs, 'enable', False):
            gen_batch_output.meta_info["bdrs_step_advantage_w"] = float(getattr(self.config.algorithm.bdrs, 'step_advantage_w', 1.0))
            gen_batch_output.meta_info["bdrs_mode"] = str(getattr(self.config.algorithm.bdrs, 'mode', 'mean_std_norm'))
            # 传递BDRS统计信息（从局部变量读取，避免struct mode冲突）
            if 'meta_info_bdrs' in locals():
                gen_batch_output.meta_info['bdrs_stats'] = meta_info_bdrs
        if hasattr(self.config.algorithm, 'rebel') and getattr(self.config.algorithm.rebel, 'enable', False):
            gen_batch_output.meta_info["rebel_step_advantage_w"] = float(getattr(self.config.algorithm.rebel, 'step_advantage_w', 1.0))
            gen_batch_output.meta_info["rebel_mode"] = str(getattr(self.config.algorithm.rebel, 'mode', 'mean_norm'))
            gen_batch_output.meta_info["rebel_belief_granularity"] = str(getattr(self.config.algorithm.rebel, 'belief_granularity', 'subgoal'))
            gen_batch_output.meta_info["rebel_summarize_groups"] = bool(getattr(self.config.algorithm.rebel, 'summarize_groups', False))
            # V2: Task-aware configuration
            gen_batch_output.meta_info["rebel_task_aware_grouping"] = bool(getattr(self.config.algorithm.rebel, 'task_aware_grouping', False))
            gen_batch_output.meta_info["rebel_per_task_normalization"] = bool(getattr(self.config.algorithm.rebel, 'per_task_normalization', False))
            # V5: Conditional normalization configuration
            gen_batch_output.meta_info["rebel_conditional_norm"] = bool(getattr(self.config.algorithm.rebel, 'conditional_norm', True))
            gen_batch_output.meta_info["rebel_min_samples_for_norm"] = int(getattr(self.config.algorithm.rebel, 'min_samples_for_norm', 10))
            gen_batch_output.meta_info["rebel_min_std_for_norm"] = float(getattr(self.config.algorithm.rebel, 'min_std_for_norm', 0.1))
            # V8: Task adaptive weighting configuration
            gen_batch_output.meta_info["rebel_use_task_weighting"] = bool(getattr(self.config.algorithm.rebel, 'use_task_weighting', False))
            gen_batch_output.meta_info["rebel_weight_alpha"] = float(getattr(self.config.algorithm.rebel, 'weight_alpha', 2.0))
            gen_batch_output.meta_info["rebel_weight_min"] = float(getattr(self.config.algorithm.rebel, 'weight_min', 0.3))
            gen_batch_output.meta_info["rebel_weight_max"] = float(getattr(self.config.algorithm.rebel, 'weight_max', 3.0))
            gen_batch_output.meta_info["rebel_weight_baseline_sr"] = float(getattr(self.config.algorithm.rebel, 'weight_baseline_sr', 0.85))
            # V8: task_success_rates 将在 ray_trainer.py 中动态更新
            gen_batch_output.meta_info["rebel_task_success_rates"] = {}
            # 传递ReBel统计信息
            if 'meta_info_rebel' in locals():
                gen_batch_output.meta_info['rebel_stats'] = meta_info_rebel

            # V11: Pass HiBO configuration to meta_info (for rebel_hibo advantage estimator)
            gen_batch_output.meta_info["hibo_step_advantage_w"] = float(getattr(self.config.algorithm.rebel, 'step_advantage_w', 0.5))
            gen_batch_output.meta_info["hibo_mode"] = str(getattr(self.config.algorithm.rebel, 'mode', 'mean_norm'))
            gen_batch_output.meta_info["hibo_min_obs_group_size"] = int(getattr(self.config.algorithm.rebel, 'min_obs_group_size', 2))
            gen_batch_output.meta_info["hibo_summarize"] = bool(getattr(self.config.algorithm.rebel, 'summarize_groups', False))
        return gen_batch_output
