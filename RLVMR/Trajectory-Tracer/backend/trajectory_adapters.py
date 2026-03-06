"""
Trajectory Adapters - 支持多种轨迹数据格式
提供统一的接口来处理不同来源的轨迹数据
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Check if HuggingFace datasets is available (without importing it yet)
DATASETS_AVAILABLE = False
try:
    import importlib.util
    spec = importlib.util.find_spec("datasets")
    if spec is not None:
        DATASETS_AVAILABLE = True
except (ImportError, AttributeError):
    pass

if not DATASETS_AVAILABLE:
    print("Warning: HuggingFace datasets library not available. HuggingFace format will be disabled.")


class Message:
    """统一的消息格式"""
    def __init__(self, role: str, content: str, thought: Optional[str] = None,
                 action: Optional[str] = None, metadata: Optional[Dict] = None):
        self.role = role  # 'human' 或 'agent'
        self.content = content
        self.thought = thought
        self.action = action
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
            'thought': self.thought,
            'action': self.action,
            'metadata': self.metadata
        }


class Trajectory:
    """统一的轨迹格式"""
    def __init__(self, id: str, task: str, status: str, steps: int,
                 task_type: str, messages: List[Message], environment: str = "",
                 metadata: Optional[Dict] = None):
        self.id = id
        self.task = task
        self.status = status  # 'success', 'failed', 'unknown'
        self.steps = steps
        self.task_type = task_type
        self.messages = messages
        self.environment = environment
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            'id': self.id,
            'task': self.task,
            'status': self.status,
            'steps': self.steps,
            'task_type': self.task_type,
            'messages': [m.to_dict() for m in self.messages],
            'environment': self.environment,
            'metadata': self.metadata
        }


class TrajectoryAdapter(ABC):
    """轨迹适配器基类"""

    @abstractmethod
    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载原始数据"""
        pass

    @abstractmethod
    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """将原始数据转换为统一的 Trajectory 格式"""
        pass

    def load_and_parse(self, path: Path) -> List[Trajectory]:
        """加载并解析所有轨迹"""
        raw_data = self.load(path)
        trajectories = []
        for idx, item in enumerate(raw_data):
            try:
                trajectory = self.parse(item, idx)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Warning: Failed to parse trajectory {idx}: {e}")
        return trajectories


class HuggingFaceDatasetAdapter(TrajectoryAdapter):
    """HuggingFace datasets 格式适配器"""

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载 HuggingFace dataset"""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("HuggingFace datasets library is not available. Cannot load this format.")

        # Import only when needed
        from datasets import load_from_disk
        dataset = load_from_disk(str(path))
        return list(dataset)

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """解析 HuggingFace 格式的轨迹"""
        conversations = raw_item.get('conversations', [])
        messages = []

        task = ""
        environment = ""
        task_type = "unknown"

        for conv in conversations:
            role = 'human' if conv['from'] == 'human' else 'agent'
            value = conv['value']

            # 提取任务信息
            if 'Your task is to:' in value:
                task_start = value.find('Your task is to:') + len('Your task is to:')
                task_end = value.find('\n', task_start) if '\n' in value[task_start:] else len(value)
                task = value[task_start:task_end].strip()

                # 提取任务类型
                first_word = task.split()[0].lower() if task else 'unknown'
                if first_word in ['put', 'clean', 'heat', 'cool', 'find', 'examine', 'use']:
                    task_type = first_word
                else:
                    task_type = 'other'

                # 提取环境描述
                env_end = value.find('Your task is to:')
                environment = value[:env_end].strip() if env_end > 0 else ""

            # 解析 agent 的思考和动作
            thought = None
            action = None
            if role == 'agent':
                if 'Thought:' in value:
                    parts = value.split('Action:')
                    thought = parts[0].replace('Thought:', '').strip()
                    action = parts[1].strip() if len(parts) > 1 else ""
                elif 'Action:' in value:
                    action = value.replace('Action:', '').strip()

            messages.append(Message(role, value, thought, action))

        # 判断是否成功
        last_message = conversations[-1]['value'].lower() if conversations else ""
        status = 'success' if any(word in last_message for word in
                                 ['succeed', 'success', 'task completed', 'congratulations']) else 'unknown'

        # 计算步数
        steps = len([m for m in messages if m.role == 'agent' and m.action])

        return Trajectory(
            id=f"hf_traj_{idx:05d}",
            task=task,
            status=status,
            steps=steps,
            task_type=task_type,
            messages=messages,
            environment=environment,
            metadata={'item_id': raw_item.get('item_id', ''), 'source': 'huggingface'}
        )


class REBELJSONAdapter(TrajectoryAdapter):
    """REBEL JSON 格式适配器"""

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSON 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """解析 REBEL 格式的轨迹"""
        task = raw_item.get('task', '')
        done = raw_item.get('done', 'False')
        data = raw_item.get('data', [])

        messages = []

        # 解析每个步骤
        for step_data in data:
            step_num = step_data.get('step', 0)
            obs = step_data.get('obs', '')
            response = step_data.get('response', '')

            # 添加观察（环境反馈）
            if obs:
                messages.append(Message(
                    role='human',
                    content=obs,
                    metadata={'step': step_num, 'type': 'observation'}
                ))

            # 解析 agent 响应
            if response:
                thought = None
                action = None
                belief = None
                reasoning = None

                # 尝试解析结构化响应
                if '<belief>' in response and '</belief>' in response:
                    belief_start = response.find('<belief>') + len('<belief>')
                    belief_end = response.find('</belief>')
                    belief = response[belief_start:belief_end].strip()

                if '<reasoning>' in response and '</reasoning>' in response:
                    reasoning_start = response.find('<reasoning>') + len('<reasoning>')
                    reasoning_end = response.find('</reasoning>')
                    reasoning = response[reasoning_start:reasoning_end].strip()
                    thought = reasoning  # 使用 reasoning 作为 thought

                if '<action>' in response and '</action>' in response:
                    action_start = response.find('<action>') + len('<action>')
                    action_end = response.find('</action>')
                    action = response[action_start:action_end].strip()

                messages.append(Message(
                    role='agent',
                    content=response,
                    thought=thought,
                    action=action,
                    metadata={
                        'step': step_num,
                        'belief': belief,
                        'type': 'agent_response'
                    }
                ))

        # 提取任务类型
        task_type = 'unknown'
        if task:
            first_word = task.split()[0].lower()
            if first_word in ['put', 'clean', 'heat', 'cool', 'find', 'examine', 'use']:
                task_type = first_word
            else:
                task_type = 'other'

        # 状态判断
        status = 'success' if done == 'True' else 'failed' if done == 'False' else 'unknown'

        # 计算步数
        steps = len([m for m in messages if m.role == 'agent' and m.action])

        # 环境信息（从第一个观察中提取）
        environment = ""
        if data and data[0].get('obs'):
            first_obs = data[0]['obs']
            if 'You are in the middle of a room' in first_obs:
                env_end = first_obs.find('\nYour task is to:')
                if env_end > 0:
                    environment = first_obs[:env_end].strip()

        return Trajectory(
            id=f"rebel_traj_{idx:05d}",
            task=task,
            status=status,
            steps=steps,
            task_type=task_type,
            messages=messages,
            environment=environment,
            metadata={'source': 'rebel', 'done': done}
        )


class WebShopRebelJSONLAdapter(TrajectoryAdapter):
    """WebShop ReBel Hindsight JSONL 适配器

    加载由 generate_webshop_rebel_hindsight.py 产生的标注数据。
    每行一条轨迹，格式: {conversations, item_id, num_steps, annotation_success_rate, task}
    """

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSONL 文件或包含 JSONL 的目录"""
        trajectories = []
        if path.is_dir():
            jsonl_files = sorted(path.glob('*.jsonl'))
            for jsonl_file in jsonl_files:
                trajectories.extend(self._load_file(jsonl_file))
        else:
            trajectories = self._load_file(path)
        return trajectories

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {path.name}: {e}")
        return items

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """解析 WebShop ReBel hindsight 轨迹"""
        import re

        item_id = raw_item.get('item_id', f'webshop_rebel_{idx:05d}')
        task = raw_item.get('task', '')
        num_steps = raw_item.get('num_steps', 0)
        success_rate = raw_item.get('annotation_success_rate', 0.0)
        conversations = raw_item.get('conversations', [])

        messages = []
        step_num = 0
        skipped_initial = 0

        for conv in conversations:
            role_raw = conv.get('from', '')
            value = conv.get('value', '')
            loss = conv.get('loss', None)

            # 跳过最初的 system prompt 和 ack (前两个 loss=False 的轮次)
            if loss is False and skipped_initial < 2:
                skipped_initial += 1
                continue

            if role_raw == 'human':
                # 环境观察轮
                # 提取 observation 部分 (去掉 Task: 和 Current Belief State: 等元信息)
                obs_display = value
                obs_section = ''
                if 'Observation:' in value:
                    obs_start = value.find('Observation:') + len('Observation:')
                    obs_end = value.find('Current Belief State:')
                    if obs_end == -1:
                        obs_end = value.find('Available Actions:')
                    if obs_end == -1:
                        obs_end = len(value)
                    obs_section = value[obs_start:obs_end].strip()

                messages.append(Message(
                    role='human',
                    content=obs_display,
                    metadata={
                        'step': step_num + 1,
                        'type': 'observation',
                        'observation': obs_section
                    }
                ))

            elif role_raw == 'gpt' and loss is True:
                step_num += 1

                # 解析 <belief>, <reasoning>, <action>
                belief = None
                belief_json = None
                reasoning = None
                action = None

                belief_match = re.search(r'<belief>\s*(.*?)\s*</belief>', value, re.DOTALL)
                if belief_match:
                    belief = belief_match.group(1).strip()
                    try:
                        belief_json = json.loads(belief)
                    except json.JSONDecodeError:
                        belief_json = None

                reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', value, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()

                action_match = re.search(r'<action>\s*(.*?)\s*</action>', value, re.DOTALL)
                if action_match:
                    action = action_match.group(1).strip()

                # 提取信念状态的关键字段用于快速展示
                belief_summary = {}
                if belief_json:
                    pu = belief_json.get('product_understanding', {})
                    sp = belief_json.get('search_progress', {})
                    es = belief_json.get('exploration_state', {})
                    belief_summary = {
                        'product_match': pu.get('current_product_match', 'none'),
                        'price_constraint': pu.get('price_constraint', 'any'),
                        'search_status': sp.get('search_status', 'unknown'),
                        'subgoal': sp.get('updated_subgoal', ''),
                        'queries_count': len(es.get('queries_tried', [])),
                        'products_viewed_count': len(es.get('products_viewed', [])),
                        'options_selected_count': len(es.get('options_selected', [])),
                    }

                messages.append(Message(
                    role='agent',
                    content=value,
                    thought=reasoning,
                    action=action,
                    metadata={
                        'step': step_num,
                        'belief': belief,
                        'belief_json': belief_json,
                        'belief_summary': belief_summary,
                        'type': 'agent_response'
                    }
                ))

        # 确定状态 - 检查最后一个action是否是Buy Now
        last_action = ''
        for m in reversed(messages):
            if m.role == 'agent' and m.action:
                last_action = m.action
                break
        status = 'success' if 'buy now' in last_action.lower() else 'unknown'

        return Trajectory(
            id=item_id,
            task=task,
            status=status,
            steps=num_steps,
            task_type='webshop',
            messages=messages,
            environment='WebShop E-commerce Environment',
            metadata={
                'source': 'webshop_rebel',
                'annotation_success_rate': success_rate,
                'item_id': item_id
            }
        )


class WebShopRLJSONLAdapter(TrajectoryAdapter):
    """WebShop RL训练轨迹适配器

    加载由 extract_rl_trajectories.py 从训练日志提取的轨迹。
    支持 group_id 字段以实现同任务分组可视化，并标注每步reward。
    """

    def load(self, path: Path) -> List[Dict[str, Any]]:
        trajectories = []
        if path.is_dir():
            for f in sorted(path.glob('*.jsonl')):
                trajectories.extend(self._load_file(f))
        else:
            trajectories = self._load_file(path)
        return trajectories

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return items

    def parse(self, raw_item: Dict[str, Any], idx: int) -> 'Trajectory':
        task = raw_item.get('task', f'WebShop RL Task #{idx}')
        done = raw_item.get('done', 'False')
        total_reward = raw_item.get('total_reward', 0.0)
        group_id = raw_item.get('group_id', '')
        traj_index = raw_item.get('traj_index', idx)
        data = raw_item.get('data', [])

        import re

        def _extract_tag(text, tag):
            m = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL)
            return m.group(1).strip() if m else ''

        messages = []
        for step_data in data:
            step_num = step_data.get('step', 0)
            obs = step_data.get('obs', '')
            response = step_data.get('response', '')
            reward = step_data.get('reward', 0.0)
            is_valid = step_data.get('is_action_valid', True)

            # Extract from response if not directly provided
            action = step_data.get('action', '') or _extract_tag(response, 'action')
            reasoning = step_data.get('reasoning', '') or _extract_tag(response, 'reasoning')
            belief = step_data.get('belief', '') or _extract_tag(response, 'belief')

            # 环境观察
            if obs:
                messages.append(Message(
                    role='human',
                    content=obs,
                    metadata={'step': step_num, 'type': 'observation'}
                ))

            # Agent响应
            messages.append(Message(
                role='agent',
                content=response,
                thought=reasoning,
                action=action,
                metadata={
                    'step': step_num,
                    'type': 'agent_response',
                    'reward': reward,
                    'belief': belief,
                    'belief_json': belief,
                    'total_reward': total_reward,
                    'is_action_valid': is_valid,
                }
            ))

        # Prefer explicit 'won' field (saved by rollout_loop); fall back to reward threshold.
        # done='True' means agent clicked Buy Now; won=True means correct product purchased.
        # REBEL intrinsic rewards are at most ~1.5 total, so threshold=4.0 is safe.
        WIN_REWARD_THRESHOLD = 4.0
        won = raw_item.get('won', None)
        if won is True:
            status = 'success'
        elif done == 'True' and total_reward >= WIN_REWARD_THRESHOLD:
            status = 'success'
        elif done == 'True':
            status = 'failed'  # clicked Buy Now but wrong product
        else:
            status = 'failed'  # never reached Buy Now

        traj_id = raw_item.get('id', f'rl_traj_{idx:05d}')

        return Trajectory(
            id=traj_id,
            task=task,
            status=status,
            steps=len(data),
            task_type='webshop_rl',
            messages=messages,
            environment='WebShop RL Training',
            metadata={
                'source': 'webshop_rl',
                'total_reward': total_reward,
                'group_id': group_id,
                'traj_index': traj_index,
                'done': done,
                'exp_tag': raw_item.get('exp_tag', ''),
            }
        )


class RetrosynthesisJSONLAdapter(TrajectoryAdapter):
    """逆合成 JSONL 格式适配器

    支持逆合成环境的轨迹数据，格式为JSONL，每行一个轨迹。
    主要字段包括：target_molecule, success, steps 等
    """

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSONL 文件"""
        trajectories = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trajectories.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {e}")
        return trajectories

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """解析逆合成格式的轨迹"""
        # 基本信息
        trajectory_id = raw_item.get('trajectory_id', f'retro_{idx:05d}')
        target_molecule = raw_item.get('target_molecule', '')
        success = raw_item.get('success', False)
        total_steps = raw_item.get('total_steps', 0)
        total_reward = raw_item.get('total_reward', 0.0)
        pathway_length = raw_item.get('pathway_length', 0)
        epoch = raw_item.get('epoch', 0)
        prompt_id = raw_item.get('prompt_id', 0)

        steps_data = raw_item.get('steps', [])
        anchor_states = raw_item.get('anchor_states', [])
        final_pathway = raw_item.get('final_pathway', [])
        pathway_validity = raw_item.get('pathway_validity', False)
        ground_truth_length = raw_item.get('ground_truth_length', 0)

        messages = []

        # 添加初始环境描述
        env_description = f"目标分子: {target_molecule}\n逆合成规划任务"

        # 解析每个步骤
        for step_data in steps_data:
            step_num = step_data.get('step', 0)
            observation = step_data.get('observation', '')
            action_text = step_data.get('action', '')
            tool_name = step_data.get('tool_name', '')
            tool_args = step_data.get('tool_args', {})
            tool_response = step_data.get('tool_response', '')
            reward = step_data.get('reward', 0.0)
            cumulative_reward = step_data.get('cumulative_reward', 0.0)
            is_valid = step_data.get('is_valid_action', False)
            action_type = step_data.get('action_type', 'unknown')

            # 解析 action 中的 THINK 和 ACTION 部分
            thought = None
            action = None

            if '[THINK]' in action_text and '[/THINK]' in action_text:
                think_start = action_text.find('[THINK]') + len('[THINK]')
                think_end = action_text.find('[/THINK]')
                thought = action_text[think_start:think_end].strip()

            if '[ACTION]' in action_text and '[/ACTION]' in action_text:
                action_start = action_text.find('[ACTION]') + len('[ACTION]')
                action_end = action_text.find('[/ACTION]')
                action = action_text[action_start:action_end].strip()
            elif tool_name:
                # 如果没有解析到 action，使用 tool_name 和 tool_args
                action = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"

            # 先添加 agent 的动作和思考
            if action_text or action:
                messages.append(Message(
                    role='agent',
                    content=action_text if action_text else action,
                    thought=thought,
                    action=action,
                    metadata={
                        'step': step_num,
                        'tool_name': tool_name,
                        'tool_args': tool_args,
                        'reward': reward,
                        'cumulative_reward': cumulative_reward,
                        'is_valid_action': is_valid,
                        'action_type': action_type,
                        'type': 'agent_action'
                    }
                ))

            # 再添加工具响应（环境反馈）
            if tool_response:
                messages.append(Message(
                    role='human',
                    content=tool_response,
                    metadata={
                        'step': step_num,
                        'type': 'tool_response',
                        'tool_name': tool_name
                    }
                ))

        # 确定任务类型
        task_type = 'retrosynthesis'

        # 状态判断
        status = 'success' if success else 'failed'

        # 计算有效步数
        valid_steps = len([m for m in messages if m.role == 'agent' and m.metadata.get('is_valid_action', False)])

        return Trajectory(
            id=trajectory_id,
            task=f"逆合成规划: {target_molecule[:50]}..." if len(target_molecule) > 50 else f"逆合成规划: {target_molecule}",
            status=status,
            steps=total_steps,
            task_type=task_type,
            messages=messages,
            environment=env_description,
            metadata={
                'source': 'retrosynthesis',
                'target_molecule': target_molecule,
                'total_reward': total_reward,
                'pathway_length': pathway_length,
                'ground_truth_length': ground_truth_length,
                'epoch': epoch,
                'prompt_id': prompt_id,
                'valid_steps': valid_steps,
                'anchor_states': anchor_states,
                'final_pathway': final_pathway,
                'pathway_validity': pathway_validity
            }
        )


class RetrosynthesisJSONLDirAdapter(TrajectoryAdapter):
    """逆合成 JSONL 目录适配器

    支持从目录中加载多个 JSONL 文件
    """

    def __init__(self):
        self.jsonl_adapter = RetrosynthesisJSONLAdapter()

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载目录中所有 JSONL 文件"""
        all_items = []
        jsonl_files = sorted(path.glob('*.jsonl'))
        for jsonl_file in jsonl_files:
            items = self.jsonl_adapter.load(jsonl_file)
            all_items.extend(items)
        return all_items

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """使用基础 JSONL 适配器解析"""
        return self.jsonl_adapter.parse(raw_item, idx)


class TrajectoryLoader:
    """轨迹加载器 - 自动检测并使用合适的适配器"""

    def __init__(self):
        self.adapters = {
            'rebel_json': REBELJSONAdapter(),
            'webshop_rebel_jsonl': WebShopRebelJSONLAdapter(),
            'webshop_rl_jsonl': WebShopRLJSONLAdapter(),
            'retrosynthesis_jsonl': RetrosynthesisJSONLAdapter(),
            'retrosynthesis_jsonl_dir': RetrosynthesisJSONLDirAdapter(),
        }
        # Only add HuggingFace adapter if datasets library is available
        if DATASETS_AVAILABLE:
            self.adapters['huggingface'] = HuggingFaceDatasetAdapter()

    def detect_format(self, path: Path) -> Optional[str]:
        """自动检测轨迹格式"""
        if path.is_dir():
            # 检查是否是 HuggingFace dataset
            if DATASETS_AVAILABLE and (path / 'dataset_info.json').exists() and (path / 'state.json').exists():
                return 'huggingface'
            # 优先检查是否有 rollout_*.jsonl 文件（WebShop RL训练保存格式）
            rollout_files = list(path.glob('rollout_*.jsonl'))
            if rollout_files:
                return 'webshop_rl_jsonl'
            # 检查目录中是否有 JSONL 文件
            jsonl_files = list(path.glob('*.jsonl'))
            if jsonl_files:
                # 检查第一个 JSONL 文件内容来区分格式
                try:
                    with open(jsonl_files[0], 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            first_item = json.loads(first_line)
                            if 'conversations' in first_item and 'rebel' in first_item.get('item_id', '').lower():
                                return 'webshop_rebel_jsonl'
                            if 'group_id' in first_item and 'total_reward' in first_item:
                                return 'webshop_rl_jsonl'
                            if 'trajectory_id' in first_item or ('target_molecule' in first_item and 'steps' in first_item):
                                return 'retrosynthesis_jsonl_dir'
                except Exception:
                    pass
                return 'retrosynthesis_jsonl_dir'
        elif path.is_file():
            # 检查文件扩展名
            if path.suffix == '.jsonl':
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            first_item = json.loads(first_line)
                            # WebShop RL 训练轨迹格式（含group_id和total_reward）
                            if 'group_id' in first_item and 'total_reward' in first_item:
                                return 'webshop_rl_jsonl'
                            # WebShop ReBel hindsight 格式
                            if 'conversations' in first_item and ('rebel' in first_item.get('item_id', '').lower() or 'annotation_success_rate' in first_item):
                                return 'webshop_rebel_jsonl'
                            # 逆合成格式
                            if 'trajectory_id' in first_item or ('target_molecule' in first_item and 'steps' in first_item):
                                return 'retrosynthesis_jsonl'
                except Exception:
                    pass
            elif path.suffix == '.json':
                # 尝试读取文件内容判断格式
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            first_item = data[0]
                            # 检查是否是 REBEL 格式
                            if 'task' in first_item and 'done' in first_item and 'data' in first_item:
                                return 'rebel_json'
                except Exception:
                    pass
        return None

    def load(self, path: Path, format_type: Optional[str] = None) -> List[Trajectory]:
        """
        加载轨迹数据

        Args:
            path: 数据路径
            format_type: 格式类型，如果为 None 则自动检测

        Returns:
            轨迹列表
        """
        if format_type is None:
            format_type = self.detect_format(path)
            if format_type is None:
                raise ValueError(f"Cannot detect format for path: {path}")

        if format_type not in self.adapters:
            raise ValueError(f"Unsupported format: {format_type}")

        adapter = self.adapters[format_type]
        print(f"Loading trajectories from {path} using {format_type} adapter...")
        trajectories = adapter.load_and_parse(path)
        print(f"Loaded {len(trajectories)} trajectories")

        return trajectories

    def load_multiple(self, paths: List[Path]) -> List[Trajectory]:
        """加载多个数据源"""
        all_trajectories = []
        for path in paths:
            try:
                trajectories = self.load(path)
                all_trajectories.extend(trajectories)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
        return all_trajectories
