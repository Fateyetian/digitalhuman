"""
简单测试 REBEL JSON 适配器
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class Message:
    """统一的消息格式"""
    def __init__(self, role: str, content: str, thought: Optional[str] = None,
                 action: Optional[str] = None, metadata: Optional[Dict] = None):
        self.role = role
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
        self.status = status
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


def parse_rebel_trajectory(raw_item: Dict[str, Any], idx: int) -> Trajectory:
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

            # 尝试解析结构化响应
            if '<belief>' in response and '</belief>' in response:
                belief_start = response.find('<belief>') + len('<belief>')
                belief_end = response.find('</belief>')
                belief = response[belief_start:belief_end].strip()

            if '<reasoning>' in response and '</reasoning>' in response:
                reasoning_start = response.find('<reasoning>') + len('<reasoning>')
                reasoning_end = response.find('</reasoning>')
                thought = response[reasoning_start:reasoning_end].strip()

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

    # 环境信息
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


def test_rebel_adapter():
    """测试 REBEL JSON 适配器"""
    print("Testing REBEL JSON Adapter")
    print("=" * 60)

    json_path = Path("alfworld_expert_traj/rebel_coldstart_clean.json")

    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        return

    print(f"Loading: {json_path}")
    print(f"File size: {json_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"[OK] Loaded {len(raw_data)} raw trajectories")

        # 解析前 5 条轨迹
        trajectories = []
        for idx in range(min(5, len(raw_data))):
            traj = parse_rebel_trajectory(raw_data[idx], idx)
            trajectories.append(traj)

        print(f"[OK] Parsed {len(trajectories)} trajectories\n")

        # 显示详细信息
        for i, traj in enumerate(trajectories):
            traj_dict = traj.to_dict()
            print(f"[{i+1}] {traj_dict['id']}")
            print(f"    Task: {traj_dict['task']}")
            print(f"    Status: {traj_dict['status']}")
            print(f"    Type: {traj_dict['task_type']}")
            print(f"    Steps: {traj_dict['steps']}")
            print(f"    Messages: {len(traj_dict['messages'])}")
            print(f"    Environment length: {len(traj_dict['environment'])} chars")

            # 显示第一个 agent 消息
            agent_msgs = [m for m in traj_dict['messages'] if m['role'] == 'agent']
            if agent_msgs:
                first_agent = agent_msgs[0]
                print(f"    First action: {first_agent['action'][:50] if first_agent['action'] else 'None'}...")
                if first_agent['thought']:
                    print(f"    First thought: {first_agent['thought'][:50]}...")
            print()

        print("=" * 60)
        print("[OK] Test passed!")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_rebel_adapter()
