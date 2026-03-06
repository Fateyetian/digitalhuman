"""
测试逆合成轨迹适配器
"""
import sys
sys.path.insert(0, './backend')

from pathlib import Path
from trajectory_adapters import TrajectoryLoader, RetrosynthesisJSONLAdapter

def test_single_file():
    """测试单个 JSONL 文件加载"""
    print("=" * 60)
    print("测试单个 JSONL 文件加载")
    print("=" * 60)

    loader = TrajectoryLoader()
    test_file = Path("retro_traj/example01.jsonl")

    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return

    # 检测格式
    fmt = loader.detect_format(test_file)
    print(f"检测到的格式: {fmt}")

    # 加载轨迹
    trajectories = loader.load(test_file)
    print(f"加载了 {len(trajectories)} 条轨迹")

    # 显示前几条轨迹的信息
    for i, traj in enumerate(trajectories[:3]):
        print(f"\n--- 轨迹 {i+1} ---")
        print(f"ID: {traj.id}")
        print(f"任务: {traj.task[:80]}..." if len(traj.task) > 80 else f"任务: {traj.task}")
        print(f"状态: {traj.status}")
        print(f"步数: {traj.steps}")
        print(f"任务类型: {traj.task_type}")
        print(f"消息数量: {len(traj.messages)}")
        print(f"元数据: target_molecule={traj.metadata.get('target_molecule', '')[:30]}...")
        print(f"        total_reward={traj.metadata.get('total_reward', 0):.2f}")
        print(f"        valid_steps={traj.metadata.get('valid_steps', 0)}")

        # 显示前几条消息
        print(f"\n  前3条消息:")
        for j, msg in enumerate(traj.messages[:3]):
            content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
            print(f"    [{msg.role}] {content_preview}")
            if msg.thought:
                thought_preview = msg.thought[:40] + "..." if len(msg.thought) > 40 else msg.thought
                print(f"      思考: {thought_preview}")
            if msg.action:
                action_preview = msg.action[:40] + "..." if len(msg.action) > 40 else msg.action
                print(f"      动作: {action_preview}")

def test_directory():
    """测试目录加载"""
    print("\n" + "=" * 60)
    print("测试目录加载")
    print("=" * 60)

    loader = TrajectoryLoader()
    test_dir = Path("retro_traj")

    if not test_dir.exists():
        print(f"测试目录不存在: {test_dir}")
        return

    # 检测格式
    fmt = loader.detect_format(test_dir)
    print(f"检测到的格式: {fmt}")

    if fmt:
        trajectories = loader.load(test_dir)
        print(f"从目录加载了 {len(trajectories)} 条轨迹")

        # 统计
        success_count = sum(1 for t in trajectories if t.status == 'success')
        failed_count = sum(1 for t in trajectories if t.status == 'failed')
        total_steps = sum(t.steps for t in trajectories)
        avg_steps = total_steps / len(trajectories) if trajectories else 0

        print(f"\n统计信息:")
        print(f"  成功: {success_count}")
        print(f"  失败: {failed_count}")
        print(f"  平均步数: {avg_steps:.1f}")

def test_dict_conversion():
    """测试转换为字典格式"""
    print("\n" + "=" * 60)
    print("测试字典转换（用于 API 响应）")
    print("=" * 60)

    loader = TrajectoryLoader()
    test_file = Path("retro_traj/example01.jsonl")

    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return

    trajectories = loader.load(test_file)

    if trajectories:
        traj_dict = trajectories[0].to_dict()
        print(f"字典格式的键: {list(traj_dict.keys())}")
        print(f"消息数量: {len(traj_dict['messages'])}")
        print(f"元数据: {list(traj_dict['metadata'].keys())}")

if __name__ == "__main__":
    test_single_file()
    test_directory()
    test_dict_conversion()
    print("\n测试完成!")
