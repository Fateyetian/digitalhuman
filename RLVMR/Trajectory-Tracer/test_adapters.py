"""
测试轨迹适配器
验证不同格式的轨迹数据能否正确加载和解析
"""
import sys
from pathlib import Path

# 添加 backend 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from trajectory_adapters import TrajectoryLoader

def test_adapters():
    """测试所有适配器"""
    loader = TrajectoryLoader()
    base_path = Path(__file__).parent

    # 测试数据源
    test_sources = [
        {
            'name': 'HuggingFace Dataset',
            'path': base_path / 'alfworld_expert_traj',
            'expected_format': 'huggingface'
        },
        {
            'name': 'REBEL JSON',
            'path': base_path / 'alfworld_expert_traj' / 'rebel_coldstart_clean.json',
            'expected_format': 'rebel_json'
        }
    ]

    results = []

    for source in test_sources:
        print(f"\n{'='*60}")
        print(f"Testing: {source['name']}")
        print(f"Path: {source['path']}")
        print(f"{'='*60}")

        if not source['path'].exists():
            print(f"❌ SKIPPED: Path does not exist")
            results.append({'name': source['name'], 'status': 'skipped', 'reason': 'path not found'})
            continue

        try:
            # 检测格式
            detected_format = loader.detect_format(source['path'])
            print(f"Detected format: {detected_format}")

            if detected_format != source['expected_format']:
                print(f"⚠️  WARNING: Expected {source['expected_format']}, got {detected_format}")

            # 加载轨迹
            trajectories = loader.load(source['path'])
            print(f"✅ Loaded {len(trajectories)} trajectories")

            # 显示前3条轨迹的摘要
            print(f"\nFirst 3 trajectories:")
            for i, traj in enumerate(trajectories[:3]):
                traj_dict = traj.to_dict()
                print(f"\n  [{i+1}] {traj_dict['id']}")
                print(f"      Task: {traj_dict['task'][:60]}...")
                print(f"      Status: {traj_dict['status']}")
                print(f"      Type: {traj_dict['task_type']}")
                print(f"      Steps: {traj_dict['steps']}")
                print(f"      Messages: {len(traj_dict['messages'])}")
                print(f"      Source: {traj_dict['metadata'].get('source', 'unknown')}")

            # 统计信息
            statuses = {}
            task_types = {}
            for traj in trajectories:
                traj_dict = traj.to_dict()
                status = traj_dict['status']
                task_type = traj_dict['task_type']
                statuses[status] = statuses.get(status, 0) + 1
                task_types[task_type] = task_types.get(task_type, 0) + 1

            print(f"\nStatistics:")
            print(f"  By Status: {statuses}")
            print(f"  By Task Type: {task_types}")

            results.append({
                'name': source['name'],
                'status': 'success',
                'count': len(trajectories),
                'format': detected_format
            })

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': source['name'],
                'status': 'failed',
                'error': str(e)
            })

    # 总结
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for result in results:
        status_icon = '✅' if result['status'] == 'success' else '❌' if result['status'] == 'failed' else '⚠️'
        print(f"{status_icon} {result['name']}: {result['status']}")
        if result['status'] == 'success':
            print(f"   Format: {result['format']}, Count: {result['count']}")
        elif result['status'] == 'failed':
            print(f"   Error: {result['error']}")

    total_success = sum(1 for r in results if r['status'] == 'success')
    print(f"\nTotal: {total_success}/{len(results)} tests passed")

if __name__ == '__main__':
    test_adapters()
