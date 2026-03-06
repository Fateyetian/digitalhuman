#!/usr/bin/env python3
"""
WebShop ReBel SFT Data Pipeline — Stratified Sampling + Batch Annotation

1. Stratified sample 500 trajectories from 3930 expert pool
2. Annotate with Teacher LLM using hindsight method
3. Filter for 100% annotation success rate
4. Generate cold-start parquet for SFT training
"""

import os
import sys
import json
import re
import random
import time
import argparse
import statistics
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, '/root/testttt/RLVMR/code')

from openai import OpenAI
from scripts.generate_webshop_rebel_hindsight import (
    generate_webshop_rebel_dataset,
    convert_to_coldstart_format,
)


# ============================================================================
# Part 1: Stratified Sampling
# ============================================================================

ATTR_KEYWORDS = [
    'machine wash', 'wash cold', 'cotton', 'polyester', 'spandex', 'rubber',
    'leather', 'moisture', 'slip resist', 'classic fit', 'loose fit', 'slim fit',
    'short sleeve', 'long sleeve', 'lace', 'zipper', 'elastic', 'nylon',
    'rayon', 'fleece', 'waterproof', 'breathable', 'quick dry',
]


def extract_task(conversations: List[Dict]) -> str:
    for turn in conversations:
        if turn.get('from') == 'human':
            m = re.search(r'Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]', turn.get('value', ''))
            if m:
                return m.group(1).strip()
    return ''


def classify_category(task: str) -> str:
    tl = task.lower()
    # Footwear (check before clothing to catch "men's boots")
    if any(w in tl for w in ['shoe', 'boot', 'sneaker', 'sandal', 'loafer', 'slipper', 'clog', 'mule']):
        return 'footwear'
    if "men's" in tl and "women" not in tl:
        return 'mens_clothing'
    if "women" in tl:
        return 'womens_clothing'
    if any(w in tl for w in ['cabinet', 'desk', 'chair', 'table', 'shelf', 'lamp', 'rug',
                              'curtain', 'bed', 'sofa', 'furniture', 'drawer', 'bookcase',
                              'organizer', 'storage', 'rack']):
        return 'furniture_home'
    return 'other'


def compute_complexity(task: str) -> int:
    tl = task.lower()
    return sum(1 for kw in ATTR_KEYWORDS if kw in tl)


def stratified_sample(
    all_data: List[Dict],
    total_n: int = 500,
    seed: int = 42
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Stratified sampling:
    - By category (mens_clothing, womens_clothing, footwear, furniture_home, other)
    - Within each category, 50% from complexity >= 3, 50% from complexity < 3
    """
    random.seed(seed)

    # Classify each trajectory
    indexed = []
    for i, t in enumerate(all_data):
        task = extract_task(t['conversations'])
        cat = classify_category(task)
        comp = compute_complexity(task)
        steps = sum(1 for c in t['conversations'] if c.get('from') == 'gpt' and c.get('loss') is True)
        indexed.append({
            'idx': i, 'data': t, 'task': task,
            'category': cat, 'complexity': comp, 'steps': steps
        })

    # Target allocation
    allocation = {
        'mens_clothing':   int(total_n * 0.35),  # 175
        'womens_clothing': int(total_n * 0.30),  # 150
        'footwear':        int(total_n * 0.15),  # 75
        'furniture_home':  int(total_n * 0.10),  # 50
        'other':           int(total_n * 0.10),  # 50
    }
    # Adjust rounding
    diff = total_n - sum(allocation.values())
    if diff != 0:
        allocation['mens_clothing'] += diff

    # Group by category
    by_cat = {}
    for item in indexed:
        by_cat.setdefault(item['category'], []).append(item)

    sampled = []
    report = {'allocation': {}, 'pool_sizes': {}}

    for cat, target_n in allocation.items():
        pool = by_cat.get(cat, [])
        random.shuffle(pool)

        # Split by complexity
        high_comp = [x for x in pool if x['complexity'] >= 3]
        low_comp = [x for x in pool if x['complexity'] < 3]
        random.shuffle(high_comp)
        random.shuffle(low_comp)

        half = target_n // 2
        # Take half from high, half from low; fill remainder from whichever has more
        from_high = high_comp[:half]
        from_low = low_comp[:half]
        remaining = target_n - len(from_high) - len(from_low)

        if remaining > 0:
            leftover_high = high_comp[half:]
            leftover_low = low_comp[half:]
            leftover = leftover_high + leftover_low
            random.shuffle(leftover)
            from_rest = leftover[:remaining]
        else:
            from_rest = []

        cat_sampled = from_high + from_low + from_rest
        sampled.extend(cat_sampled)

        report['allocation'][cat] = {
            'target': target_n, 'sampled': len(cat_sampled),
            'pool_total': len(pool),
            'high_complexity': len(from_high), 'low_complexity': len(from_low),
            'fill': len(from_rest)
        }
        report['pool_sizes'][cat] = len(pool)

    random.shuffle(sampled)  # Shuffle final order

    report['total_sampled'] = len(sampled)
    report['avg_complexity'] = statistics.mean(x['complexity'] for x in sampled)
    report['avg_steps'] = statistics.mean(x['steps'] for x in sampled)
    report['complexity_dist'] = dict(Counter(x['complexity'] for x in sampled).most_common())

    return [x['data'] for x in sampled], report


# ============================================================================
# Part 2: Batch Annotation with Incremental Save
# ============================================================================

def annotate_batch(
    trajectories: List[Dict],
    client: OpenAI,
    model_name: str,
    output_dir: str,
    save_interval: int = 20,
    max_workers: int = 3,
) -> List[Dict]:
    """Annotate trajectories with incremental saving."""

    jsonl_path = os.path.join(output_dir, 'rebel_hindsight.jsonl')
    progress_path = os.path.join(output_dir, 'annotation_progress.json')

    # Resume from previous progress if exists
    completed = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    completed[t['item_id']] = t
                except:
                    pass
        print(f"Resuming: {len(completed)} trajectories already annotated")

    results = list(completed.values())
    total = len(trajectories)
    new_count = 0
    fail_count = 0

    pbar = tqdm(total=total, desc="Annotating", ncols=100,
                initial=len(completed))

    for i, traj in enumerate(trajectories):
        item_id = traj.get('item_id', f'unknown_{i}')
        expected_id = f"{item_id}_rebel_hindsight"

        # Skip if already done
        if expected_id in completed:
            continue

        try:
            result = generate_webshop_rebel_dataset(traj, client, model_name)

            if result and result['annotation_success_rate'] > 0:
                results.append(result)
                completed[result['item_id']] = result
                new_count += 1
            else:
                fail_count += 1
                tqdm.write(f"  [{item_id}] Failed annotation")

        except Exception as e:
            fail_count += 1
            tqdm.write(f"  [{item_id}] Error: {e}")

        pbar.update(1)

        # Incremental save
        if (new_count > 0 and new_count % save_interval == 0) or (i == total - 1):
            with open(jsonl_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

            # Save progress report
            progress = {
                'total_target': total,
                'completed': len(results),
                'new_this_run': new_count,
                'failed': fail_count,
                'success_rate': len(results) / max(1, len(results) + fail_count),
            }
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)

    pbar.close()

    # Final save
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return results


# ============================================================================
# Part 3: Post-processing & Cold-Start Generation
# ============================================================================

def filter_and_generate_coldstart(
    results: List[Dict],
    output_dir: str,
    min_success_rate: float = 1.0
) -> Dict[str, Any]:
    """Filter for quality and generate cold-start parquet."""

    # Filter
    passed = [r for r in results if r['annotation_success_rate'] >= min_success_rate]
    failed = [r for r in results if r['annotation_success_rate'] < min_success_rate]

    print(f"\nQuality filter (success_rate >= {min_success_rate}):")
    print(f"  Passed: {len(passed)}/{len(results)} ({len(passed)/max(1,len(results))*100:.1f}%)")
    if failed:
        print(f"  Failed items: {[r['item_id'] for r in failed[:10]]}")

    # Save filtered JSONL
    filtered_path = os.path.join(output_dir, 'rebel_hindsight_filtered.jsonl')
    with open(filtered_path, 'w') as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Generate cold-start pairs
    all_cs_pairs = []
    for r in passed:
        pairs = convert_to_coldstart_format(r)
        all_cs_pairs.extend(pairs)

    print(f"  Cold-start pairs: {len(all_cs_pairs)} steps from {len(passed)} trajectories")

    # Save parquet
    if all_cs_pairs:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({
                'extra_info': [
                    {'question': p['question'], 'answer': p['answer']}
                    for p in all_cs_pairs
                ]
            })
            parquet_path = os.path.join(output_dir, 'train.parquet')
            pq.write_table(table, parquet_path)
            print(f"  Saved parquet: {parquet_path}")
        except ImportError:
            print("  WARNING: pyarrow not installed, skipping parquet generation")
            # Save as JSON fallback
            json_path = os.path.join(output_dir, 'train_coldstart.json')
            with open(json_path, 'w') as f:
                json.dump(all_cs_pairs, f, ensure_ascii=False, indent=2)
            print(f"  Saved JSON fallback: {json_path}")

    report = {
        'total_annotated': len(results),
        'passed_filter': len(passed),
        'failed_filter': len(failed),
        'pass_rate': len(passed) / max(1, len(results)),
        'total_steps': sum(r['num_steps'] for r in passed),
        'avg_steps': statistics.mean(r['num_steps'] for r in passed) if passed else 0,
        'total_coldstart_pairs': len(all_cs_pairs),
    }

    report_path = os.path.join(output_dir, 'sft_data_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WebShop ReBel SFT Data Pipeline")
    parser.add_argument('--expert_data', type=str,
                        default='data/webshop_expert_traj/webshop_train.json')
    parser.add_argument('--output_dir', type=str,
                        default='data/webshop_rebel_sft')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--teacher_model_url', type=str,
                        default='https://www.dmxapi.cn/v1')
    parser.add_argument('--teacher_model_name', type=str,
                        default='claude-sonnet-4-6-cc')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--skip_annotation', action='store_true',
                        help='Skip annotation, only do post-processing on existing data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("WebShop ReBel SFT Data Pipeline")
    print("=" * 80)

    # ---- Step 1: Load & Sample ----
    print("\n[1/4] Loading expert trajectories...")
    with open(args.expert_data) as f:
        all_data = json.load(f)
    print(f"  Pool: {len(all_data)} trajectories")

    sample_path = os.path.join(args.output_dir, 'sampled_trajectories.json')
    sample_report_path = os.path.join(args.output_dir, 'sampling_report.json')

    if os.path.exists(sample_path):
        print(f"  Loading existing sample from {sample_path}")
        with open(sample_path) as f:
            sampled = json.load(f)
        with open(sample_report_path) as f:
            sample_report = json.load(f)
    else:
        print(f"\n[2/4] Stratified sampling {args.num_samples} trajectories (seed={args.seed})...")
        sampled, sample_report = stratified_sample(all_data, args.num_samples, args.seed)

        with open(sample_path, 'w') as f:
            json.dump(sampled, f, ensure_ascii=False)
        with open(sample_report_path, 'w') as f:
            json.dump(sample_report, f, indent=2)

    print(f"  Sampled: {sample_report['total_sampled']} trajectories")
    print(f"  Avg complexity: {sample_report['avg_complexity']:.1f}")
    print(f"  Avg steps: {sample_report['avg_steps']:.1f}")
    print(f"  Allocation:")
    for cat, info in sample_report['allocation'].items():
        print(f"    {cat}: {info['sampled']}/{info['pool_total']} "
              f"(high={info['high_complexity']}, low={info['low_complexity']}, fill={info['fill']})")

    if args.skip_annotation:
        print("\n[3/4] Skipping annotation (--skip_annotation)")
    else:
        # ---- Step 2: Annotate ----
        print(f"\n[3/4] Annotating {len(sampled)} trajectories...")
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY", "EMPTY")
        client = OpenAI(api_key=api_key, base_url=args.teacher_model_url)

        # Quick connectivity check
        try:
            resp = client.chat.completions.create(
                model=args.teacher_model_name,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            print(f"  Connected: {args.teacher_model_url} ({args.teacher_model_name})")
        except Exception as e:
            print(f"  Connection failed: {e}")
            return

        results = annotate_batch(
            sampled, client, args.teacher_model_name,
            args.output_dir, args.save_interval
        )

    # ---- Step 3: Post-process ----
    print(f"\n[4/4] Post-processing & generating cold-start data...")
    jsonl_path = os.path.join(args.output_dir, 'rebel_hindsight.jsonl')
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            results = [json.loads(line) for line in f if line.strip()]
    else:
        print("  No annotation results found!")
        return

    report = filter_and_generate_coldstart(results, args.output_dir)

    print(f"\n{'=' * 80}")
    print("Pipeline Complete!")
    print(f"{'=' * 80}")
    print(f"  Total annotated:     {report['total_annotated']}")
    print(f"  Passed filter:       {report['passed_filter']}")
    print(f"  Pass rate:           {report['pass_rate']:.1%}")
    print(f"  Total SFT steps:     {report['total_steps']}")
    print(f"  Avg steps/traj:      {report['avg_steps']:.1f}")
    print(f"  Cold-start pairs:    {report['total_coldstart_pairs']}")
    print(f"  Output dir:          {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
