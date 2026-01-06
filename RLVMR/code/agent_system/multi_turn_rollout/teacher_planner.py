"""
Teacher Model Planner for ReBel Framework

Uses a high-capability model (e.g., Claude Opus) to generate task plans
before the RL agent starts executing. This provides high-quality initial
belief states that the smaller RL model can then learn to follow.
"""

import json
import re
import os
import time
import hashlib
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 持久化缓存目录 - 使用绝对路径确保 Ray worker 也能正确访问
_CODE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录
CACHE_DIR = _CODE_DIR / "cache" / "teacher_plans"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TeacherPlanner:
    """
    Teacher model planner that uses Claude/GPT to generate task plans.

    This is used during RL training to provide high-quality initial plans,
    avoiding the cold-start problem where the untrained model generates
    invalid JSON plans.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        api_base: str = "https://api.yourapi.cn",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_workers: int = 16,
        cache_plans: bool = True,
    ):
        """
        Initialize the teacher planner.

        Args:
            model: Model identifier (e.g., "claude-opus-4-5-20251101")
            api_base: API base URL
            api_key: API key (can also be set via TEACHER_API_KEY env var)
            max_retries: Maximum retries for failed API calls
            timeout: Timeout for API calls in seconds
            max_workers: Maximum parallel API calls
            cache_plans: Whether to cache plans for identical prompts
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.model = model
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key or os.environ.get('TEACHER_API_KEY')
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = max_workers
        self.cache_plans = cache_plans

        # Initialize client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.api_base}/v1",
            timeout=self.timeout,
        )

        # Cache for plans (task_description -> plan)
        self._plan_cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'cache_hits': 0,
            'failed_calls': 0,
            'total_time': 0.0,
        }
        self._stats_lock = threading.Lock()

        print(f"[TeacherPlanner] Initialized with model={model}, api_base={api_base}")
        print(f"[TeacherPlanner] Persistent cache dir: {CACHE_DIR}")

        # Load existing cache from disk
        self._load_persistent_cache()

    def _get_cache_key(self, task_description: str, observation: str = "") -> str:
        """Generate a cache key from task description using hash."""
        # Use hash for consistent, short key
        content = task_description.strip()
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the cache file path for a key."""
        return CACHE_DIR / f"{cache_key}.json"

    def _load_persistent_cache(self):
        """Load all cached plans from disk."""
        count = 0
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self._plan_cache[cache_file.stem] = data['plan']
                    count += 1
            except:
                pass
        if count > 0:
            print(f"[TeacherPlanner] Loaded {count} cached plans from disk")

    def _save_to_persistent_cache(self, cache_key: str, task_desc: str, plan: dict):
        """Save a plan to persistent cache."""
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'task_description': task_desc[:200],  # Save first 200 chars for reference
                    'plan': plan,
                    'timestamp': time.time()
                }, f, indent=2)
        except Exception as e:
            print(f"[TeacherPlanner] Failed to save cache: {e}")

    def _parse_json_response(self, response_text: str) -> Optional[dict]:
        """Parse JSON from model response, handling various formats."""
        # Try direct JSON parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in the response
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',       # Generic code block
            r'\{[\s\S]*\}',                   # Raw JSON object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def _validate_plan(self, plan: dict) -> bool:
        """Validate that a plan has the required structure."""
        required_fields = ['main_goal', 'plan_steps']
        for field in required_fields:
            if field not in plan:
                return False

        # Validate plan_steps is a non-empty list
        if not isinstance(plan.get('plan_steps'), list) or len(plan['plan_steps']) == 0:
            return False

        return True

    def _call_api_single(self, prompt: str, index: int) -> tuple:
        """Make a single API call with retries."""
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert task planner. Output ONLY valid JSON, no other text."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Low temperature for consistent planning
                    max_tokens=1024,
                )

                response_text = response.choices[0].message.content
                plan = self._parse_json_response(response_text)

                if plan and self._validate_plan(plan):
                    elapsed = time.time() - start_time
                    with self._stats_lock:
                        self.stats['successful_calls'] += 1
                        self.stats['total_time'] += elapsed
                    return (index, plan)
                else:
                    # Invalid response, retry
                    if attempt < self.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"[TeacherPlanner] API call failed for index {index}: {e}")

        with self._stats_lock:
            self.stats['failed_calls'] += 1
        return (index, None)

    def generate_plans(self, planning_prompts: List[str]) -> List[Optional[dict]]:
        """
        Generate task plans for a batch of prompts.
        Uses persistent cache to avoid redundant API calls.

        Args:
            planning_prompts: List of planning prompt strings

        Returns:
            List of plan dicts (or None for failed generations)
        """
        batch_size = len(planning_prompts)
        results = [None] * batch_size
        prompts_to_call = []
        prompt_task_map = {}  # index -> (cache_key, task_desc)

        with self._stats_lock:
            self.stats['total_calls'] += batch_size

        # Check cache first (both memory and disk)
        cache_hits = 0
        for i, prompt in enumerate(planning_prompts):
            # Extract task description from prompt for cache key
            task_match = re.search(r'【Task Description】\s*\n(.*?)(?:\n\n|\n【)', prompt, re.DOTALL)
            if task_match:
                task_desc = task_match.group(1).strip()
                cache_key = self._get_cache_key(task_desc)
                prompt_task_map[i] = (cache_key, task_desc)

                # Check memory cache first, then disk
                with self._cache_lock:
                    if cache_key in self._plan_cache:
                        results[i] = self._plan_cache[cache_key]
                        cache_hits += 1
                        continue

                prompts_to_call.append((i, prompt))
            else:
                prompts_to_call.append((i, prompt))
                prompt_task_map[i] = (None, "")

        with self._stats_lock:
            self.stats['cache_hits'] += cache_hits

        if not prompts_to_call:
            print(f"[TeacherPlanner] All {batch_size} plans loaded from cache!")
            return results

        # Make parallel API calls
        print(f"[TeacherPlanner] Generating {len(prompts_to_call)} new plans (batch: {batch_size}, cached: {cache_hits})")

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts_to_call))) as executor:
            futures = {
                executor.submit(self._call_api_single, prompt, idx): idx
                for idx, prompt in prompts_to_call
            }

            for future in as_completed(futures):
                idx, plan = future.result()
                results[idx] = plan

                # Cache successful plans (both memory and disk)
                if plan:
                    cache_key, task_desc = prompt_task_map.get(idx, (None, ""))
                    if cache_key:
                        with self._cache_lock:
                            self._plan_cache[cache_key] = plan
                        # Save to persistent cache
                        self._save_to_persistent_cache(cache_key, task_desc, plan)

        # Log statistics
        valid_count = sum(1 for r in results if r is not None)
        print(f"[TeacherPlanner] Generated {valid_count}/{batch_size} valid plans (new API calls: {len(prompts_to_call)}, cached: {cache_hits})")

        return results

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._stats_lock:
            stats = self.stats.copy()

        if stats['successful_calls'] > 0:
            stats['avg_time_per_call'] = stats['total_time'] / stats['successful_calls']
        else:
            stats['avg_time_per_call'] = 0.0

        stats['cache_size'] = len(self._plan_cache)
        return stats

    def clear_cache(self):
        """Clear the plan cache."""
        with self._cache_lock:
            self._plan_cache.clear()


# Global instance for easy access
_teacher_planner: Optional[TeacherPlanner] = None


def get_teacher_planner(
    model: str = "claude-opus-4-5-20251101",
    api_base: str = "https://api.yourapi.cn",
    api_key: Optional[str] = None,
    **kwargs
) -> TeacherPlanner:
    """
    Get or create the global teacher planner instance.

    Args:
        model: Model identifier
        api_base: API base URL
        api_key: API key
        **kwargs: Additional arguments for TeacherPlanner

    Returns:
        TeacherPlanner instance
    """
    global _teacher_planner

    if _teacher_planner is None:
        _teacher_planner = TeacherPlanner(
            model=model,
            api_base=api_base,
            api_key=api_key,
            **kwargs
        )

    return _teacher_planner


def generate_plans_with_teacher(
    planning_prompts: List[str],
    model: str = "claude-opus-4-5-20251101",
    api_base: str = "https://api.yourapi.cn",
    api_key: Optional[str] = None,
) -> List[Optional[dict]]:
    """
    Convenience function to generate plans using teacher model.

    Args:
        planning_prompts: List of planning prompt strings
        model: Model identifier
        api_base: API base URL
        api_key: API key

    Returns:
        List of plan dicts (or None for failed generations)
    """
    planner = get_teacher_planner(model=model, api_base=api_base, api_key=api_key)
    return planner.generate_plans(planning_prompts)
