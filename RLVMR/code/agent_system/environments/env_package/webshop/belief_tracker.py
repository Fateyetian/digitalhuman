"""
WebShop ReBel (Reward Belief) Framework - Belief State Tracking and Intrinsic Reward Calculation

This module implements WebShop-specific components of ReBel:
1. Belief state parsing from model outputs (WebShop belief schema)
2. Ground truth state tracking from WebShop observations
3. Three intrinsic rewards adapted for e-commerce: consistency, progress, exploration
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class WebShopBeliefStateParser:
    """Parse belief states from model output for WebShop environment"""

    @staticmethod
    def parse_belief(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract belief state from model output.
        Expected format:
        <belief>
        {
          "product_understanding": {
            "target_attributes": {"attr": "value"},
            "current_product_match": "partial/exact/none",
            "price_constraint": "under $X / any"
          },
          "attribute_verification": {
            "verified": [{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "..."}],
            "unverified": ["material"],
            "inferred_only": [{"attribute": "...", "inference": "...", "required_action": "..."}]
          },
          "search_progress": {
            "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
            "evidence": "...",
            "updated_subgoal": "..."
          },
          "exploration_state": {
            "queries_tried": ["query1", "query2"],
            "products_viewed": ["product1"],
            "options_selected": ["option1"],
            "tabs_clicked": ["Description"]
          }
        }
        </belief>
        """
        belief_match = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL | re.IGNORECASE)
        if not belief_match:
            return None

        belief_text = belief_match.group(1).strip()

        try:
            belief_json = belief_text.replace("'", '"')
            belief_data = json.loads(belief_json)

            if isinstance(belief_data, dict) and (
                'product_understanding' in belief_data or
                'search_progress' in belief_data or
                'exploration_state' in belief_data
            ):
                return {
                    'product_understanding': belief_data.get('product_understanding', {}),
                    'search_progress': belief_data.get('search_progress', {}),
                    'exploration_state': belief_data.get('exploration_state', {})
                }

        except json.JSONDecodeError:
            pass

        return None


class WebShopGroundTruthTracker:
    """Track ground truth state by parsing WebShop observations"""

    def __init__(self):
        self.page_type = 'search'  # search, results, product, done
        self.products_seen = []
        self.options_clicked = []
        self.search_queries = []
        self.current_product_title = None
        self.current_product_price = None
        self.current_product_options = {}
        self.available_clickables = []
        self.has_search_bar = False
        self.task_goal = None
        self.interaction_history = []

    def update_from_observation(self, obs: str, action: str = None, info: dict = None):
        """Update ground truth state from WebShop observation text and info"""
        obs_lower = obs.lower()

        # Parse page type from observation structure
        if 'search' in obs_lower and '[sep]' not in obs_lower:
            self.page_type = 'search'
        elif 'back to search' in obs_lower:
            # Product or results page
            if 'buy now' in obs_lower:
                self.page_type = 'product'
            else:
                self.page_type = 'results'
        elif 'thank you' in obs_lower or 'your score' in obs_lower:
            self.page_type = 'done'

        # Parse available actions from info
        if info and 'available_actions' in info:
            avail = info['available_actions']
            self.has_search_bar = avail.get('has_search_bar', False)
            self.available_clickables = avail.get('clickables', [])

        # Track actions
        if action:
            action_lower = action.lower().strip()
            self.interaction_history.append(action_lower)

            # Track search queries
            search_match = re.match(r'search\[(.+)\]', action_lower)
            if search_match:
                query = search_match.group(1).strip()
                if query not in self.search_queries:
                    self.search_queries.append(query)
                self.page_type = 'results'

            # Track click actions
            click_match = re.match(r'click\[(.+)\]', action_lower)
            if click_match:
                clicked = click_match.group(1).strip()
                if clicked == 'buy now':
                    self.page_type = 'done'
                elif clicked == 'back to search':
                    self.page_type = 'search'
                elif clicked.startswith('b0') or re.match(r'^b\d+', clicked):
                    # Product ASIN click - viewing a product
                    if clicked not in self.products_seen:
                        self.products_seen.append(clicked)
                    self.page_type = 'product'
                else:
                    # Option selection or navigation
                    if clicked not in self.options_clicked:
                        self.options_clicked.append(clicked)

        # Try to extract product info from observation
        self._parse_product_info(obs)

    def _parse_product_info(self, obs: str):
        """Extract product information from observation text"""
        parts = obs.split(' [SEP] ')
        parts_lower = [p.lower().strip() for p in parts]

        # Look for price pattern
        for part in parts:
            price_match = re.search(r'\$[\d.]+', part)
            if price_match:
                self.current_product_price = price_match.group(0)

        # The first substantial text after page type indicators is usually the title
        for part in parts:
            part_stripped = part.strip()
            if len(part_stripped) > 20 and '$' not in part_stripped and part_stripped.lower() not in [
                'back to search', 'buy now', 'search', '< prev', 'next >'
            ]:
                self.current_product_title = part_stripped
                break

    def get_ground_truth_state(self) -> Dict[str, Any]:
        """Return current ground truth state"""
        return {
            'page_type': self.page_type,
            'products_seen': self.products_seen.copy(),
            'options_clicked': self.options_clicked.copy(),
            'search_queries': self.search_queries.copy(),
            'current_product_title': self.current_product_title,
            'current_product_price': self.current_product_price,
            'available_clickables': self.available_clickables.copy(),
            'has_search_bar': self.has_search_bar,
            'task_goal': self.task_goal,
            'interactions': self.interaction_history.copy()
        }


class WebShopRebelRewardCalculator:
    """
    Calculate three intrinsic rewards for WebShop ReBel framework.

    Reuses the same weight/decay infrastructure as ALFWorld's RebelRewardCalculator,
    but overrides the three component reward calculations for the WebShop domain:
    - Consistency: product understanding matches actual product page
    - Progress: search_status accuracy, evidence quality, subgoal logic
    - Exploration: efficient searching, diverse queries, not revisiting products
    """

    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, delta=0.1, use_belief_reward=True,
                 belief_reward_decay_enable=False, belief_reward_decay_method='cosine',
                 belief_reward_warmup_epochs=5, belief_reward_decay_start_epoch=10,
                 belief_reward_decay_end_epoch=60, belief_reward_min_weight=0.1,
                 belief_reward_adaptive_decay=False,
                 belief_reward_target_sr=0.90,
                 belief_reward_decay_alpha=2.0,
                 belief_reward_progress_decay_rate=0.7,
                 belief_reward_consistency_decay_rate=1.0,
                 belief_reward_exploration_decay_rate=2.0):
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
        self.current_epoch = 0

        # V11: Adaptive decay parameters
        self.belief_reward_adaptive_decay = belief_reward_adaptive_decay
        self.belief_reward_target_sr = belief_reward_target_sr
        self.belief_reward_decay_alpha = belief_reward_decay_alpha
        self.current_success_rate = None

        # V11: Differential component decay rates
        self.progress_decay_rate = belief_reward_progress_decay_rate
        self.consistency_decay_rate = belief_reward_consistency_decay_rate
        self.exploration_decay_rate = belief_reward_exploration_decay_rate

        # Reward scales (tuned for WebShop's continuous [0,1] reward)
        self.consistency_scale = 0.1
        self.progress_scale = 0.1
        self.exploration_scale = 0.05

        # Format validity rewards
        self.format_valid_reward = 0.01
        self.format_invalid_penalty = -0.05
        self.action_invalid_penalty = -0.02

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_success_rate(self, success_rate: float):
        self.current_success_rate = success_rate

    def get_belief_reward_weight(self) -> float:
        """V9/V11: Calculate belief reward weight based on epoch and success rate."""
        import math

        if not self.belief_reward_decay_enable:
            return 1.0 if self.use_belief_reward else 0.0

        epoch = self.current_epoch
        warmup = self.belief_reward_warmup_epochs
        decay_start = self.belief_reward_decay_start_epoch
        decay_end = self.belief_reward_decay_end_epoch
        min_weight = self.belief_reward_min_weight

        if epoch < warmup:
            return epoch / warmup if warmup > 0 else 1.0

        base_weight = 1.0

        if self.belief_reward_adaptive_decay and self.current_success_rate is not None and epoch >= warmup:
            sr = self.current_success_rate
            target_sr = self.belief_reward_target_sr
            alpha = self.belief_reward_decay_alpha
            if target_sr > 0:
                decay_factor = max(min_weight, 1.0 - (sr / target_sr) ** alpha)
            else:
                decay_factor = 1.0
            base_weight = base_weight * decay_factor

        if not self.belief_reward_adaptive_decay and epoch < decay_start:
            return 1.0

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
                base_weight = min(base_weight, cosine_weight)
            else:
                base_weight = cosine_weight

        return max(min_weight, base_weight)

    def compute_component_weights(self, base_weight: float) -> Dict[str, float]:
        """V11: Compute per-component weights with differential decay rates."""
        if base_weight <= 0:
            return {'progress': 0.0, 'consistency': 0.0, 'exploration': 0.0, 'format': 1.0}

        bw = max(1e-8, min(1.0, base_weight))
        return {
            'progress': bw ** self.progress_decay_rate,
            'consistency': bw ** self.consistency_decay_rate,
            'exploration': bw ** self.exploration_decay_rate,
            'format': 1.0,
        }

    def calculate_consistency_reward(
        self,
        belief_product: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int = -1,
        belief_state: Dict[str, Any] = None
    ) -> float:
        """
        Calculate consistency reward for WebShop.
        Measures: Does the agent's product_understanding match the actual product page?

        Components:
        - target_attributes reasonableness (35%)
        - current_product_match accuracy (25%)
        - price_constraint awareness (25%)
        - attribute_verification quality (15%)
        """
        if not belief_product:
            return 0.0

        components = []

        # Component 1: Target attributes (35% weight)
        target_attrs = belief_product.get('target_attributes', {})
        if not isinstance(target_attrs, dict):
            target_attrs = {}

        if target_attrs and len(target_attrs) > 0:
            # Has extracted target attributes from the task instruction
            task_goal = ground_truth.get('task_goal', '') or ''
            if task_goal:
                # Check if attributes reference terms from the task
                task_lower = task_goal.lower()
                matches = 0
                for attr, val in target_attrs.items():
                    val_str = str(val).lower()
                    attr_str = str(attr).lower()
                    if val_str in task_lower or attr_str in task_lower:
                        matches += 1
                if len(target_attrs) > 0:
                    components.append(min(1.0, 0.5 + 0.5 * matches / len(target_attrs)))
                else:
                    components.append(0.5)
            else:
                components.append(0.6)  # Has attributes but can't verify
        else:
            components.append(0.3)  # No target attributes extracted

        # Component 2: Product match assessment (25% weight)
        match_level = str(belief_product.get('current_product_match', 'none')).lower()
        page_type = ground_truth.get('page_type', 'search')

        if page_type == 'product':
            # On a product page - match assessment should be meaningful
            if match_level in ('exact', 'partial', 'none'):
                components.append(0.8)  # Valid assessment
            else:
                components.append(0.4)  # Invalid assessment format
        elif page_type in ('search', 'results'):
            # Not on a product page
            if match_level == 'none':
                components.append(0.8)  # Correct: no product to match
            else:
                components.append(0.5)  # Debatable but not wrong
        else:
            components.append(0.5)  # Neutral

        # Component 3: Price constraint awareness (25% weight)
        price_constraint = str(belief_product.get('price_constraint', 'any')).lower()
        task_goal = ground_truth.get('task_goal', '') or ''

        # Check if task mentions price
        price_match = re.search(r'(?:under|less than|below|cheaper than)\s*\$?([\d.]+)', task_goal.lower())
        if price_match:
            # Task has a price constraint
            if 'under' in price_constraint or '$' in price_constraint:
                components.append(0.9)  # Correctly identified price constraint
            else:
                components.append(0.3)  # Missed the price constraint
        else:
            # No explicit price constraint in task
            if 'any' in price_constraint or 'no' in price_constraint:
                components.append(0.8)  # Correctly identified no constraint
            else:
                components.append(0.5)  # Neutral

        # Component 4: Attribute verification quality (15% weight)
        av_score = 0.5  # Default neutral
        av = {}
        if belief_state and isinstance(belief_state, dict):
            av = belief_state.get('attribute_verification', {})
        if isinstance(av, dict):
            verified = av.get('verified', [])
            unverified = av.get('unverified', [])
            inferred_only = av.get('inferred_only', [])

            if not isinstance(verified, list):
                verified = []
            if not isinstance(unverified, list):
                unverified = []
            if not isinstance(inferred_only, list):
                inferred_only = []

            total_tracked = len(verified) + len(unverified) + len(inferred_only)
            if total_tracked > 0:
                # Reward for tracking attributes at all
                av_score = 0.6
                # Bonus for having verified items
                if len(verified) > 0:
                    av_score = min(1.0, 0.6 + 0.2 * len(verified) / total_tracked)
                # Penalty if match is "exact" but inferred_only is non-empty (inconsistency)
                if match_level == 'exact' and len(inferred_only) > 0:
                    av_score = max(0.0, av_score - 0.3)
            elif page_type == 'product':
                # On product page but no verification tracking
                av_score = 0.3

        components.append(av_score)

        score = 0.35 * components[0] + 0.25 * components[1] + 0.25 * components[2] + 0.15 * components[3]
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
        Calculate progress reward for WebShop.
        Measures: Is search_status accurate? Does evidence reference real observations?

        Components:
        - search_status accuracy (40%)
        - evidence quality (30%)
        - updated_subgoal logic (30%)
        """
        if not belief_progress:
            return 0.0

        # Component 1: Search status accuracy (40% weight)
        status_score = 0.0
        search_status = str(belief_progress.get('search_status', 'not_started')).lower()
        page_type = ground_truth.get('page_type', 'search')
        queries = ground_truth.get('search_queries', [])
        products_seen = ground_truth.get('products_seen', [])
        options_clicked = ground_truth.get('options_clicked', [])

        # Map ground truth to expected status
        if done:
            expected_statuses = ['ready_to_buy', 'done']
        elif options_clicked:
            expected_statuses = ['options_selecting', 'ready_to_buy']
        elif products_seen and page_type == 'product':
            expected_statuses = ['product_found', 'options_selecting']
        elif queries and page_type == 'results':
            expected_statuses = ['searching', 'product_found']
        elif not queries:
            expected_statuses = ['not_started', 'searching']
        else:
            expected_statuses = ['searching']

        if search_status in expected_statuses:
            status_score = 1.0
        elif any(s in search_status for s in expected_statuses):
            status_score = 0.7
        else:
            status_score = 0.3

        # Component 2: Evidence quality (30% weight)
        evidence_score = 0.0
        evidence = belief_progress.get('evidence', '')
        if not isinstance(evidence, str):
            evidence = str(evidence) if evidence else ''

        if evidence and len(evidence.strip()) > 10:
            evidence_score = 0.7
            # Bonus for referencing actual observations
            if ground_truth.get('current_product_title'):
                title_words = ground_truth['current_product_title'].lower().split()[:5]
                for word in title_words:
                    if len(word) > 3 and word in evidence.lower():
                        evidence_score = min(1.0, evidence_score + 0.1)
                        break
            if queries:
                for q in queries[-2:]:
                    if q.lower() in evidence.lower():
                        evidence_score = min(1.0, evidence_score + 0.1)
                        break
        elif evidence:
            evidence_score = 0.3
        else:
            evidence_score = 0.0

        # Component 3: Subgoal logic (30% weight)
        subgoal_score = 0.5
        subgoal = belief_progress.get('updated_subgoal', '')
        if not isinstance(subgoal, str):
            subgoal = str(subgoal) if subgoal else ''

        if subgoal and len(subgoal.strip()) > 5:
            subgoal_score = 0.7
            # Check if subgoal relates to task
            task_goal = ground_truth.get('task_goal', '') or ''
            if task_goal:
                task_keywords = set(w for w in task_goal.lower().split() if len(w) > 3)
                subgoal_keywords = set(w for w in subgoal.lower().split() if len(w) > 3)
                overlap = len(task_keywords & subgoal_keywords)
                if overlap > 0:
                    subgoal_score = min(1.0, 0.7 + overlap * 0.1)

        score = 0.4 * status_score + 0.3 * evidence_score + 0.3 * subgoal_score
        return score * self.progress_scale

    def calculate_exploration_reward(
        self,
        belief_exploration: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int
    ) -> float:
        """
        Calculate exploration reward for WebShop.
        Measures: Efficient searching? Not revisiting products? Diverse queries?

        Components:
        - queries_tried accuracy (40%)
        - products_viewed tracking (30%)
        - options_selected tracking (30%)
        """
        if not belief_exploration:
            return 0.0

        # Component 1: Queries tried accuracy (40% weight)
        queries_score = 0.5
        claimed_queries = belief_exploration.get('queries_tried', [])
        if not isinstance(claimed_queries, list):
            claimed_queries = [claimed_queries] if claimed_queries else []

        actual_queries = ground_truth.get('search_queries', [])

        if claimed_queries:
            # Check overlap with actual queries
            claimed_set = set(q.lower().strip() for q in claimed_queries if isinstance(q, str))
            actual_set = set(q.lower().strip() for q in actual_queries if isinstance(q, str))

            if actual_set:
                correct = len(claimed_set & actual_set)
                false_claims = len(claimed_set - actual_set)
                if claimed_set:
                    precision = correct / len(claimed_set)
                    queries_score = max(0.0, precision - false_claims * 0.15)
                else:
                    queries_score = 0.5
            else:
                # No actual queries yet
                if len(claimed_set) == 0:
                    queries_score = 0.8  # Correctly empty
                else:
                    queries_score = max(0.0, 0.5 - len(claimed_set) * 0.1)

            # Bonus for query diversity
            if len(claimed_set) > 1:
                queries_score = min(1.0, queries_score + 0.1)
        elif not actual_queries:
            queries_score = 0.6  # Both empty, reasonable

        # Component 2: Products viewed tracking (30% weight)
        products_score = 0.5
        claimed_products = belief_exploration.get('products_viewed', [])
        if not isinstance(claimed_products, list):
            claimed_products = [claimed_products] if claimed_products else []

        actual_products = ground_truth.get('products_seen', [])

        if claimed_products:
            claimed_set = set(str(p).lower().strip() for p in claimed_products if p)
            actual_set = set(str(p).lower().strip() for p in actual_products if p)

            if actual_set:
                correct = len(claimed_set & actual_set)
                if claimed_set:
                    products_score = correct / len(claimed_set)
                else:
                    products_score = 0.5
            else:
                products_score = max(0.0, 0.5 - len(claimed_set) * 0.1)
        elif not actual_products:
            products_score = 0.6

        # Component 3: Options selected tracking (30% weight)
        options_score = 0.5
        claimed_options = belief_exploration.get('options_selected', [])
        if not isinstance(claimed_options, list):
            claimed_options = [claimed_options] if claimed_options else []

        actual_options = ground_truth.get('options_clicked', [])

        if claimed_options:
            claimed_set = set(str(o).lower().strip() for o in claimed_options if o)
            actual_set = set(str(o).lower().strip() for o in actual_options if o)

            if actual_set:
                correct = len(claimed_set & actual_set)
                if claimed_set:
                    options_score = correct / len(claimed_set)
                else:
                    options_score = 0.5
            else:
                options_score = max(0.0, 0.5 - len(claimed_set) * 0.1)
        elif not actual_options:
            options_score = 0.6

        score = 0.4 * queries_score + 0.3 * products_score + 0.3 * options_score
        return score * self.exploration_scale

    def calculate_format_reward(self, is_format_valid: bool, is_action_available: bool) -> float:
        """Calculate format validity reward (same as ALFWorld)"""
        reward = 0.0
        if is_format_valid:
            reward += self.format_valid_reward
        else:
            reward += self.format_invalid_penalty
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
        Calculate total intrinsic reward from all four components.

        Args:
            belief_state: Parsed belief state from model
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
        product_understanding = belief_state.get('product_understanding', {})
        if not isinstance(product_understanding, dict):
            product_understanding = {}
        search_progress = belief_state.get('search_progress', {})
        if not isinstance(search_progress, dict):
            search_progress = {}
        exploration_state = belief_state.get('exploration_state', {})
        if not isinstance(exploration_state, dict):
            exploration_state = {}

        r_consistency = self.calculate_consistency_reward(product_understanding, ground_truth, step, belief_state=belief_state)
        r_progress = self.calculate_progress_reward(search_progress, ground_truth, step, done, success)
        r_exploration = self.calculate_exploration_reward(exploration_state, ground_truth, step)
        r_format = self.calculate_format_reward(is_format_valid, is_action_available)

        # Transition legality penalty: ready_to_buy with inferred_only items
        transition_penalty = 0.0
        av = belief_state.get('attribute_verification', {})
        if isinstance(av, dict):
            inferred_only = av.get('inferred_only', [])
            if isinstance(inferred_only, list) and len(inferred_only) > 0:
                search_status = search_progress.get('search_status', '')
                if str(search_status).lower() == 'ready_to_buy':
                    transition_penalty = -0.02 * len(inferred_only)

        belief_weight = self.get_belief_reward_weight()

        if not self.use_belief_reward and not self.belief_reward_decay_enable:
            belief_weight = 0.0

        component_weights = self.compute_component_weights(belief_weight)

        r_consistency_weighted = r_consistency * component_weights['consistency']
        r_progress_weighted = r_progress * component_weights['progress']
        r_exploration_weighted = r_exploration * component_weights['exploration']

        total_reward = (
            self.alpha * r_consistency_weighted +
            self.beta * r_progress_weighted +
            self.gamma * r_exploration_weighted +
            self.delta * r_format +
            transition_penalty
        )

        breakdown = {
            'r_consistency': r_consistency,
            'r_progress': r_progress,
            'r_exploration': r_exploration,
            'r_format': r_format,
            'r_intrinsic_total': total_reward,
            'belief_weight': belief_weight,
            'r_consistency_weighted': r_consistency_weighted,
            'r_progress_weighted': r_progress_weighted,
            'r_exploration_weighted': r_exploration_weighted,
            'transition_penalty': transition_penalty,
        }

        return total_reward, breakdown


# ============================================================================
# Factory functions (mirror ALFWorld pattern)
# ============================================================================

def create_webshop_rebel_tracker():
    """Create a WebShop ReBel belief state parser and ground truth tracker"""
    return WebShopBeliefStateParser(), WebShopGroundTruthTracker()


def create_webshop_reward_calculator(**kwargs):
    """Create a WebShop ReBel reward calculator with given parameters"""
    return WebShopRebelRewardCalculator(**kwargs)
