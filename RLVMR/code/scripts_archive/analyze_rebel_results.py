#!/usr/bin/env python3
"""
ReBel Test Results Analyzer
Extracts, analyzes, and formats ReBel test results for research paper writing.
"""

import json
import yaml
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

class RebelResultsAnalyzer:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.results = {
            "metadata": {},
            "belief_grouping": {},
            "rewards": {},
            "performance": {},
            "belief_states": [],
            "errors": [],
            "paper_ready": {}
        }

    def load_logs(self):
        """Load all log files from result directory"""
        training_log = self.result_dir / "training.log"
        config_file = self.result_dir / "test_config.yaml"

        if config_file.exists():
            with open(config_file, 'r') as f:
                self.results["metadata"] = yaml.safe_load(f)

        if training_log.exists():
            with open(training_log, 'r') as f:
                self.log_content = f.read()
        else:
            self.log_content = ""

    def extract_belief_grouping_stats(self):
        """Extract belief grouping statistics"""
        # ReBel-specific grouping stats
        patterns = {
            "num_groups": r"Number of groups:\s*(\d+)",
            "mean_group_size": r"Mean group size:\s*([\d.]+)",
            "median_group_size": r"Median group size:\s*([\d.]+)",
            "min_group_size": r"Min group size:\s*(\d+)",
            "max_group_size": r"Max group size:\s*(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, self.log_content)
            if match:
                value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                self.results["belief_grouping"][key] = value

        # Extract group size distribution
        dist_pattern = r"Size\s+\|\s+Count\s+\|\s+Proportion\s*\n-+\n((?:\s*\d+\s+\|\s+\d+\s+\|\s+[\d.]+%\s*\n?)+)"
        dist_match = re.search(dist_pattern, self.log_content)
        if dist_match:
            distribution = []
            for line in dist_match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    size = int(parts[0])
                    count = int(parts[1])
                    prop = parts[2].replace('%', '')
                    distribution.append({
                        "size": size,
                        "count": count,
                        "proportion": float(prop) / 100
                    })
            self.results["belief_grouping"]["distribution"] = distribution

    def extract_reward_stats(self):
        """Extract reward statistics"""
        patterns = {
            "episode_reward_mean": r"episode_rewards_mean[:\s]+([\d.]+)",
            "episode_reward_min": r"episode_rewards_min[:\s]+([\d.]+)",
            "episode_reward_max": r"episode_rewards_max[:\s]+([\d.]+)",
            "intrinsic_reward_mean": r"rebel_stats/intrinsic_reward/mean[:\s]+([\d.]+)",
            "intrinsic_reward_std": r"rebel_stats/intrinsic_reward/std[:\s]+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, self.log_content)
            if match:
                self.results["rewards"][key] = float(match.group(1))

    def extract_performance_metrics(self):
        """Extract performance metrics"""
        patterns = {
            "success_rate": r"success_rate[:\s]+([\d.]+)",
            "avg_episode_length": r"episode_lengths_mean[:\s]+([\d.]+)",
            "min_episode_length": r"episode_lengths_min[:\s]+([\d.]+)",
            "max_episode_length": r"episode_lengths_max[:\s]+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, self.log_content)
            if match:
                self.results["performance"][key] = float(match.group(1))

        # Compute efficiency if both metrics available
        if "success_rate" in self.results["performance"] and "avg_episode_length" in self.results["performance"]:
            sr = self.results["performance"]["success_rate"]
            avg_len = self.results["performance"]["avg_episode_length"]
            if avg_len > 0:
                self.results["performance"]["efficiency"] = sr / avg_len

    def extract_belief_states_samples(self):
        """Extract sample belief states from logs"""
        # Look for belief state outputs
        belief_pattern = r'\*\*Belief State Update:\*\*\s*```json\s*(\{[^`]+\})\s*```'
        matches = re.finditer(belief_pattern, self.log_content, re.DOTALL)

        samples = []
        for i, match in enumerate(matches):
            if i >= 5:  # Limit to 5 samples
                break
            try:
                belief_json = json.loads(match.group(1))
                samples.append(belief_json)
            except json.JSONDecodeError:
                pass

        self.results["belief_states"] = samples

    def check_errors(self):
        """Check for errors and warnings"""
        error_patterns = [
            r"ERROR:.*",
            r"Traceback.*",
            r"Exception.*",
            r"Failed.*",
        ]

        errors = []
        for pattern in error_patterns:
            matches = re.finditer(pattern, self.log_content, re.MULTILINE)
            for match in matches:
                errors.append(match.group(0))

        self.results["errors"] = errors[:20]  # Limit to 20 errors

    def generate_paper_ready_output(self):
        """Generate paper-ready formatted results"""
        paper = {}

        # Table 1: Belief Grouping Statistics
        if self.results["belief_grouping"]:
            bg = self.results["belief_grouping"]
            paper["table_belief_grouping"] = {
                "caption": "Belief-based grouping statistics for ReBel",
                "data": {
                    "Number of Groups": bg.get("num_groups", "N/A"),
                    "Mean Group Size": f"{bg.get('mean_group_size', 0):.2f}",
                    "Median Group Size": f"{bg.get('median_group_size', 0):.1f}",
                    "Group Size Range": f"[{bg.get('min_group_size', 0)}, {bg.get('max_group_size', 0)}]",
                }
            }

        # Table 2: Performance Metrics
        if self.results["performance"]:
            perf = self.results["performance"]
            paper["table_performance"] = {
                "caption": "ReBel performance on ALFWorld tasks",
                "data": {
                    "Success Rate (%)": f"{perf.get('success_rate', 0) * 100:.1f}",
                    "Avg Episode Length": f"{perf.get('avg_episode_length', 0):.1f}",
                    "Efficiency": f"{perf.get('efficiency', 0):.3f}",
                }
            }

        # Table 3: Reward Analysis
        if self.results["rewards"]:
            rew = self.results["rewards"]
            paper["table_rewards"] = {
                "caption": "Reward statistics for ReBel training",
                "data": {
                    "Episode Reward (mean)": f"{rew.get('episode_reward_mean', 0):.3f}",
                    "Intrinsic Reward (mean)": f"{rew.get('intrinsic_reward_mean', 0):.3f}",
                    "Intrinsic Reward (std)": f"{rew.get('intrinsic_reward_std', 0):.3f}",
                }
            }

        # LaTeX table format
        paper["latex_belief_grouping"] = self._format_latex_table(
            paper.get("table_belief_grouping", {})
        )
        paper["latex_performance"] = self._format_latex_table(
            paper.get("table_performance", {})
        )

        self.results["paper_ready"] = paper

    def _format_latex_table(self, table_data):
        """Format data as LaTeX table"""
        if not table_data or "data" not in table_data:
            return ""

        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{table_data.get('caption', 'Results')}}}\n"
        latex += "\\begin{tabular}{|l|r|}\n"
        latex += "\\hline\n"
        latex += "Metric & Value \\\\\n"
        latex += "\\hline\n"

        for key, value in table_data["data"].items():
            latex += f"{key} & {value} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def save_results(self):
        """Save all results to files"""
        # JSON format
        with open(self.result_dir / "analysis_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        # YAML format
        with open(self.result_dir / "analysis_results.yaml", 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)

        # Paper-ready markdown
        self._save_markdown_report()

        # LaTeX tables
        self._save_latex_tables()

    def _save_markdown_report(self):
        """Save markdown report for easy reading"""
        md = f"# ReBel Test Results Analysis\n\n"
        md += f"**Test Date:** {self.results['metadata'].get('timestamp', 'Unknown')}\n\n"

        md += "## 1. Belief Grouping Statistics\n\n"
        if self.results["belief_grouping"]:
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            for key, value in self.results["belief_grouping"].items():
                if key != "distribution":
                    md += f"| {key.replace('_', ' ').title()} | {value} |\n"

            if "distribution" in self.results["belief_grouping"]:
                md += "\n### Group Size Distribution\n\n"
                md += "| Group Size | Count | Proportion |\n"
                md += "|------------|-------|------------|\n"
                for item in self.results["belief_grouping"]["distribution"]:
                    md += f"| {item['size']} | {item['count']} | {item['proportion']*100:.1f}% |\n"

        md += "\n## 2. Performance Metrics\n\n"
        if self.results["performance"]:
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            for key, value in self.results["performance"].items():
                display_key = key.replace('_', ' ').title()
                if 'rate' in key:
                    display_value = f"{value * 100:.1f}%"
                else:
                    display_value = f"{value:.3f}"
                md += f"| {display_key} | {display_value} |\n"

        md += "\n## 3. Reward Statistics\n\n"
        if self.results["rewards"]:
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            for key, value in self.results["rewards"].items():
                md += f"| {key.replace('_', ' ').title()} | {value:.3f} |\n"

        md += "\n## 4. Sample Belief States\n\n"
        if self.results["belief_states"]:
            for i, belief in enumerate(self.results["belief_states"][:3], 1):
                md += f"### Sample {i}\n\n"
                md += "```json\n"
                md += json.dumps(belief, indent=2)
                md += "\n```\n\n"

        if self.results["errors"]:
            md += "\n## 5. Errors and Warnings\n\n"
            md += "```\n"
            for error in self.results["errors"][:10]:
                md += f"{error}\n"
            md += "```\n"

        with open(self.result_dir / "ANALYSIS_REPORT.md", 'w') as f:
            f.write(md)

    def _save_latex_tables(self):
        """Save LaTeX tables to separate file"""
        if "paper_ready" in self.results:
            latex_content = "% ReBel Test Results - LaTeX Tables\n"
            latex_content += "% Auto-generated from test results\n\n"

            for key, value in self.results["paper_ready"].items():
                if key.startswith("latex_"):
                    table_name = key.replace("latex_", "")
                    latex_content += f"% Table: {table_name}\n"
                    latex_content += value
                    latex_content += "\n\n"

            with open(self.result_dir / "paper_tables.tex", 'w') as f:
                f.write(latex_content)

    def print_summary(self):
        """Print summary to console"""
        print("\n" + "="*70)
        print("REBEL TEST RESULTS SUMMARY")
        print("="*70)

        if self.results["metadata"]:
            print(f"\nTest: {self.results['metadata'].get('test_name', 'Unknown')}")
            print(f"Time: {self.results['metadata'].get('timestamp', 'Unknown')}")

        print("\n--- Belief Grouping ---")
        if self.results["belief_grouping"]:
            bg = self.results["belief_grouping"]
            print(f"Number of Groups: {bg.get('num_groups', 'N/A')}")
            print(f"Mean Group Size: {bg.get('mean_group_size', 0):.2f}")
            print(f"Group Size Range: [{bg.get('min_group_size', 0)}, {bg.get('max_group_size', 0)}]")
        else:
            print("No belief grouping data found")

        print("\n--- Performance ---")
        if self.results["performance"]:
            perf = self.results["performance"]
            print(f"Success Rate: {perf.get('success_rate', 0) * 100:.1f}%")
            print(f"Avg Episode Length: {perf.get('avg_episode_length', 0):.1f}")
            print(f"Efficiency: {perf.get('efficiency', 0):.3f}")
        else:
            print("No performance data found")

        print("\n--- Rewards ---")
        if self.results["rewards"]:
            rew = self.results["rewards"]
            print(f"Episode Reward: {rew.get('episode_reward_mean', 0):.3f}")
            print(f"Intrinsic Reward: {rew.get('intrinsic_reward_mean', 0):.3f} ± {rew.get('intrinsic_reward_std', 0):.3f}")
        else:
            print("No reward data found")

        if self.results["errors"]:
            print(f"\n⚠️  Found {len(self.results['errors'])} errors/warnings")

        print("\n--- Output Files ---")
        print(f"Results Directory: {self.result_dir}")
        print(f"  • analysis_results.json  : Structured results (JSON)")
        print(f"  • analysis_results.yaml  : Structured results (YAML)")
        print(f"  • ANALYSIS_REPORT.md     : Human-readable markdown report")
        print(f"  • paper_tables.tex       : LaTeX tables for paper")

        print("="*70 + "\n")

    def analyze(self):
        """Run complete analysis pipeline"""
        print("Loading logs...")
        self.load_logs()

        print("Extracting belief grouping statistics...")
        self.extract_belief_grouping_stats()

        print("Extracting reward statistics...")
        self.extract_reward_stats()

        print("Extracting performance metrics...")
        self.extract_performance_metrics()

        print("Extracting sample belief states...")
        self.extract_belief_states_samples()

        print("Checking for errors...")
        self.check_errors()

        print("Generating paper-ready output...")
        self.generate_paper_ready_output()

        print("Saving results...")
        self.save_results()

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Analyze ReBel test results")
    parser.add_argument("result_dir", help="Path to test results directory")

    args = parser.parse_args()

    analyzer = RebelResultsAnalyzer(args.result_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()
