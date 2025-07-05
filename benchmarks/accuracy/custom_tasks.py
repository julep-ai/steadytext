"""Custom LightEval tasks for evaluating SteadyText-specific properties.

These tasks test determinism, consistency, and other unique aspects of SteadyText.
"""

from typing import Dict, List, Any, Optional
import json
import numpy as np

try:
    from lighteval.tasks.task import LightevalTask  # type: ignore[import-not-found]
    from lighteval.tasks.requests import GreedyUntilRequest  # type: ignore[import-not-found]

    LIGHTEVAL_AVAILABLE = True
except ImportError:
    LIGHTEVAL_AVAILABLE = False

    # Create proper base classes for type compatibility
    class LightevalTask:
        def __init__(
            self,
            name=None,
            prompt_function=None,
            suite=None,
            metric=None,
            few_shots_split=None,
            few_shots_select=None,
            generation_size=None,
            stop_sequence=None,
            output_regex=None,
            **kwargs,
        ):
            pass

    class GreedyUntilRequest:
        def __init__(
            self, context=None, stop_sequence=None, generation_size=None, **kwargs
        ):
            self.context = context
            self.stop_sequence = stop_sequence
            self.generation_size = generation_size


import steadytext


class DeterminismTask(LightevalTask):
    """Task to verify deterministic behavior of SteadyText."""

    def __init__(self):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError("LightEval is required for custom tasks")

        super().__init__(
            name="steadytext_determinism",
            prompt_function=self.create_determinism_prompt,
            suite=["steadytext"],
            metric=[self.determinism_metric],
            few_shots_split="train",
            few_shots_select="random",
            generation_size=512,
            stop_sequence=[],
            output_regex=None,
        )

        # Test cases for determinism
        self.test_prompts = [
            "Write a Python function to sort a list",
            "Explain the concept of recursion",
            "What is machine learning?",
            "Describe the water cycle",
            "How do neural networks work?",
            "What are the benefits of exercise?",
            "Explain quantum computing",
            "What is climate change?",
            "How does the internet work?",
            "What is artificial intelligence?",
        ]

    def create_determinism_prompt(self, sample: Dict[str, Any]) -> GreedyUntilRequest:
        """Create a prompt for determinism testing."""
        prompt = sample.get(
            "prompt", self.test_prompts[sample.get("idx", 0) % len(self.test_prompts)]
        )

        return GreedyUntilRequest(
            context=prompt,
            stop_sequence=[],
            generation_size=512,
            task_name=self.name,
            sample_index=sample.get("idx", 0),
            request_index=0,
            metric_categories=["determinism"],
        )

    @staticmethod
    def determinism_metric(
        predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Metric that checks if multiple generations are identical."""
        determinism_scores = []

        for i, prompt in enumerate(predictions):
            # Generate multiple times and check consistency
            generations = []
            for _ in range(3):  # Generate 3 times
                output = steadytext.generate(prompt)
                generations.append(output)

            # Check if all generations are identical
            is_deterministic = all(g == generations[0] for g in generations)
            determinism_scores.append(1.0 if is_deterministic else 0.0)

        return {
            "determinism_rate": (
                sum(determinism_scores) / len(determinism_scores)
                if determinism_scores
                else 0.0
            ),
            "fully_deterministic": (
                1.0 if all(s == 1.0 for s in determinism_scores) else 0.0
            ),
        }

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Return dataset for determinism testing."""
        return [
            {"idx": i, "prompt": prompt} for i, prompt in enumerate(self.test_prompts)
        ]


class ConsistencyTask(LightevalTask):
    """Task to test consistency of outputs across similar prompts."""

    def __init__(self):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError("LightEval is required for custom tasks")

        super().__init__(
            name="steadytext_consistency",
            prompt_function=self.create_consistency_prompt,
            suite=["steadytext"],
            metric=[self.consistency_metric],
            few_shots_split="train",
            few_shots_select="random",
            generation_size=512,
            stop_sequence=[],
            output_regex=None,
        )

        # Prompt variations for consistency testing
        self.prompt_groups = [
            {
                "base": "Write a function to calculate factorial",
                "variations": [
                    "Write a function to calculate factorial",
                    "Write a function to compute factorial",
                    "Create a function to calculate factorial",
                    "Implement a function to calculate factorial",
                ],
            },
            {
                "base": "Explain machine learning",
                "variations": [
                    "Explain machine learning",
                    "What is machine learning?",
                    "Describe machine learning",
                    "Define machine learning",
                ],
            },
            {
                "base": "Sort a list in Python",
                "variations": [
                    "Sort a list in Python",
                    "How to sort a list in Python",
                    "Python list sorting",
                    "Sort Python list",
                ],
            },
        ]

    def create_consistency_prompt(
        self, sample: Dict[str, Any]
    ) -> List[GreedyUntilRequest]:
        """Create prompts for consistency testing."""
        group_idx = sample.get("group_idx", 0) % len(self.prompt_groups)
        group = self.prompt_groups[group_idx]

        return [
            GreedyUntilRequest(
                context=variation,
                stop_sequence=[],
                generation_size=512,
                task_name=self.name,
                sample_index=sample.get("group_idx", 0),
                request_index=i,
                metric_categories=["consistency"],
            )
            for i, variation in enumerate(group["variations"])
        ]

    @staticmethod
    def consistency_metric(
        predictions: List[List[str]], references: List[str]
    ) -> Dict[str, float]:
        """Metric that measures consistency across prompt variations."""
        consistency_scores = []

        for group_predictions in predictions:
            # Calculate similarity between outputs for variations
            # For now, we'll use a simple approach: check if key terms appear consistently
            outputs = [steadytext.generate(prompt) for prompt in group_predictions]

            # Extract key terms (simplified - in practice, use better NLP)
            key_terms_sets = []
            for output in outputs:
                words = set(output.lower().split())
                # Filter common words
                key_terms = {w for w in words if len(w) > 4}
                key_terms_sets.append(key_terms)

            # Calculate Jaccard similarity between all pairs
            similarities = []
            for i in range(len(key_terms_sets)):
                for j in range(i + 1, len(key_terms_sets)):
                    set_i, set_j = key_terms_sets[i], key_terms_sets[j]
                    if set_i or set_j:
                        similarity = len(set_i & set_j) / len(set_i | set_j)
                        similarities.append(similarity)

            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0.0
            )
            consistency_scores.append(avg_similarity)

        return {
            "consistency_score": (
                sum(consistency_scores) / len(consistency_scores)
                if consistency_scores
                else 0.0
            ),
            "high_consistency_rate": (
                sum(1 for s in consistency_scores if s > 0.7) / len(consistency_scores)
                if consistency_scores
                else 0.0
            ),
        }

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Return dataset for consistency testing."""
        return [{"group_idx": i} for i in range(len(self.prompt_groups))]


class FallbackBehaviorTask(LightevalTask):
    """Task to test fallback behavior when models are unavailable."""

    def __init__(self):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError("LightEval is required for custom tasks")

        super().__init__(
            name="steadytext_fallback",
            prompt_function=self.create_fallback_prompt,
            suite=["steadytext"],
            metric=[self.fallback_metric],
            few_shots_split="train",
            few_shots_select="random",
            generation_size=512,
            stop_sequence=[],
            output_regex=None,
        )

        self.test_cases = [
            {"prompt": "Test fallback generation", "type": "generation"},
            {"prompt": "Test fallback embedding", "type": "embedding"},
            {"prompt": "", "type": "empty"},
            {"prompt": 123, "type": "invalid"},  # Invalid type
        ]

    def create_fallback_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create test case for fallback testing."""
        return sample

    @staticmethod
    def fallback_metric(
        predictions: List[Dict[str, Any]], references: List[str]
    ) -> Dict[str, float]:
        """Metric that verifies fallback behavior."""
        fallback_scores = {
            "generation_fallback_works": 0,
            "embedding_fallback_works": 0,
            "empty_handling_works": 0,
            "invalid_type_handling_works": 0,
        }

        for test_case in predictions:
            test_type = test_case["type"]
            prompt = test_case["prompt"]

            try:
                if test_type == "generation":
                    output = steadytext.generate(prompt)
                    # Check if it returns a string (even if it's an error message)
                    if isinstance(output, str) and len(output) > 0:
                        fallback_scores["generation_fallback_works"] = 1.0

                elif test_type == "embedding":
                    output = steadytext.embed(prompt)
                    # Check if it returns a numpy array of correct shape
                    if isinstance(output, np.ndarray) and output.shape == (
                        steadytext.EMBEDDING_DIMENSION,
                    ):
                        fallback_scores["embedding_fallback_works"] = 1.0

                elif test_type == "empty":
                    gen_output = steadytext.generate(prompt)
                    emb_output = steadytext.embed(prompt)
                    if isinstance(gen_output, str) and isinstance(
                        emb_output, np.ndarray
                    ):
                        fallback_scores["empty_handling_works"] = 1.0

                elif test_type == "invalid":
                    # These should handle gracefully without crashing
                    try:
                        gen_output = steadytext.generate(prompt)
                        emb_output = steadytext.embed(prompt)
                        fallback_scores["invalid_type_handling_works"] = 1.0
                    except Exception:
                        # If it raises an exception, that's actually not following "never fails"
                        fallback_scores["invalid_type_handling_works"] = 0.0

            except Exception as e:
                # Any unhandled exception means the fallback didn't work properly
                print(f"Fallback test failed for {test_type}: {e}")

        # Overall fallback score
        fallback_scores["overall_fallback_score"] = sum(fallback_scores.values()) / len(
            fallback_scores
        )

        return fallback_scores

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Return dataset for fallback testing."""
        return self.test_cases


class PerformanceRegressionTask(LightevalTask):
    """Task to detect performance regressions."""

    def __init__(self, baseline_file: Optional[str] = None):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError("LightEval is required for custom tasks")

        super().__init__(
            name="steadytext_performance_regression",
            prompt_function=self.create_performance_prompt,
            suite=["steadytext"],
            metric=[self.regression_metric],
            few_shots_split="train",
            few_shots_select="random",
            generation_size=512,
            stop_sequence=[],
            output_regex=None,
        )

        self.baseline_file = baseline_file
        self.baseline_data = self.load_baseline() if baseline_file else None

        # Standard prompts for performance testing
        self.performance_prompts = [
            "Write a quicksort implementation",
            "Explain the theory of relativity",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis",
            "How does a computer processor work?",
        ]

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance data."""
        try:
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def create_performance_prompt(self, sample: Dict[str, Any]) -> GreedyUntilRequest:
        """Create prompt for performance testing."""
        prompt = self.performance_prompts[
            sample.get("idx", 0) % len(self.performance_prompts)
        ]

        return GreedyUntilRequest(
            context=prompt,
            stop_sequence=[],
            generation_size=512,
            task_name=self.name,
            sample_index=sample.get("idx", 0),
            request_index=0,
            metric_categories=["performance"],
        )

    def regression_metric(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Metric that checks for performance regressions."""
        import time

        current_timings = []

        for prompt in predictions:
            start_time = time.time()
            _ = steadytext.generate(prompt)
            elapsed = time.time() - start_time
            current_timings.append(elapsed)

        avg_time = (
            sum(current_timings) / len(current_timings) if current_timings else 0.0
        )

        results = {
            "avg_generation_time": avg_time,
            "max_generation_time": max(current_timings) if current_timings else 0.0,
            "min_generation_time": min(current_timings) if current_timings else 0.0,
        }

        # Compare with baseline if available
        if self.baseline_data:
            baseline_avg = self.baseline_data.get("avg_generation_time", avg_time)
            regression_threshold = 1.2  # 20% slower is considered regression

            results["regression_detected"] = (
                1.0 if avg_time > baseline_avg * regression_threshold else 0.0
            )
            results["performance_ratio"] = (
                avg_time / baseline_avg if baseline_avg > 0 else 1.0
            )

        return results

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Return dataset for performance testing."""
        return [{"idx": i} for i in range(len(self.performance_prompts))]


def register_steadytext_tasks():
    """Register all SteadyText custom tasks with LightEval."""
    if not LIGHTEVAL_AVAILABLE:
        raise ImportError("LightEval is required to register tasks")

    try:
        from lighteval.tasks.registry import Registry  # type: ignore[unresolved-import]

        registry = Registry()

        # Register each custom task
        tasks = [
            DeterminismTask(),
            ConsistencyTask(),
            FallbackBehaviorTask(),
            PerformanceRegressionTask(),
        ]

        for task in tasks:
            register_func = getattr(registry, "register_task", None)
            add_func = getattr(registry, "add_task", None)

            if register_func is not None and callable(register_func):
                register_func(task)
            elif add_func is not None and callable(add_func):
                add_func(task)
            else:
                # Fallback - just store in a dict-like structure
                setattr(registry, task.name, task)
    except (ImportError, AttributeError):
        # Create a simple registry fallback
        class SimpleRegistry:
            def __init__(self):
                self.tasks = {}

            def register_task(self, task):
                self.tasks[task.name] = task

        registry = SimpleRegistry()
        for task in [
            DeterminismTask(),
            ConsistencyTask(),
            FallbackBehaviorTask(),
            PerformanceRegressionTask(),
        ]:
            registry.register_task(task)

    return registry
