"""LightEval integration for SteadyText accuracy benchmarking.

This module provides a custom model backend for LightEval to evaluate
SteadyText's performance on standard NLP benchmarks.
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    from lighteval.models.abstract_model import LightevalModel  # type: ignore[import-not-found]
    from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn  # type: ignore[import-not-found]
    from lighteval.tasks.requests import (  # type: ignore[import-not-found]
        GreedyUntilRequest,
        LoglikelihoodRequest,
        LoglikelihoodRollingRequest,
        LoglikelihoodSingleTokenRequest,
    )

    LIGHTEVAL_AVAILABLE = True
except ImportError:
    LIGHTEVAL_AVAILABLE = False

    # Fallback for type hints
    class LightevalModel:
        def __init__(self, *args, **kwargs):
            pass

    class GreedyUntilRequest:
        def __init__(self, context=None, stop_sequence=None, **kwargs):
            self.context = context
            self.stop_sequence = stop_sequence

    class LoglikelihoodRequest:
        def __init__(self, context=None, continuation=None, **kwargs):
            self.context = context
            self.continuation = continuation

    class LoglikelihoodRollingRequest:
        def __init__(self, context=None, **kwargs):
            self.context = context

    class LoglikelihoodSingleTokenRequest:
        def __init__(self, context=None, **kwargs):
            self.context = context

    class GenerateReturn:
        def __init__(self, result=None, logits=None, generated_tokens=None, **kwargs):
            self.result = result
            self.logits = logits
            self.generated_tokens = generated_tokens

    class LoglikelihoodReturn:
        def __init__(self, result=None, logits=None, generated_tokens=None, **kwargs):
            self.result = result
            self.logits = logits
            self.generated_tokens = generated_tokens


import steadytext


@dataclass
class SteadyTextConfig:
    """Configuration for SteadyText in LightEval."""

    model_name: str = "steadytext"
    deterministic: bool = True
    verify_determinism: bool = True
    max_length: int = 512


class SteadyTextLightEvalModel(LightevalModel):
    """SteadyText model adapter for LightEval.

    This class wraps SteadyText to work with LightEval's evaluation framework.
    """

    def __init__(self, config: Optional[SteadyTextConfig] = None):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError(
                "LightEval is not installed. Install with: pip install lighteval"
            )

        self.config = config or SteadyTextConfig()
        self._name = self.config.model_name

        # Preload models
        steadytext.preload_models(verbose=True)

        # Track determinism verification
        self.determinism_checks = []

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information for logging."""
        return {
            "model_name": self.model_name,
            "model_type": "steadytext",
            "deterministic": self.config.deterministic,
            "max_length": self.config.max_length,
            "version": steadytext.__version__,
        }

    def greedy_until(
        self, requests: List[GreedyUntilRequest], override_bs: Optional[int] = None
    ) -> List[GenerateReturn]:
        """Generate text greedily until stop sequences are met.

        This is the main generation method used by most LightEval tasks.
        """
        results = []

        for request in requests:
            # Extract prompt and stop sequences
            prompt = request.context
            stop_sequences = request.stop_sequence or []

            # Ensure prompt is a string
            if not isinstance(prompt, str):
                prompt = str(prompt) if prompt is not None else ""

            # Generate text
            start_time = time.time()
            generated_text = steadytext.generate(prompt)
            generation_time = time.time() - start_time

            # Verify determinism if requested
            if self.config.verify_determinism:
                second_generation = steadytext.generate(prompt)
                is_deterministic = generated_text == second_generation
                self.determinism_checks.append(
                    {
                        "prompt": prompt[:50] + "...",
                        "deterministic": is_deterministic,
                        "generation_time": generation_time,
                    }
                )

            # Truncate at stop sequences
            truncated_text = generated_text if generated_text is not None else ""
            for stop_seq in stop_sequences:
                if truncated_text and stop_seq in truncated_text:
                    truncated_text = truncated_text.split(stop_seq)[0]

            # Create result
            if LIGHTEVAL_AVAILABLE:
                result = GenerateReturn(
                    result=truncated_text,
                    logits=None,  # SteadyText doesn't expose logits in the same way
                    generated_tokens=[],  # Would need tokenizer integration
                )
            else:
                result = GenerateReturn(
                    result=truncated_text, logits=None, generated_tokens=[]
                )
            results.append(result)

        return results

    def loglikelihood(
        self,
        requests: List[Union[LoglikelihoodRequest, LoglikelihoodRollingRequest]],
        override_bs: Optional[int] = None,
    ) -> List[LoglikelihoodReturn]:
        """Compute log-likelihood of continuations.

        This is used for multiple-choice and classification tasks.
        """
        results = []

        for request in requests:
            if isinstance(request, LoglikelihoodRollingRequest):
                # For rolling requests, we need the full context
                context = str(request.context) if request.context is not None else ""
                continuation = ""
            else:
                # For regular requests, concatenate context and continuation
                context = str(request.context) if request.context is not None else ""
                continuation = str(getattr(request, "continuation", ""))
                context + continuation

            # Generate with logprobs to get likelihood information
            result = steadytext.generate(prompt=context, return_logprobs=True)

            # Handle case where model is not loaded
            if result is None or (isinstance(result, tuple) and result[0] is None):
                generated_text = ""
                logprobs_dict = None
            else:
                generated_text, logprobs_dict = result

            # Since SteadyText is deterministic and doesn't provide
            # true log-likelihoods, we'll use a proxy based on whether
            # the model would generate the continuation
            if isinstance(request, LoglikelihoodRollingRequest):
                # For rolling, return a dummy likelihood
                loglikelihood = 0.0
            else:
                # Check if generated text starts with the continuation
                # This is a simplified approach for deterministic models
                if generated_text.startswith(continuation):
                    loglikelihood = 0.0  # High likelihood (log scale)
                else:
                    loglikelihood = -100.0  # Low likelihood

            if LIGHTEVAL_AVAILABLE:
                result = LoglikelihoodReturn(
                    result=(loglikelihood, False),  # (loglikelihood, is_greedy)
                    logits=None,
                    generated_tokens=[],
                )
            else:
                result = LoglikelihoodReturn(
                    result=(loglikelihood, False), logits=None, generated_tokens=[]
                )
            results.append(result)

        return results

    def loglikelihood_single_token(
        self,
        requests: List[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> List[LoglikelihoodReturn]:
        """Compute log-likelihood for single token predictions.

        Used for tasks that require next-token prediction.
        """
        # For SteadyText, we'll treat this similarly to regular loglikelihood
        # but focusing on just the first token
        return self.loglikelihood(requests, override_bs)

    def get_model_context_length(self) -> int:
        """Return the model's context length."""
        # SteadyText uses GENERATION_MAX_NEW_TOKENS for output length
        # Context length would depend on the underlying model
        return 2048  # Conservative estimate

    def cleanup(self):
        """Cleanup model resources if needed."""
        # SteadyText handles its own cleanup
        pass

    def get_determinism_report(self) -> Dict[str, Any]:
        """Get a report on determinism verification."""
        if not self.determinism_checks:
            return {"checks_performed": 0}

        total_checks = len(self.determinism_checks)
        deterministic_count = sum(
            1 for check in self.determinism_checks if check["deterministic"]
        )

        return {
            "checks_performed": total_checks,
            "deterministic_count": deterministic_count,
            "determinism_rate": (
                deterministic_count / total_checks if total_checks > 0 else 0
            ),
            "average_generation_time": np.mean(
                [check["generation_time"] for check in self.determinism_checks]
            ),
        }


class SteadyTextEvaluator:
    """High-level evaluator for running SteadyText through LightEval benchmarks."""

    def __init__(self, config: Optional[SteadyTextConfig] = None):
        if not LIGHTEVAL_AVAILABLE:
            raise ImportError(
                "LightEval is not installed. Install with: pip install lighteval"
            )

        self.config = config or SteadyTextConfig()
        self.model = SteadyTextLightEvalModel(config)

    def evaluate(
        self,
        tasks: List[str],
        num_shots: int = 0,
        batch_size: int = 1,
        output_dir: Optional[str] = None,
        save_details: bool = True,
    ) -> Dict[str, Any]:
        """Run evaluation on specified tasks.

        Args:
            tasks: List of task names (e.g., ["truthfulqa:mc", "gsm8k"])
            num_shots: Number of few-shot examples
            batch_size: Batch size for evaluation
            output_dir: Directory to save results
            save_details: Whether to save detailed results

        Returns:
            Dictionary with evaluation results
        """
        try:
            from lighteval.evaluator import evaluate  # type: ignore[import-not-found]
            from lighteval.tasks.registry import Registry  # type: ignore[import-not-found]

            # Initialize task registry
            Registry()
        except ImportError:
            raise ImportError("LightEval evaluator module is not available")

        # Configure evaluation
        eval_config = {
            "model": self.model,
            "tasks": tasks,
            "num_fewshot": num_shots,
            "batch_size": batch_size,
            "output_path": output_dir,
            "save_details": save_details,
        }

        # Run evaluation
        results = evaluate(**eval_config)

        # Add determinism report
        results["determinism_report"] = self.model.get_determinism_report()

        return results

    def evaluate_standard_benchmarks(
        self, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run evaluation on a standard set of benchmarks.

        This includes common benchmarks like TruthfulQA, GSM8K, etc.
        """
        standard_tasks = [
            "leaderboard|truthfulqa:mc|0|0",  # TruthfulQA multiple choice
            "leaderboard|gsm8k|0|0",  # Grade school math
            "leaderboard|hellaswag|0|0",  # Common sense reasoning
            "leaderboard|arc:easy|0|0",  # AI2 Reasoning Challenge (Easy)
        ]

        return self.evaluate(
            tasks=standard_tasks,
            num_shots=0,  # Zero-shot evaluation
            output_dir=output_dir,
        )


def create_steadytext_model(
    model_name: str = "steadytext", **kwargs
) -> SteadyTextLightEvalModel:
    """Factory function to create a SteadyText model for LightEval.

    This function can be registered with LightEval's model registry.
    """
    config = SteadyTextConfig(model_name=model_name, **kwargs)
    return SteadyTextLightEvalModel(config)
