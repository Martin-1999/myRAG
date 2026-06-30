from __future__ import annotations

class RagasEvaluator:
    def evaluate(self, samples: list[dict]) -> dict[str, float]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )
        return {key: float(value) for key, value in result.items()}
