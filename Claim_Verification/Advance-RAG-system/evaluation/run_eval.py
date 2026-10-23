import json

from evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)

from evaluation.metrics.hit_rate import (
    HitRate
)

from evaluation.metrics.recall import (
    RecallAtK
)

from evaluation.metrics.mrr import (
    MRR
)


def run():

    with open(
        "evaluation/datasets/eval_datasets.json",
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    evaluator = (
        RetrievalEvaluator()
    )

    results = []

    # For testing first question only
    for sample in dataset:

        result = (
            evaluator.evaluate_question(
                sample["question"],
                sample["expected_chunk"],
                sample["expected_metadata"]
            )
        )

        results.append(
            result
        )

    hit_rate = (
        HitRate.calculate(
            results
        )
    )

    recall = (
        RecallAtK.calculate(
            results
        )
    )

    mrr = (
        MRR.calculate(
            results
        )
    )

    avg_latency = (

        sum(

            result["latency"]

            for result in results

        )

        /

        len(results)

    ) if results else 0

    report = {

        "hit_rate":
        hit_rate,

        "recall_at_5":
        recall,

        "mrr":
        mrr,

        "average_latency":
        avg_latency,

        "total_questions":
        len(results),

        "results":
        results
    }

    with open(
        "evaluation/reports/results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n====================")

    print(
        f"Questions: "
        f"{len(results)}"
    )

    print(
        f"Hit Rate: "
        f"{hit_rate:.2%}"
    )

    print(
        f"Recall@5: "
        f"{recall:.2%}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print(
        f"Average Latency: "
        f"{avg_latency:.2f}s"
    )

    print("====================\n")


if __name__ == "__main__":

    run()