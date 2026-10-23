class MRR:

    @staticmethod
    def calculate(results):

        total = len(results)

        score = sum(
            result["mrr"]
            for result in results
        )

        return score / total if total else 0