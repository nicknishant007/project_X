class HitRate:

    @staticmethod
    def calculate(results):

        total = len(results)

        hits = sum(
            result["hit"]
            for result in results
        )

        return hits / total if total else 0