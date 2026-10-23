class RecallAtK:

    @staticmethod
    def calculate(results):

        total = len(results)

        recalls = sum(
            result["recall"]
            for result in results
        )

        return recalls / total if total else 0