import numpy as np

import yaml

from sklearn.metrics.pairwise import (
    cosine_similarity
)

with open(
    "configs/ingestion.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


class MMRRetriever:

    @staticmethod
    def rerank(
        query_embedding,
        candidate_embeddings,
        candidate_chunks,
        top_k=config["retrieval"]["mmr"]["top_k"],
        lambda_param=config["retrieval"]["mmr"]["lambda"],
        threshold_ratio=config["retrieval"]["mmr"]["threshold_ratio"],
        minimum_threshold=config["retrieval"]["mmr"]["minimum_threshold"]
    ):

        if not candidate_chunks:
            return []

        if len(candidate_chunks) != len(candidate_embeddings):

            raise ValueError(
                f"MMR Error: "
                f"{len(candidate_chunks)} chunks but "
                f"{len(candidate_embeddings)} embeddings"
            )

        # ----------------------------------------------------
        # Query Similarity
        # ----------------------------------------------------

        similarity_to_query = cosine_similarity(
            [query_embedding],
            candidate_embeddings
        )[0]

        # ----------------------------------------------------
        # Dynamic Threshold
        # ----------------------------------------------------

        best_score = float(np.max(similarity_to_query))

        dynamic_threshold = max(
            minimum_threshold,
            best_score * threshold_ratio
        )

        print("\n" + "=" * 80)
        print("MMR RETRIEVAL DEBUG")
        print("=" * 80)
        print(f"Best Similarity      : {best_score:.4f}")
        print(f"Threshold Ratio      : {threshold_ratio}")
        print(f"Minimum Threshold    : {minimum_threshold}")
        print(f"Dynamic Threshold    : {dynamic_threshold:.4f}")
        print()

        valid_indices = []

        for idx, score in enumerate(similarity_to_query):

            keep = score >= dynamic_threshold

            print(
                f"Chunk {idx + 1:02d} | "
                f"Similarity = {score:.4f} | "
                f"{'KEEP' if keep else 'REJECT'}"
            )

            if keep:
                valid_indices.append(idx)

        print()

        # ----------------------------------------------------
        # Nothing Passed
        # ----------------------------------------------------

        if not valid_indices:

            best = np.argmax(similarity_to_query)

            print(
                "⚠ No chunk passed threshold."
            )

            print(
                f"Returning best chunk ({best + 1}) "
                f"with similarity {best_score:.4f}"
            )

            return [candidate_chunks[best]]

        # ----------------------------------------------------
        # First Selection
        # ----------------------------------------------------

        selected_indices = []

        first_index = max(
            valid_indices,
            key=lambda idx: similarity_to_query[idx]
        )

        selected_indices.append(first_index)

        print(
            f"\nSelected First Chunk : "
            f"{first_index + 1}"
        )

        # ----------------------------------------------------
        # MMR Selection
        # ----------------------------------------------------

        while len(selected_indices) < min(
            top_k,
            len(valid_indices)
        ):

            remaining_indices = [

                idx

                for idx in valid_indices

                if idx not in selected_indices

            ]

            if not remaining_indices:
                break

            mmr_scores = []

            for idx in remaining_indices:

                relevance = similarity_to_query[idx]

                diversity = max(
                    cosine_similarity(
                        [candidate_embeddings[idx]],
                        [
                            candidate_embeddings[i]
                            for i in selected_indices
                        ]
                    )[0]
                )

                mmr_score = (
                    lambda_param * relevance
                    -
                    (1 - lambda_param) * diversity
                )

                mmr_scores.append(
                    (
                        idx,
                        mmr_score,
                        relevance,
                        diversity
                    )
                )

            best_index, best_mmr, rel, div = max(
                mmr_scores,
                key=lambda x: x[1]
            )

            print(
                f"Selected Chunk {best_index + 1} | "
                f"MMR={best_mmr:.4f} | "
                f"Relevance={rel:.4f} | "
                f"Diversity={div:.4f}"
            )

            selected_indices.append(best_index)

        print("\nFinal Selected Chunks:")

        for idx in selected_indices:

            print(
                f"Chunk {idx + 1}"
            )

        print("=" * 80)

        return [

            candidate_chunks[idx]

            for idx in selected_indices

        ]