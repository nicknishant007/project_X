import os
import pickle

from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class BM25Indexer:

    documents: List = []

    tokenized_docs: List = []

    bm25: BM25Okapi | None = None

    def __init__(self):

        self.documents = []

        self.tokenized_docs = []

        self.bm25 = None

        self.storage_path = (
            BASE_DIR
            / "storage"
            / "bm25"
            / "bm25_store.pkl"
        )

        os.makedirs(
            os.path.dirname(self.storage_path),
            exist_ok=True
        )

        self.load()

    def add_documents(
        self,
        chunks
    ):

        texts = []

        valid_chunks = []

        for chunk in chunks:

            text = chunk.page_content.strip()

            if not text:
                continue

            texts.append(text)

            valid_chunks.append(chunk)

        if not texts:
            return

        self.documents.extend(
            valid_chunks
        )

        tokenized = [

            text.lower().split()

            for text in texts

        ]

        if not tokenized:
            return

        self.tokenized_docs.extend(
            tokenized
        )

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

        self.save()

    def search(
        self,
        query,
        top_k=7
    ):

        if self.bm25 is None:
            return []

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(
                self.documents,
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        return [

            chunk

            for chunk, _ in ranked[:top_k]

        ]

    def save(self):

        data = {

            "documents": self.documents,

            "tokenized_docs": self.tokenized_docs

        }

        with open(
            self.storage_path,
            "wb"
        ) as f:

            pickle.dump(
                data,
                f
            )

    def load(self):

        if not os.path.exists(
            self.storage_path
        ):
            return

        if os.path.getsize(
            self.storage_path
        ) == 0:
            return

        try:

            with open(
                self.storage_path,
                "rb"
            ) as f:

                data = pickle.load(f)

            self.documents = data.get(
                "documents",
                []
            )

            self.tokenized_docs = data.get(
                "tokenized_docs",
                []
            )

            if self.tokenized_docs:

                self.bm25 = BM25Okapi(
                    self.tokenized_docs
                )

        except Exception as e:

            print(
                f"BM25 load failed: {e}"
            )