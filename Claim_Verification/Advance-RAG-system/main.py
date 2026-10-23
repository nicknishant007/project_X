from app.ingestion.registry.registry_manager import (
    RegistryManager
)

from app.ingestion.pipeline.ingestion_pipeline import (
    IngestionPipeline
)

from app.ingestion.vectordb.qdrant_client import (
    QdrantManager
)

from app.generation.rag_pipeline import (
    RAGPipeline
)


def run_ingestion():

    RegistryManager.initialize()

    qdrant = QdrantManager()

    qdrant.initialize_collection()

    chunks = IngestionPipeline.run()

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )


def run_chat():

    rag = RAGPipeline()

    while True:

        query = input(
            "\nAsk Question (or 'exit'): "
        )

        if query.lower() == "exit":

            break

        print("\n===================")
        print("ANSWER")
        print("===================\n")

        for token in rag.stream_query(query):

            print(
                token,
                end="",
                flush=True
            )

        print("\n")


def main():

    print("\n1. Run Ingestion")
    print("2. Start RAG Chat")

    choice = input("\nSelect Option: ")

    if choice == "1":

        run_ingestion()

    elif choice == "2":

        run_chat()

    else:

        print("Invalid Choice")


if __name__ == "__main__":

    main()