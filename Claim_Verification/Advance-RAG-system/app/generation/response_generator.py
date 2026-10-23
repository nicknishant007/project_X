from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder
from guardrails.output_guard import OutputGuard


class ResponseGenerator:

    @staticmethod
    def stream_response(query, retrieved_chunks):

        print("\n" + "=" * 120)
        print("🔍 RESPONSE GENERATOR")
        print("=" * 120)

        print(f"\nQuestion: {query}")
        print(f"\nRetrieved Chunks: {len(retrieved_chunks)}")

        if not retrieved_chunks:
            print("\n❌ No chunks received!")
        else:
            for i, chunk in enumerate(retrieved_chunks, start=1):

                print("\n" + "-" * 100)
                print(f"CHUNK #{i}")

                print("\nMetadata:")
                print(chunk.metadata)

                print("\nLength:", len(chunk.page_content))

                print("\nContent:")
                print(chunk.page_content)

        # --------------------------------------------------------
        # Build Context
        # --------------------------------------------------------

        context = "\n\n".join(
            chunk.page_content
            for chunk in retrieved_chunks
        )

        print("\n" + "=" * 120)
        print("FINAL CONTEXT")
        print("=" * 120)

        print(context)
        print("\nContext Length:", len(context))

        print("=" * 120 + "\n")

        # --------------------------------------------------------
        # Build Chain
        # --------------------------------------------------------

        prompt = PromptBuilder.build()

        llm = LLM.get_llm()

        chain = prompt | llm

        print("🚀 Sending prompt to Gemini...\n")

        # --------------------------------------------------------
        # Generate Complete Answer
        # --------------------------------------------------------

        answer = ""

        try:

            first_token = True

            for token in chain.stream(
                {
                    "context": context,
                    "question": query,
                }
            ):

                content = getattr(token, "content", None)

                if not content:
                    continue

                if first_token:
                    print("✅ Gemini started generating.\n")
                    first_token = False

                print(content, end="", flush=True)

                answer += content

        except Exception as e:

            print(f"\n❌ LLM Generation Failed: {e}")

            raise

        print("\n")

        # --------------------------------------------------------
        # Output Guard
        # --------------------------------------------------------

        safe, reason = OutputGuard.validate(
            question=query,
            answer=answer,
            context=context,
        )

        if not safe:

            print(f"❌ Output Guard Blocked Response")
            print(f"Reason: {reason}")

            answer = (
                "I couldn't answer this question reliably using the retrieved "
                "documents. Please try rephrasing your question or provide "
                "more relevant documents."
            )

        else:

            print("✅ Output Guard Passed")

        # --------------------------------------------------------
        # Stream Safe Response
        # --------------------------------------------------------

        print("\n🚀 Streaming response to client...\n")

        for ch in answer:
            yield ch

        print("\n" + "=" * 120)
        print("✅ STREAM FINISHED")
        print("=" * 120)