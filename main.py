from src.pipeline import RAGPipeline

def main():
    # Initialize and load pipeline
    pipeline = RAGPipeline()
    pipeline.load()

    print("\nRAG Pipeline ready! Type our question below.")
    print("   Commands: 'quit' to exit | 'batch' to run batch test\n")

    while True:
        question = input("our question: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question.lower() == "batch":
            # Run a batch of test questions
            test_questions = [
                "What is Naive RAG?",
                "What are the limitations of RAG systems?",
                "How does Advanced RAG improve upon Naive RAG?",
                "What is the role of the retriever in RAG?",
                "What evaluation metrics are used for RAG?",
            ]
            print(f"\nRunning batch of {len(test_questions)} questions...\n")
            results = pipeline.run_batch(test_questions)
            for r in results:
                pipeline.display(r)
        else:
            result = pipeline.query(question)
            pipeline.display(result)


if __name__ == "__main__":
    main()