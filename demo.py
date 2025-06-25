import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dctrs import (
    SBertEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation,
    GPT4QAModel, GPT4SummarizationModel, GPTModule
)


def initialize_retrieval_augmentation():
    """Initialize RetrievalAugmentation configuration."""
    config = RetrievalAugmentationConfig(
        embedding_model=SBertEmbeddingModel(),
        qa_model=GPT4QAModel(),
        summarization_model=GPT4SummarizationModel()
    )
    return RetrievalAugmentation(config=config)


def process_question(question, toc, full_paper_text, summary_tree, llm_module):
    """Process a single question and return predicted answers."""
    complex_question = llm_module.is_complex_question(question, toc)
    print("need to summarize? →",complex_question)

    if complex_question and toc:
        sub_question_list = llm_module.decompose_question(question, toc)
        summary_tree.add_documents_and_queries(full_paper_text, sub_question_list, cluster_method="global")
        predicted_evidence = summary_tree.retrieve(question=question, return_layer_information=False)
        predicted_answer = summary_tree.answer_question(question=question, answer_type="generate")
    else:
        summary_tree.add_faiss_documents(full_paper_text)
        predicted_evidence = summary_tree.retrieve_faiss(question)
        predicted_answer = summary_tree.answer_question_faiss(question=question, context=predicted_evidence,
                                                    answer_type="generate")

    return predicted_answer, predicted_evidence


if __name__ == "__main__":
    # Replace with your full text and question.
    full_paper_text = "What are the key steps involved in implementing a BIM system for bridge design?"
    question = '''When implementing a BIM system for bridge design, the first focus should be on the external interfaces with management software. These interfaces handle the data exchange between the system and the central management platform, ensuring seamless information flow. The design of these external interfaces is crucial, as it involves compatibility with other systems and efficient collaboration. On the other hand, the design of internal interfaces mainly concerns the interaction and data transmission between different functional modules within the system. These modules include user-defined strategy management, automated report generation, and report generation enhanced by large language models (LLM).

    In the setup of internal interfaces, the report generation function is particularly important, as it relies not only on the accuracy of the data but also on how automation can improve efficiency. Additionally, the system should support data push modes, which help facilitate the rapid transfer of information, especially when real-time synchronization occurs across multiple systems and devices.

    For the bridge design itself, the BIM system needs to perform precise coordinate and length calculations, particularly for the main longitudinal beams. Ensuring the accuracy of the dimensions and positions of these structural elements is critical to the safety of the bridge. Specifically, the alignment of the bridge piers in the longitudinal direction is essential to guarantee the structural stability and long-term durability of the bridge.'''
    summary_tree = initialize_retrieval_augmentation()
    llm_module = GPTModule()
    toc = llm_module.generate_table_of_contents(full_paper_text)
    predicted_answer, predicted_evidence = process_question(question, toc, full_paper_text,
                                                            summary_tree, llm_module)
    print(predicted_evidence)
    print(predicted_answer)