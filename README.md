# DTCRS: Dynamic Tree Construction for Recursive Summarization

## Introduction
**DTCRS** is a novel method for enhancing Retrieval-Augmented Generation (RAG) by dynamically constructing hierarchical summary trees based on document structure and query semantics. Traditional recursive summarization methods often generate redundant summary nodes, leading to increased construction time and potential degradation in question answering performance. DTCRS addresses these limitations by:
1. **Question-Type Analysis**: Determines if a summary tree is necessary based on the question type.
2. **Dynamic Tree Construction**: Decomposes questions into sub-questions and uses their embeddings as initial cluster centers, reducing redundancy.
3. **Efficiency & Relevance**: Significantly reduces construction time while improving the relevance of summaries to the target question.

Our approach achieves substantial improvements across three QA tasks ([NarrativeQA](https://github.com/google-deepmind/narrativeqa), [QASPER](https://github.com/allenai/qasper-led-baseline), [QuALITY](https://github.com/nyu-mll/quality?tab=readme-ov-file)).

---

## Quick Start

### 1 Install dependencies from the requirements file

Make sure to install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### 2 Configure API Keys and URLs in the env file

Set your API key and base URL in the  `.env` configuration file:

```bash
# API KEY
OPENAI_API_KEY=
# BASE URL
OPENAI_API_BASE=
# LOCAL URL
LOCAL_MODEL_API_URL=
```

### 3 Run demo.py

In `demo.py`, replace the document and question with the appropriate content:
```bash
full_paper_text = "" 
question = ""
```
Initialize the summary tree and select the models:
```bash
config = RetrievalAugmentationConfig(
        embedding_model=SBertEmbeddingModel(),
        qa_model=GPT4QAModel(),
        summarization_model=GPT4SummarizationModel()
    )

summary_tree = RetrievalAugmentation(config=config)
```
Use the summary tree to retrieve evidence and predict the answer:
```bash
summary_tree.add_documents_and_queries(full_paper_text, sub_question_list, cluster_method="global")
predicted_evidence = summary_tree.retrieve(question=question, return_layer_information=False)
predicted_answer = summary_tree.answer_question(question=question, answer_type="generate")
```
## Prediction
The prediction files are located in three datasets under the `data` directory. Please run them in their respective directories.

#### NarrativeQA
```bash
python predict.py --output_file "path_to_output.jsonl" --test_file "path_to_input.csv" --content_dir "path_to_content_directory" --cluster_method "global" --answer_type "generate"
```
- `--output_file` should be the path to the output JSONL file.
- `--test_file` should be the path to the input CSV file.
- `--content_dir` should be the directory where .content files are stored.
- `--cluster_method` can be either "global" or "hierarchical" (default is "global").
- `--answer_type` can be either "choose" or "generate" (default is "generate").
#### QASPER
```bash
python predict.py --output_file "path_to_output.jsonl" --test_file "path_to_input.jsonl" --cluster_method "global" --answer_type "generate"
```
#### QuALITY
```bash
python predict.py --output_file "path_to_output.jsonl" --test_file "path_to_input.jsonl" --cluster_method "global" --answer_type "choose"
```


## Evaluation

#### NarrativeQA
only the result file needs to be input during evaluation, as we have included both the predicted answers and the correct answers in the result.
```bash
python NarrativeQA_evaluator.py --predictions /path/to/your/predictions.jsonl
```
#### QASPER
```bash
python qasper_evaluator.py --predictions /path/to/your/predictions.jsonl --gold /path/to/your/gold.jsonl
```
#### QuALITY
The QuALITY test set does not provide gold answers, so we do not provide a script. To view the prediction results, please first run `data/QuALITY/code/jsonl2txt.py` to convert the prediction file into TXT format. Then, submit the Google Form at [this link](https://docs.google.com/forms/d/e/1FAIpQLSdFBTnD-RoND30qrchQJTps2AGCrpx4h1T9IQNAgyxadFzZ9Q/viewform).



