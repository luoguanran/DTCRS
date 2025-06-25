""" Evaluation script for NarrativeQA dataset. """
import json
import nltk
import argparse
# nltk.download("punkt")
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")
#     nltk.download("wordnet")

import rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import copy

rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type="words",
    apply_avg=True,
    apply_best=True,
    alpha=0.5,
    weight_factor=1.2,
    stemming=True,
)


def bleu_1(p, g):
    return sentence_bleu(g, p, weights=(1, 0, 0, 0))


def bleu_4(p, g):
    return sentence_bleu(g, p, weights=(0, 0, 0, 1))


def meteor(p, g):
    return meteor_score([x.split() for x in g], p.split())


def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenize=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if tokenize:
            score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
        else:
            score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and "rouge-l" in score:
        max_score = copy.deepcopy(score)
        max_score["rouge-l"]["f"] = round(
            max([score["rouge-l"]["f"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["p"] = round(
            max([score["rouge-l"]["p"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["r"] = round(
            max([score["rouge-l"]["r"] for score in scores_for_ground_truths]), 2
        )
        return max_score
    else:
        return round(max(scores_for_ground_truths), 2)


def get_metric_score(prediction, ground_truths):
    bleu_1_score = metric_max_over_ground_truths(bleu_1, prediction, ground_truths, tokenize=True)
    bleu_4_score = metric_max_over_ground_truths(bleu_4, prediction, ground_truths, tokenize=True)
    meteor_score = metric_max_over_ground_truths(meteor, prediction, ground_truths, tokenize=False)
    rouge_l_score = metric_max_over_ground_truths(
        rouge_l, prediction, ground_truths, tokenize=False
    )

    return (
        bleu_1_score,
        bleu_4_score,
        meteor_score,
        rouge_l_score["rouge-l"]["f"],
        rouge_l_score["rouge-l"]["p"],
        rouge_l_score["rouge-l"]["r"],
    )


def read_jsonl(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        data = []
        for line in file:
            try:
                entry = json.loads(line)
                if 'predicted_answer' in entry and 'answers' in entry:
                    data.append(entry)
                else:
                    print(f"Skipping invalid line (missing predicted_answer or answers): {line}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
        return data


if __name__ == "__main__":
    # Load predictions and ground truths from a JSONL file
    parser = argparse.ArgumentParser(description="Evaluation script for NarrativeQA dataset.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()

    data = read_jsonl(args.predictions)

    # Lists to store scores for each metric
    bleu_1_scores = []
    bleu_4_scores = []
    meteor_scores = []
    rouge_l_f_scores = []
    rouge_l_precision_scores = []
    rouge_l_recall_scores = []
    all_scores = []  # List to store all scores for saving

    # Iterate over each entry in the data
    for entry in data:
        prediction = entry['predicted_answer']
        ground_truths = entry['answers']

        # Get the metric scores
        scores = get_metric_score(prediction, ground_truths)

        # Append scores to respective lists
        bleu_1_scores.append(scores[0])
        bleu_4_scores.append(scores[1])
        meteor_scores.append(scores[2])
        rouge_l_f_scores.append(scores[3])
        rouge_l_precision_scores.append(scores[4])
        rouge_l_recall_scores.append(scores[5])

        # Store the score for each entry
        all_scores.append([prediction, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5]])

    # Calculate average scores
    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_rouge_l_f = sum(rouge_l_f_scores) / len(rouge_l_f_scores) if rouge_l_f_scores else 0
    avg_rouge_l_precision = sum(rouge_l_precision_scores) / len(
        rouge_l_precision_scores) if rouge_l_precision_scores else 0
    avg_rouge_l_recall = sum(rouge_l_recall_scores) / len(rouge_l_recall_scores) if rouge_l_recall_scores else 0

    # Print the average scores
    print("Average BLEU-1 Score:", avg_bleu_1)
    print("Average BLEU-4 Score:", avg_bleu_4)
    print("Average METEOR Score:", avg_meteor)
    print("Average ROUGE-L F-Score:", avg_rouge_l_f)
    print("Average ROUGE-L Precision:", avg_rouge_l_precision)
    print("Average ROUGE-L Recall:", avg_rouge_l_recall))