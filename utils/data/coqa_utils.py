from typing import Dict, List, Tuple

from utils.structures import Label, Sample
from utils.constants import ANSWER_DELIM

def get_docs(data: Dict) -> Dict[str, str]:
    """Extract the documents from CoQA dataset and return as a dictionary
    where keys are the document IDs and values are the document text.

    Args:
        data: Raw CoQA dataset dictionary (JSON)

    Returns:
        Dictionary mapping document IDs to their text content
    """

    docs = {}
    for item in data["data"]:
        doc_id = str(item["id"])
        docs[doc_id] = item["story"]

    return docs


def get_XY(data: Dict) -> Tuple[List[Sample], List[Label]]:
    """Extract input samples (X) and labels (Y) from CoQA dataset.

    Args:
        data: Raw CoQA dataset dictionary (JSON)

    Returns:
        Tuple containing:
        - List of Sample objects (X) with conversation context
        - List of Label objects (Y) with answers
    """
    # Questions:

    samples = []
    labels = []

    for item in data["data"]:
        doc_id = str(item["id"])
        questions = []
        all_answers = []    # 2d array. all_answers[i] = list of answers for question[i]. all_answers[i][j] represents the jth possible valid answer for the ith question

        # Extract QA pairs from conversation
        for qa in item["questions"]:
            questions.append(qa["input_text"])
        for ans in item["answers"]:
            all_answers.append([
                {
                    "text": ans[
                        "input_text"
                    ],  # this is the answer text (drawn from the span_text)
                    "span_text": ans[
                        "span_text"
                    ],  # direct verbatim answer from the context
                    
                    # NOTE: not saving the span_start and span_end for now since we don't need them
                }]
            )
            
        # Add check for additional_answers key
        if 'additional_answers' in item:
            for version in item['additional_answers']:
                i = 0
                for curr_answer in item['additional_answers'][version]:
                    all_answers[i].append({
                        "text": curr_answer['input_text'],
                        "span_text": curr_answer['span_text']
                    })
                    i += 1                

        # Create samples and labels for each turn in conversation
        conv_history = []
        for question, curr_qs_answers in zip(questions, all_answers):
            conv_history.append({"role": "user", "content": question})

            samples.append(
                Sample(
                    document_ids=[doc_id],
                    conversation=conv_history.copy(),
                )
            )            
                
            labels.append(
                Label(
                    document_relevant=any([curr_ans['span_text'] != 'unknown' for curr_ans in curr_qs_answers]),
                    segments=[curr_ans["span_text"] for curr_ans in curr_qs_answers],
                    answer=ANSWER_DELIM.join([curr_ans["text"] for curr_ans in curr_qs_answers]),
                )
            )

    return samples, labels
