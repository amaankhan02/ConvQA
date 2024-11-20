from typing import Dict, List, Tuple

from .structures import Label, Sample

ANSWER_DELIM = "||"

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
        answers = []

        # Extract QA pairs from conversation
        for qa in item["questions"]:
            questions.append(qa["input_text"])
        for ans in item["answers"]:
            answers.append([
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
            
        for version in item['additional_answers']:
            i = 0
            for curr_answer in item['additional_answers'][version]:
                answers[i].append({
                    "text": curr_answer['input_text'],
                    "span_text": curr_answer['span_text']
                })
                i += 1                

        # Create samples and labels for each turn in conversation
        conv_history = []
        for question, answer in zip(questions, answers):
            conv_history.append({"role": "user", "content": question})

            samples.append(
                Sample(
                    document_ids=[doc_id],
                    conversation=conv_history.copy(),
                )
            )

            is_doc_relevant = True if answer['span_text'] != 'unknown' else False
            labels.append(
                Label(
                    document_relevant=is_doc_relevant,
                    segments=[answer["span_text"]],
                    answer=ANSWER_DELIM.join([ans["text"] for ans in answer]),
                )
            )

    return samples, labels
