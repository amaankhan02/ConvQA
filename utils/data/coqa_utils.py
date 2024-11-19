from typing import Dict, List, Tuple

from .structures import Label, Sample


def get_docs(data: Dict) -> Dict[str, str]:
    """Extract the documents from CoQA dataset and return as a dictionary
    where keys are the document IDs and values are the document text.

    Args:
        data: Raw CoQA dataset dictionary (JSON)

    Returns:
        Dictionary mapping document IDs to their text content
    """
    # Questions:
    # - we can ignore the 'source', 'filename', 'name' fields, right? because we are using the 'id' field to create the doc IDs?

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
    # - should we store input_text and span_text in the Label object?
    # - what is the additional_answers field? TODO: read on this
    # - should i put 'user' or 'student' in the role field?
    # - should i make the question_id (q_id) the turn_id?
    # - I added a new field called 'id' to the Sample object, is that okay?
    # - ** In the structures.py, in Sample, why don't we have a question field? or do we just assume the current question in the last qs in the conversation history?
    # - Why is the document ID not in the Label object?

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
            answers.append(
                {
                    "text": ans[
                        "input_text"
                    ],  # this is the answer text (drawn from the span_text)
                    "span_start": int(ans["span_start"]),
                    "span_end": int(ans["span_end"]),
                    "span_text": ans[
                        "span_text"
                    ],  # direct verbatim answer from the context
                }
            )

        # Create samples and labels for each turn in conversation
        conv_history = []
        turn_id = 1
        for question, answer in zip(questions, answers):
            conv_history.append({"role": "user", "content": question, "q_id": turn_id})

            samples.append(
                Sample(
                    document_ids=[doc_id],
                    conversation=conv_history.copy(),
                    id=str(turn_id),
                )
            )

            labels.append(
                Label(
                    document_relevant=True,  # ? It's always relevant for this dataset, right?
                    segments=[answer["span_text"]],
                    answer=answer["text"],
                    q_id=str(turn_id),
                    span_start=answer["span_start"],
                    span_end=answer["span_end"],
                )
            )

            turn_id += 1

    return samples, labels
