from typing import Dict, List, Tuple

from .structures import Label, Sample


def create_unique_doc_id(title: str, context: str) -> str:
    """Create a unique document ID using the title and first few words of the context."""
    # TODO: what is the best way to create the unique ID? Ask during meeting!
    # ? should we include "section_title", "background", etc in the ID?
    return f"{title}_{context[:50].replace(' ', '_')}"


def get_docs(data: Dict) -> Dict[str, str]:
    """Extract documents from QuAC data.

    Args:
        data: Raw QuAC data

    Returns:
        Dictionary mapping section IDs to their text content
    """
    # Questions
    # - what is the best way to create the unique ID?

    docs = {}
    for article in data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            # Create unique doc ID using title and first few words of context
            doc_id = create_unique_doc_id(title, context)
            docs[doc_id] = context
    return docs


def get_XY(data: Dict) -> Tuple[List[Sample], List[Label]]:
    """Convert QuAC data into X (samples) and Y (labels) pairs.

    Args:
        data: Raw QuAC data

    Returns:
        Tuple containing:
        - List of Sample objects (X)
        - List of Label objects (Y)
    """
    x_samples = []
    y_labels = []

    # ? Questions:
    # - Do we need to keep track of the conversation history? Cuz I think QuAC just references a direction citation from the context, does it use the conversation history?
    # and quac requires the answers to be verbatim directly from the context. it cannot be free-form, is this fine?
    # - Do we need to use the "followup" field?
    # - Do we need to use the yes/no field
    # - Should I add the qa['id'] to the conversation history?
    # - Do I need to add the document_relevant field since in QuAC the doc is always relevant?
    # - does the train/test_Y.json have to match exactly with the way multiwoz does it? do the fields have to be the same?
    # - what exactly is the segments field in multiwoz? and why is it a list? is it just the answer, or the citation?
    # - do i need a segments field and a answer field? or just one is fine? what's the difference?
    # - should I store the span_start in the y_labels?
    # - what do i do when the answer is CANNOTANSWER? should the document_relevant be false then? should segments be empty?
    # - should i put 'user' or 'student' in the role field?

    for article in data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            doc_id = create_unique_doc_id(title, context)

            # Track conversation history
            conv_history = []

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][
                    0
                ]  # Take first answer as ground truth (that's what huggingface said you can do)

                # add the current question to the conversation history
                conv_history.append(
                    {
                        "role": "user",  # TODO: should i put 'user' or 'student'?
                        "content": question,
                        "q_id": qa["id"],  # TODO: do we need this?
                    }
                )

                x_samples.append(
                    Sample(
                        document_ids=[
                            doc_id
                        ],  # For QuAC, there is only one relevant context/document, not multiple
                        conversation=conv_history.copy(),
                        id=qa["id"],
                    )
                )

                y_labels.append(
                    Label(
                        document_relevant=True,  # TODO: do i need this? the doc is always relevant for this dataset
                        segments=[
                            answer["text"]
                        ],  # TODO: is this the answer or the citation?
                        answer=answer["text"],
                        # TODO: should I store the span_start and the question_id in the y_labels?
                        q_id=qa["id"],
                        span_start=answer["answer_start"],
                        span_end=answer["answer_start"] + len(answer["text"]),
                    )
                )

    return x_samples, y_labels
