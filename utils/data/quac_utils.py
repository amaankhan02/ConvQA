from typing import Dict, List, Tuple

from utils.structures import Label, Sample
from utils.constants import ANSWER_DELIM

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

    for article in data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            doc_id = create_unique_doc_id(title, context)
            conv_history = []

            for qa in paragraph["qas"]:
                question = qa["question"]

                # Take first answer as ground truth (that's what huggingface said you can do)
                answers = qa["answers"]

                # add the current question to the conversation history
                conv_history.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                x_samples.append(
                    Sample(
                        # For QuAC, there is only one relevant context/document, not multiple
                        document_ids=[doc_id],
                        conversation=conv_history.copy()
                    )
                )

                is_doc_relevant = False
                for ans in answers:
                    if ans["text"] != "CANNOTANSWER":
                        is_doc_relevant = True
                        break
                
                y_labels.append(
                    Label(
                        document_relevant=is_doc_relevant,
                        segments=[ans["text"] for ans in answers],
                        answer = ANSWER_DELIM.join([ans["text"] for ans in answers]),
                    )
                )

    return x_samples, y_labels
