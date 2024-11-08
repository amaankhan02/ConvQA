from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from ..structures import *


def preprocess_labels(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for label in labels:
        turn_documents = label.get("knowledge", [])
        for document in turn_documents:
            for key, value in document.items():
                if type(value) != str:
                    document[key] = str(value)
    return labels


def get_docs(knowledge: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, str]:
    docs = {}

    for domain in knowledge:
        for doc in knowledge[domain]:
            key = f"{domain}_{doc}"
            doc_data = knowledge[domain][doc]
            qa = []
            for qa_item in doc_data["docs"].values():
                qa.append(f"Q: {qa_item['title']}\nA: {qa_item['body']}")
            docs[key] = f"FAQ FOR {doc_data['name']}\n\n" + "\n\n".join(qa)

    return docs


def get_XY(
    logs: List[List[Dict[str, str]]],
    labels: List[Dict[str, Any]],
    knowledge: Dict[str, Dict[str, Dict[str, str]]],
    docs: Dict[str, str],
) -> Tuple[List[Sample], List[Label]]:
    X = []
    Y = []

    dialogue = []
    documents = []
    for log, label in zip(logs, labels):
        # Check if is new conversation
        if len(log) == 1:
            # Update X
            documents = list(set(documents))
            for i in range(1, len(dialogue) + 1):
                X.append(
                    Sample(
                        document_ids=documents,
                        conversation=dialogue[:i],
                    )
                )
            # Is new conversation
            dialogue = []
            documents = []

        role = "user"
        if log[-1]["speaker"] == "S":
            role = "assistant"
        dialogue.append({"role": role, "content": log[-1]["text"]})

        turn_documents = label.get("knowledge", [])
        documents.extend(
            [
                f"{document['domain']}_{document['entity_id']}"
                for document in turn_documents
            ]
        )

        document_relevant = len(turn_documents) > 0
        correct_segment = None
        correct_answer = None
        if document_relevant:
            correct_segment = [
                knowledge[turn_document["domain"]][turn_document["entity_id"]]["docs"][
                    turn_document["doc_id"]
                ]["body"]
                for turn_document in turn_documents
            ]
            correct_answer = label["response"]

        Y.append(
            Label(
                document_relevant=document_relevant,
                answer=correct_answer,
                segments=correct_segment,
            )
        )

    # Update X
    documents = list(set(documents))
    for i in range(1, len(dialogue) + 1):
        X.append(
            Sample(
                document_ids=documents,
                conversation=dialogue[:i],
            )
        )

    return X, Y
