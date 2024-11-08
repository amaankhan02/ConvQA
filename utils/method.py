import time
import torch
from transformers import pipeline

from .structures import *


class ConvRef:
    def __init__(
        self, model: str, no_summary_tree: bool = False, no_dialogue_KG: bool = False
    ) -> None:
        self.model = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.model.generation_config.pad_token_id = (
            self.model.tokenizer.eos_token_id
        )

        self.no_summary_tree = no_summary_tree
        self.no_dialogue_KG = no_dialogue_KG

    def __call__(self, X: Sample, docs: Dict[str, str]) -> Label:
        start = time.time()
        doc_context = "\n".join(
            [f"<div>{docs[doc_id]}</div>" for doc_id in X.document_ids]
        )
        history = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions: {doc_context}",
            }
        ] + X.conversation

        outputs = self.model(
            history
            + [
                {
                    "role": "user",
                    "content": 'Is this a question and if so, are the provided document(s) relevant to answer the above question? Answer "YES" or "NO" only.',
                }
            ],
            max_new_tokens=10,
        )
        response = outputs[0]["generated_text"][-1]["content"]

        document_relevant = "NO" not in response and "YES" in response

        segments = None
        answer = None
        if document_relevant:
            outputs = self.model(
                history,
                max_new_tokens=256,
            )
            answer = outputs[0]["generated_text"][-1]["content"]

        return Label(
            document_relevant=document_relevant,
            segments=segments,
            answer=answer,
            time_taken=time.time() - start,
        )
