import time
import torch
from tqdm import tqdm
from transformers import pipeline

from .structures import *
from .graph.summary_tree import SummaryTree
from .response import affirmative_resp


class ConvRef:
    def __init__(
        self, model: str, use_dialogue_kg: bool
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

        self.summary_trees = None
        self.use_dialogue_kg = use_dialogue_kg
        self.entities = set()
        self.relations = set()

    def __call__(self, X: Sample, docs: Dict[str, str]) -> Label:
        start = time.time()
        final_query = X.conversation[-1]["content"]
        relevance_eval = [
            {
                "role": "user",
                "content": f"Is this a question: \"{final_query}\"",
            }
        ]
        document_relevant = affirmative_resp(self.model, relevance_eval)
        segments = None
        answer = None
        if not document_relevant:
            return Label(
                document_relevant=document_relevant,
                segments=segments,
                answer=answer,
                time_taken=time.time() - start,
            )
        if not self.use_dialogue_kg:
            if self.summary_trees is None:
                doc_context = "\n".join(
                    [f"<div>{docs[doc_id]}</div>" for doc_id in X.document_ids]
                )
                history = [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Answer each question by quoting from the document. Documents: {doc_context}",
                    }
                ] + X.conversation

                relevance_convo = history + [
                    {
                        "role": "user",
                        "content": "Are the provided document(s) relevant to answer the above question?",
                    }
                ]
                document_relevant = affirmative_resp(self.model, relevance_convo)
                if document_relevant:
                    outputs = self.model(
                        history,
                        max_new_tokens=256,
                    )
                    answer = outputs[0]["generated_text"][-1]["content"]
            else:
                raise Exception(f"Combination not supported:\n\tSummary Tree: {self.summary_trees is not None}\n\tDialogue KG: {self.use_dialogue_kg}")
        else:  # self.use_dialogue_kg is True
            relevant_segments = []
            # - Given entities and relations, ask LLM to construct knowledge graph edge.
            #     - Knowledge graph does not exist yet in the first pass
            # - Try to find on knowledge graph.
            # - If edge exists, for each associated document segments.
            if self.entities and self.relations:
                # Ask LLM to construct KG edge
                # Try to find on KG
                pass
                
            # If no relevant document segments, look at document (tree)
            if not relevant_segments and self.summary_trees:
                current_nodes = [self.summary_trees[doc_id] for doc_id in X.document_ids]
                relevant_nodes = []
                # BFS
                while current_nodes:
                    current_node = current_nodes.pop(0)
                    current_nodes.extend(current_node.children)
                    segment_relevance_eval = [
                        {
                            "role": "user",
                            "content": f"Here is a summary of a text: {current_node.data}. Is this text useful to answer the following question: \"{final_query}\"",
                        }
                    ]
                    segment_relevant = affirmative_resp(self.model, segment_relevance_eval)
                    if segment_relevant:
                        if not current_node.children:
                            relevant_nodes.append(current_node)
                        else:
                            relevant_current_nodes.append(current_node)

            # If no relevant document segments, do a keyword search
            if not relevant_segments:
                #     - If no related passages found, identify potential keywords that relate to the query.
                #         - For each keyword, identify the sentences that contain it.
                #             - For each sentence, expand context until sure whether it is relevant or max size reached.
                #                 - If still undecided, reject.
                # - For all segments that are decided as relevant (if segment is expanded to decide, use the expanded version of the segment), add it to the context for deciding on the answer
                pass
            # Add edge associated with each relevant segment
            # Answer based on context
        return Label(
            document_relevant=document_relevant,
            segments=segments,
            answer=answer,
            time_taken=time.time() - start,
        )

    def load_summary_trees(self, summary_trees_fp: str) -> None:
        self.summary_trees = json.load(open(summary_trees_fp, "r"))

    def generate_summary_trees(self, summary_trees_fp: str, docs: Dict[str, str], emb_model: Any) -> None:
        summary_trees = {}
        for doc_id, doc in tqdm(docs.items()):
            summary_trees[doc_id] = SummaryTree(None)
            summary_trees[doc_id].generate_from(doc, self.model, emb_model)
            
            with open(summary_trees_fp, "w") as f:
                json.dump({k: v.to_dict() for k, v in summary_trees.items()}, f, indent=4)
        self.summary_trees = summary_trees
