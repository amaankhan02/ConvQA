import time
import torch
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict

from .structures import *
from .graph.summary_tree import SummaryTree
from .response import affirmative_resp, list_words, segments_to_edges

# TODO: Clean up to not need e2i, i2e, r2i, i2r
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

        self.e2i = {}
        self.r2i = {}
        self.i2e = {}
        self.i2r = {}
        self.edge_to_excerpt = defaultdict(list)
    
    def _extract_keyword_context(self, document: str, keyword: str, context_size: int = 100) -> list:
        """
        Extracts a specified number of characters around each occurrence of a keyword in the document.

        Args:
            document (str): The text document to search within.
            keyword (str): The keyword to find.
            context_size (int): The total number of characters (split equally on both sides) for context.

        Returns:
            list: A list of strings, each containing the keyword and its surrounding context.
        """
        results = []
        keyword_length = len(keyword)
        half_context = context_size // 2

        # Find all keyword occurrences
        start = 0
        start = document.find(keyword, start)
        while start != -1:
            # Calculate context bounds
            start_context = max(0, start - half_context)
            end_context = min(len(document), start + keyword_length + half_context)

            # Extract context
            context = document[start_context:end_context]
            results.append(context)

            # Move past the current keyword
            start += keyword_length
            start = document.find(keyword, start)

        return results

    def __call__(self, X: Sample, docs: Dict[str, str]) -> Label:
        # Assume if len(conversation) == 1, new conversation start. Reset entities & relations.
        if len(X.conversation) == 1:
            self.e2i = {}
            self.r2i = {}
            self.i2e = {}
            self.i2r = {}
            self.edge_to_excerpt = defaultdict(list)
            
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
                # TODO
                raise Exception(f"Combination not supported:\n\tSummary Tree: {self.summary_trees is not None}\n\tDialogue KG: {self.use_dialogue_kg}")
        else:  # self.use_dialogue_kg is True
            relevant_segments = []

            if len(self.e2i) > 1:
                history = [
                    {
                        "role": "user",
                        "content": (f"Given the following list of entities and relations, please translate the following query in "
                                  f"subject_entity|relation|tail_entity form. Do not include anything else. \n"
                                  f"Query: {final_query}\n"
                                  f"Entities: {'\n'.join(self.e2i.keys())}\n"
                                  f"Relations: {'\n'.join(self.r2i.keys())}")
                    }
                ]
                outputs = self.model(history, max_new_tokens=64)
                response = outputs[0]["generated_text"][-1]["content"].lower()
                if response in self.edge_to_excerpt:
                    relevant_segments = self.edge_to_excerpt[response]
                
            # If no relevant document segments, look at document (tree)
            if len(relevant_segments) < 1 and self.summary_trees:
                current_nodes = [self.summary_trees[doc_id].root for doc_id in X.document_ids]
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
                            current_nodes.append(current_node)

            # If no relevant document segments, do a keyword search
            if len(relevant_segments) < 1:
                # Identify potential keywords that relate to the query.
                keywords = ["".join([c for c in keyword if c.isalpha() or c.isspace()]) for keyword in list_words(self.model, [
                    {
                        "role": "user",
                        "content": f"Please provide keywords separated by a comma to search the document for, verbatim, in a document to answer the following query. Please note each comma-separated keyword will be used to retrieve sentences from the document, so find keywords that will find sentences from the document that may be relevant to answer the question. \nQuery: {final_query}",
                    }
                ])]
                print(keywords)
                # Find all in-context instances of the keyword in the document.
                for keyword in keywords:
                    for doc_id in X.document_ids:
                        preprocessed_document = "".join([c for c in docs[doc_id].lower() if c.isalpha() or c.isspace()])
                        relevant_segments.extend(self._extract_keyword_context(document=preprocessed_document, keyword=keyword))
            
            # Answer based on context
            if relevant_segments:
                document_relevant = True
                segments = relevant_segments
                history = [
                    {
                        "role": "system",
                        "content": f"Use the following documents to answer the question:\n" + "\n".join([f"<div>{segment}</div>" for segment in relevant_segments]),
                    },
                ] + X.conversation
                answer = self.model(history, max_new_tokens=256)[0]["generated_text"][-1]["content"]

                # Add edge associated with each relevant segment
                edges = segments_to_edges(self.model, relevant_segments, self.e2i, self.r2i, self.i2e, self.i2r, self.edge_to_excerpt)
                
        return Label(
            document_relevant=document_relevant,
            segments=segments,
            answer=answer,
            time_taken=time.time() - start,
        )

    def load_summary_trees(self, summary_trees_fp: str) -> None:
        self.summary_trees = {
            k: SummaryTree.from_dict(v) 
            for k, v in json.load(open(summary_trees_fp, "r")).items()
        } 

    def generate_summary_trees(self, summary_trees_fp: str, docs: Dict[str, str], emb_model: Any) -> None:
        summary_trees = {}
        for doc_id, doc in tqdm(docs.items()):
            summary_trees[doc_id] = SummaryTree(None)
            summary_trees[doc_id].generate_from(doc, self.model, emb_model)
            
            with open(summary_trees_fp, "w") as f:
                json.dump({k: v.to_dict() for k, v in summary_trees.items()}, f, indent=4)
        self.summary_trees = summary_trees
