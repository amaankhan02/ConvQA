import re
import time
import torch
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict
from difflib import SequenceMatcher

from .structures import *
from .graph.summary_tree import SummaryTree
from .response import affirmative_resp, list_words, segments_to_edges

import spacy
nlp = spacy.load("en_core_web_lg")


# TODO: Clean up to not need e2i, i2e, r2i, i2r
class ConvRef:
    def __init__(
        self, model: str, llm_only: bool, strict: bool,
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
        self.llm_only = llm_only
        self.strict = strict

        # self.e2i = {}
        # self.r2i = {}
        # self.i2e = {}
        # self.i2r = {}
        # self.edge_to_excerpt = defaultdict(list)

    def _remove_near_duplicates(self, strings, similarity_threshold=0.9):
        """
        Removes near-duplicate strings from a list. Keeps the longest version of each near-duplicate group.
        
        Args:
            strings (list): A list of strings to process.
            similarity_threshold (float): The similarity threshold (default is 0.9).
            
        Returns:
            list: A list of unique strings with near-duplicates removed.
        """
        # Sort strings by length (longest first) to keep the longest version
        strings = sorted(strings, key=len, reverse=True)
        unique_strings = []
        
        for s in strings:
            is_duplicate = False
            for unique in unique_strings:
                # Check similarity
                similarity = SequenceMatcher(None, s, unique).ratio()
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_strings.append(s)
        
        return unique_strings

    def _extract_keyword_context(self, document: str, keyword: str) -> list:
        results = []
        keyword = keyword.lower()

        # Split the document into sentences using punctuation as delimiters
        sentences = re.split(r'(?<=[.!?]) +', document)
        
        # Iterate through sentences to find keyword matches
        for i, sentence in enumerate(sentences):
            if keyword in sentence.lower():
                # Get the sentence before, the current sentence, and the one after
                before = sentences[i - 1] if i > 0 else ""
                after = sentences[i + 1] if i < len(sentences) - 1 else ""
                
                # Combine context
                context = f"{before} {sentence} {after}".strip()
                results.append(context)

        return results

    def __call__(self, X: Sample, docs: Dict[str, str]) -> Label:
        start = time.time()
        doc_context = "\n".join(
            [f"<div>{docs[doc_id]}</div>" for doc_id in X.document_ids]
        )
        if self.llm_only:
            history = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Documents: {doc_context}",
                }
            ] + X.conversation
            document_relevant = affirmative_resp(self.model, history + [
                {
                    "role": "user",
                    "content": f"Are the document(s) relevant for answering the query?",
                }
            ])
            segments = None
            answer = None
            if document_relevant:
                history[0]["content"] = f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Answer with the single most relevant snippet from the document(s) verbatim and nothing else. Documents: {doc_context}",
                outputs = self.model(
                    history,
                    max_new_tokens=256,
                )
                answer = outputs[0]["generated_text"][-1]["content"]
                for doc_id in X.document_ids:
                    if answer in docs[doc_id]:
                        segments = [answer]
            return Label(
                document_relevant=document_relevant,
                segments=segments,
                answer=answer,
                time_taken=time.time() - start,
            )
        else:  # Ours
            final_query = X.conversation[-1]["content"]
            # doc_preview= "\n".join(
            #     [f"<div>{docs[doc_id]}</div>" if len(docs[doc_id]) < 250 else f"<div>{docs[doc_id][:100]}...{docs[doc_id][-100:]}</div>" for doc_id in X.document_ids]
            # )
            segments = None
            answer = None
            # Identify potential keywords that relate to the query.
            keywords = list_words(self.model, [
                {
                    "role": "user",
                    "content": f"Here are the document(s) separated by <div>s: {doc_context}\nThis is the query I want to answer: {final_query}\nIf the document may be able to answer the query, please provide keywords separated by a comma to search the document(s) for, verbatim, in a document to answer the following query. Otherwise, give no response.\nPlease note each comma-separated keyword will be used to retrieve sentences from the document, so find keywords that will find sentences from the document that may be relevant to answer the question.",
                }
            ]) + [ent.text for ent in nlp(final_query).ents]
            keywords = list(set([v.lower() for v in keywords]))
            print("KEYWORDS", keywords)

            # Find all in-context instances of the keyword in the document.
            relevant_segments = []
            for keyword in keywords:
                keyword = keyword.strip().lower()
                if keyword:
                    for doc_id in X.document_ids:
                        if keyword in docs[doc_id].lower():
                            relevant_segments.extend(self._extract_keyword_context(document=docs[doc_id], keyword=keyword))
            relevant_segments = self._remove_near_duplicates(relevant_segments)
            if len(relevant_segments) < 1:
                document_relevant = False
            else:
                excerpt_context = "\n".join(
                    [f"<div>{segment}</div>" for segment in relevant_segments]
                )
                document_relevant = True
                if self.strict:
                    document_relevant = affirmative_resp(self.model, [X.conversation[-1]] + [
                        {
                            "role": "user",
                            "content": f"Are the following excerpts relevant for answering the query? Excerpts: {excerpt_context}",
                        }
                    ])
                if document_relevant:
                    segments = relevant_segments
                    history = [
                        {
                            "role": "system",
                            "content": f"You are a helpful assistant. Answer with the single most relevant snippet from the document(s) verbatim and nothing else. Key Excerpts: {excerpt_context}",
                        }
                    ] + X.conversation

                    outputs = self.model(
                        history,
                        max_new_tokens=256,
                    )
                    answer = outputs[0]["generated_text"][-1]["content"]
                    for doc_id in X.document_ids:
                        if answer in docs[doc_id]:
                            segments = [answer]


            # final_query = X.conversation[-1]["content"]
            # doc_preview= "\n".join(
            #     [f"<div>{docs[doc_id]}</div>" if len(docs[doc_id]) < 250 else f"<div>{docs[doc_id][:100]}...{docs[doc_id][-100:]}</div>" for doc_id in X.document_ids]
            # )

            # history = [
            #     {
            #         "role": "system",
            #         "content": f"You are a helpful assistant. Here are a preview of the documents you can use to answer questions: {doc_preview}",
            #     }
            # ] + X.conversation
            # document_relevant = affirmative_resp(self.model, history + [
            #     {
            #         "role": "user",
            #         "content": f"Could the full-text version of the document(s) be relevant for answering the query?",
            #     }
            # ])

            # segments = None
            # answer = None
            # # Identify potential keywords that relate to the query.
            # keywords = list_words(self.model, [
            #     {
            #         "role": "user",
            #         "content": f"Here is a preview of the document(s) separated by <div>s: {doc_preview}\nThis is the query I want to answer: {final_query}\nIf the document may be able to answer the query, please provide keywords separated by a comma to search the document(s) for, verbatim, in a document to answer the following query. Otherwise, give no response.\nPlease note each comma-separated keyword will be used to retrieve sentences from the document, so find keywords that will find sentences from the document that may be relevant to answer the question.",
            #     }
            # ])
            
            # # Find all in-context instances of the keyword in the document.
            # relevant_segments = []
            # for keyword in keywords:
            #     keyword = keyword.strip().lower()
            #     if keyword:
            #         for doc_id in X.document_ids:
            #             if keyword in docs[doc_id].lower():
            #                 relevant_segments.extend(self._extract_keyword_context(document=docs[doc_id], keyword=keyword))
            # relevant_segments = self._remove_near_duplicates(relevant_segments)
            # if len(relevant_segments) < 1:
            #     document_relevant = False
            # else:
            #     document_relevant = True
            #     segments = relevant_segments
            #     excerpt_context = "\n".join(
            #         [f"<div>{segment}</div>" for segment in relevant_segments]
            #     )
            #     history = [
            #         {
            #             "role": "system",
            #             "content": f"You are a helpful assistant. Answer with the single most relevant snippet from the document(s) verbatim and nothing else. Key Excerpts: {excerpt_context}",
            #         }
            #     ] + X.conversation

            #     outputs = self.model(
            #         history,
            #         max_new_tokens=256,
            #     )
            #     answer = outputs[0]["generated_text"][-1]["content"]
            #     for doc_id in X.document_ids:
            #         if answer in docs[doc_id]:
            #             segments = [answer]

            return Label(
                document_relevant=document_relevant,
                segments=segments,
                answer=answer,
                time_taken=time.time() - start,
            )


        # # Assume if len(conversation) == 1, new conversation start. Reset entities & relations.
        # if len(X.conversation) == 1:
        #     self.e2i = {}
        #     self.r2i = {}
        #     self.i2e = {}
        #     self.i2r = {}
        #     self.edge_to_excerpt = defaultdict(list)

        # start = time.time()
        # final_query = X.conversation[-1]["content"]
        # relevance_eval = [
        #     {
        #         "role": "user",
        #         "content": f"Is this a question: \"{final_query}\"",
        #     }
        # ]
        # document_relevant = affirmative_resp(self.model, relevance_eval)
        # segments = None
        # answer = None
        # if not document_relevant:
        #     return Label(
        #         document_relevant=document_relevant,
        #         segments=segments,
        #         answer=answer,
        #         time_taken=time.time() - start,
        #     )
        # if not self.use_dialogue_kg:
        #     if self.summary_trees is None:
        #         doc_context = "\n".join(
        #             [f"<div>{docs[doc_id]}</div>" for doc_id in X.document_ids]
        #         )
        #         history = [
        #             {
        #                 "role": "system",
        #                 "content": f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Answer each question by quoting from the document. Documents: {doc_context}",
        #             }
        #         ] + X.conversation

        #         relevance_convo = history + [
        #             {
        #                 "role": "user",
        #                 "content": "Are the provided document(s) relevant to answer the above question?",
        #             }
        #         ]
        #         document_relevant = affirmative_resp(self.model, relevance_convo)
        #         if document_relevant:
        #             outputs = self.model(
        #                 history,
        #                 max_new_tokens=256,
        #             )
        #             answer = outputs[0]["generated_text"][-1]["content"]
        #     else:
        #         # TODO
        #         raise Exception(f"Combination not supported:\n\tSummary Tree: {self.summary_trees is not None}\n\tDialogue KG: {self.use_dialogue_kg}")
        # else:  # self.use_dialogue_kg is True
        #     relevant_segments = []

        #     if len(self.e2i) > 1:
        #         history = [
        #             {
        #                 "role": "user",
        #                 "content": (
        #                     "Given the following list of entities and relations, please translate the following query in "
        #                     "subject_entity|relation|tail_entity form. Do not include anything else.\n"
        #                     f"Query: {final_query}\n"
        #                     "Entities: " + "\n".join(list(self.e2i.keys())) + "\n"
        #                     "Relations: " + "\n".join(list(self.r2i.keys()))
        #                 )
        #             }
        #         ]

        #         outputs = self.model(history, max_new_tokens=64)
        #         response = outputs[0]["generated_text"][-1]["content"].lower()
        #         if response in self.edge_to_excerpt:
        #             relevant_segments = self.edge_to_excerpt[response]

        #     # If no relevant document segments, look at document (tree)
        #     if len(relevant_segments) < 1 and self.summary_trees:
        #         current_nodes = [self.summary_trees[doc_id].root for doc_id in X.document_ids]
        #         relevant_nodes = []
        #         # BFS
        #         while current_nodes:
        #             current_node = current_nodes.pop(0)
        #             current_nodes.extend(current_node.children)
        #             segment_relevance_eval = [
        #                 {
        #                     "role": "user",
        #                     "content": f"Here is a summary of a text: {current_node.data}. Is this text useful to answer the following question: \"{final_query}\"",
        #                 }
        #             ]
        #             segment_relevant = affirmative_resp(self.model, segment_relevance_eval)
        #             if segment_relevant:
        #                 if not current_node.children:
        #                     relevant_nodes.append(current_node)
        #                 else:
        #                     current_nodes.append(current_node)

        #     # If no relevant document segments, do a keyword search
        #     if len(relevant_segments) < 1:
        #         # Identify potential keywords that relate to the query.
        #         keywords = ["".join([c for c in keyword if c.isalpha() or c.isspace()]) for keyword in list_words(self.model, [
        #             {
        #                 "role": "user",
        #                 "content": f"Please provide keywords separated by a comma to search the document for, verbatim, in a document to answer the following query. Please note each comma-separated keyword will be used to retrieve sentences from the document, so find keywords that will find sentences from the document that may be relevant to answer the question. \nQuery: {final_query}",
        #             }
        #         ])]
        #         print(keywords)
        #         # Find all in-context instances of the keyword in the document.
        #         for keyword in keywords:
        #             for doc_id in X.document_ids:
        #                 preprocessed_document = "".join([c for c in docs[doc_id].lower() if c.isalpha() or c.isspace()])
        #                 relevant_segments.extend(self._extract_keyword_context(document=preprocessed_document, keyword=keyword))

        #     # Answer based on context
        #     if relevant_segments:
        #         document_relevant = True
        #         segments = relevant_segments
        #         history = [
        #             {
        #                 "role": "system",
        #                 "content": "Use the following documents to answer the question:\n" + "\n".join([f"<div>{segment}</div>" for segment in relevant_segments]),
        #             },
        #         ] + X.conversation
        #         answer = self.model(history, max_new_tokens=256)[0]["generated_text"][-1]["content"]

        #         # Add edge associated with each relevant segment
        #         edges = segments_to_edges(self.model, relevant_segments, self.e2i, self.r2i, self.i2e, self.i2r, self.edge_to_excerpt)

        # return Label(
        #     document_relevant=document_relevant,
        #     segments=segments,
        #     answer=answer,
        #     time_taken=time.time() - start,
        # )

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
