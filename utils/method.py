import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

import spacy
import torch
from tqdm import tqdm
from transformers import pipeline

from .graph.summary_tree import SummaryTree
from .response import affirmative_resp, list_words, segments_to_edges
from .structures import *

nlp = spacy.load("en_core_web_lg")


# TODO: Clean up to not need e2i, i2e, r2i, i2r
class ConvRef:
    def __init__(
        self,
        model: str,
        llm_only: bool,
        strict: bool,
        use_gt_segments: bool = False,  # Flag for Stage 1 ablation
        use_gt_doc_relevancy: bool = False,  # Flag for Stage 2 ablation
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

        # Ablation flags
        self.use_gt_segments = use_gt_segments
        self.use_gt_doc_relevancy = use_gt_doc_relevancy

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
        sentences = re.split(r"(?<=[.!?]) +", document)

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

    def _run_llm_only_approach(
        self, X: Sample, docs: Dict[str, str], start: float
    ) -> Label:
        history = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Documents: {doc_context}",
            }
        ] + X.conversation
        document_relevant = affirmative_resp(
            self.model,
            history
            + [
                {
                    "role": "user",
                    "content": f"Are the document(s) relevant for answering the query?",
                }
            ],
        )
        segments = None
        answer = None
        if document_relevant:
            history[0]["content"] = (
                f"You are a helpful assistant. If needed, refer to the following provided document(s) to answer questions. Answer with the single most relevant snippet from the document(s) verbatim and nothing else. Documents: {doc_context}",
            )
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

    def _get_relevant_segments(
        self, X: Sample, docs: Dict[str, str], doc_context: str, final_query: str
    ) -> List[str]:
        """Helper function to get the key excerpts (relevant segments) from the passage. Stage 1 of the Ours approach."""
        # Identify potential keywords that relate to the query.
        keywords = list_words(
            self.model,
            [
                {
                    "role": "user",
                    "content": f"Here are the document(s) separated by <div>s: {doc_context}\nThis is the query I want to answer: {final_query}\nIf the document may be able to answer the query, please provide keywords separated by a comma to search the document(s) for, verbatim, in a document to answer the following query. Otherwise, give no response.\nPlease note each comma-separated keyword will be used to retrieve sentences from the document, so find keywords that will find sentences from the document that may be relevant to answer the question.",
                }
            ],
        ) + [ent.text for ent in nlp(final_query).ents]
        keywords = list(set([v.lower() for v in keywords]))
        print("KEYWORDS", keywords)

        # Find all in-context instances of the keyword in the document.
        relevant_segments = []
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if keyword:
                for doc_id in X.document_ids:
                    if keyword in docs[doc_id].lower():
                        relevant_segments.extend(
                            self._extract_keyword_context(
                                document=docs[doc_id], keyword=keyword
                            )
                        )
        relevant_segments = self._remove_near_duplicates(relevant_segments)
        return relevant_segments

    def _determine_document_relevancy(
        self, X: Sample, relevant_segments: List[str]
    ) -> bool:
        """Helper function to determine if the document is relevant based on the key excerpts. Stage 2 of the Ours approach."""
        excerpt_context = "\n".join(
            [f"<div>{segment}</div>" for segment in relevant_segments]
        )

        if (
            self.strict
        ):  # Ours_strict approach (ask the LLM if the document is relevant)
            document_relevant = affirmative_resp(
                self.model,
                [X.conversation[-1]]
                + [
                    {
                        "role": "user",
                        "content": f"Are the following excerpts relevant for answering the query? Excerpts: {excerpt_context}",
                    }
                ],
            )
        else:  # Ours_lax approach (assume the document is relevant)
            document_relevant = True

        return document_relevant

    def _generate_response(
        self, X: Sample, relevant_segments: List[str], docs: Dict[str, str]
    ) -> str:
        """Helper function to generate the response. Stage 3 of the Ours approach."""

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

        return answer, segments

    def _run_ours_approach(
        self, X: Sample, docs: Dict[str, str], start: float, doc_context: str, Y: Label
    ) -> Label:
        """Main function to run the Ours approach."""
        # Stage 1: Key Excerpts Selection
        if self.use_gt_segments:
            relevant_segments = Y.segments
        else:
            final_query = X.conversation[-1]["content"]
            relevant_segments = self._get_relevant_segments(
                X, docs, doc_context, final_query
            )

        # Stage 2: Relevancy Check (identify if the document is relevant)
        if len(relevant_segments) == 0:
            document_relevant = (
                False  # always set it to False when relevant_segments is empty
            )
        elif self.use_gt_doc_relevancy:
            document_relevant = Y.document_relevant
        else:  # otherwise, determine relevancy based on the key excerpts (Our Approach)
            document_relevant = self._determine_document_relevancy(X, relevant_segments)

        # Stage 3: Response Generation
        answer, segments = None, None
        if document_relevant:
            answer, segments = self._generate_response(X, relevant_segments, docs)

        return Label(
            document_relevant=document_relevant,
            segments=segments,
            answer=answer,
            time_taken=time.time() - start,
        )

    def __call__(self, X: Sample, docs: Dict[str, str], Y: Label = None) -> Label:
        """
        Call function to generate a response given a Sample and a Dict of doc ids to text.

        Args:
            X (Sample): The input sample
            docs (Dict[str, str]): A dictionary of document ids to their corresponding text
            Y (Label, optional): The ground truth label. Defaults to None.

        Returns:
            Label: The generated response
        """
        start = time.time()
        if self.llm_only:
            return self._run_llm_only_approach(X, docs, start)
        else:
            doc_context = "\n".join(
                [f"<div>{docs[doc_id]}</div>" for doc_id in X.document_ids]
            )
            return self._run_ours_approach(X, docs, start, doc_context, Y)

    def load_summary_trees(self, summary_trees_fp: str) -> None:
        self.summary_trees = {
            k: SummaryTree.from_dict(v)
            for k, v in json.load(open(summary_trees_fp, "r")).items()
        }

    def generate_summary_trees(
        self, summary_trees_fp: str, docs: Dict[str, str], emb_model: Any
    ) -> None:
        summary_trees = {}
        for doc_id, doc in tqdm(docs.items()):
            summary_trees[doc_id] = SummaryTree(None)
            summary_trees[doc_id].generate_from(doc, self.model, emb_model)

            with open(summary_trees_fp, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in summary_trees.items()}, f, indent=4
                )
        self.summary_trees = summary_trees
