from typing import Any, List, Dict

def affirmative_resp(model: Any, history: List[Dict[str, str]]) -> bool:
    history[-1]["content"] += " Answer \"YES\" or \"NO\" only."

    outputs = model(history, max_new_tokens=10)
    response = outputs[0]["generated_text"][-1]["content"].upper()

    return "NO" not in response and "YES" in response

def list_words(model: Any, history: List[Dict[str, str]]) -> bool:
    history[-1]["content"] += " Respond with a comma-seperated list only. Do not include anything else in your response."

    outputs = model(history, max_new_tokens=256)
    response = outputs[0]["generated_text"][-1]["content"].lower().split(",")

    return [v.strip() for v in response]

def resp_to_kg(output: str, entities_to_id: Dict[str, int], relations_to_id: Dict[str, int], id_to_entities: Dict[str, int], id_to_relations: Dict[str, int], update_mapping: bool) -> List[List[str]]:
    # Filter out incorrectly formatted.
    response = [[v.strip() for v in line.lower().split("|")] for line in output.lower().split("\n") if line.count("|") == 2]

    if update_mapping:    
        resp_entities = [v[0] for v in response] + [v[2] for v in response]
        resp_rels = [v[1] for v in response]
        
        # Update entities and relations dict with newly created ones.
        for e in resp_entities:
            if e not in entities_to_id:
                entities_to_id[e] = len(entities_to_id)
                id_to_entities[entities_to_id[e]] = len(e)
        
        for r in resp_rels:
            if r not in relations_to_id:
                relations_to_id[r] = len(relations_to_id)
                id_to_relations[relations_to_id[r]] = len(r)

    return [[entities_to_id[edge[0]], relations_to_id[edge[1]], entities_to_id[edge[2]]] for edge in response if edge[0] in entities_to_id and edge[2] in entities_to_id and edge[1] in relations_to_id]

def segments_to_edges(model: Any, segments: List[str], entities_to_id: Dict[str, int], relations_to_id: Dict[str, int], id_to_entities: Dict[str, int], id_to_relations: Dict[str, int], edge_to_excerpt: Dict[str, List[str]]) -> List[List[str]]:
    # Convert to edges.
    excerpts = [f"<div>{doc}</div>" for doc in segments]
    all_edges = []
    for excerpt in excerpts:
        entity_lst = "\n".join(entities_to_id.keys())
        rel_lst = "\n".join(relations_to_id.keys())
        prompt = f"""
{excerpt}
Convert the document excerpt in the div into edges of the form entity1|relation|entity2 separated with newline.
For example, the sentence "Alice had a conversation with Bob and Carol." should be converted to:
Alice|talk_with|Bob
Alice|talk_with|Carol
Whenever possible, use the following entities and relations.
Entities:
{entity_lst}
Relations:
{rel_lst}
        """.strip()
        outputs = model([
            {
                "role": "user",
                "content": prompt,
            }
        ], max_new_tokens=256)
        edges = resp_to_kg(
            outputs[0]["generated_text"][-1]["content"],
            entities_to_id,
            relations_to_id,
            id_to_entities,
            id_to_relations,
            update_mapping=True
        )
        all_edges.extend(edges)
        for edge in edges:
            key = f"{id_to_entities[edge[0]]}|{id_to_relations[edge[1]]}|{id_to_entities[edge[2]]}"
            edge_to_excerpt[key].append(excerpt)
    return all_edges