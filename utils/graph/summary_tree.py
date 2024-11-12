from typing import Any
from kneed import KneeLocator
from sklearn.cluster import KMeans

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_level(self):
        level = 0
        current_node = self
        while current_node.parent:
            level += 1
            current_node = current_node.parent
        return level
    
    def generate_children_summary(self, model):
        docs = "\n".join([f"<div>{child.data}</div>" for child in self.children])
        prompt = [{
            "role": "user",
            "content": f"Summarize the content of the following documents in 2-3 sentences:\n{docs}",
        }]
        outputs = model(prompt, max_new_tokens=256)
        self.data = outputs[0]["generated_text"][-1]["content"]

    def print_tree(self):
        spaces = ' ' * self.get_level() * 4
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()
    
    def to_dict(self):
        """Convert the node and its children to a dictionary."""
        return {
            "data": self.data,
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a TreeNode from a dictionary."""
        node = cls(data["data"])
        for child_data in data.get("children", []):
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)
        return node



# TODO: @Eunice Implemented for MultiWOZ which is just Q&A so we can split by linebreak. However, the other datasets may have more complicated structure and we may not be able to do so.
class SummaryTree:
    def __init__(self, root):
        self.root = root

    def generate_from(self, doc: str, lang_model: Any, emb_model: Any, max_nodes_per_level=10):
        chunks = list(set(doc.split("\n")))
        embeddings = emb_model.encode(chunks, task="separation")
        level_nodes = [TreeNode(chunk) for chunk in chunks]
        
        while len(level_nodes) > max_nodes_per_level:
            # Cluster leaf nodes until # of nodes per level <= max_nodes_per_level
            cluster_range = range(2, max_nodes_per_level)
            sse = []  # Sum of squared distances
            models = []
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(embeddings)
                sse.append(kmeans.inertia_)
                models.append(kmeans)

            # Find the elbow point
            kneedle = KneeLocator(cluster_range, sse, curve="convex", direction="decreasing")
            optimal_k = kneedle.elbow
            if not kneedle.elbow:
                optimal_k = max_nodes_per_level-1

            clusters = models[optimal_k - cluster_range.start].labels_

            next_level = [TreeNode("") for cluster in range(optimal_k)]
            for level_node, cluster in zip(level_nodes, clusters):
                next_level[cluster].add_child(level_node)
            for node in next_level:
                node.generate_children_summary(lang_model)
            level_nodes = next_level

        self.root = TreeNode("")
        for node in level_nodes:
            self.root.add_child(level_node)
        self.root.generate_children_summary(lang_model)

    def print_tree(self):
        if self.root:
            self.root.print_tree()
    
    def to_dict(self):
        """Convert the entire Tree to a dictionary."""
        if self.root:
            return self.root.to_dict()
        return {}

    @classmethod
    def from_dict(cls, json_data):
        """Build the Tree object from a dictionary."""
        root = TreeNode.from_dict(json_data)
        return cls(root)