# vector similarity engine from scratch
# concepts: dot product, magnitude, cosine similarity, ranking

import math

class SimilarityEngine:
    def __init__(self, items):
        self.items = items

    def dot_product(self, v1, v2):
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def magnitude(self, v):
        total = 0
        for x in v:
            total += x * x
        return math.sqrt(total)

    def cosine_similarity(self, v1, v2):
        mag1 = self.magnitude(v1)
        mag2 = self.magnitude(v2)

        if mag1 == 0 or mag2 == 0:
            return 0

        return self.dot_product(v1, v2) / (mag1 * mag2)

    def query(self, vector, top_k=3):
        scores = {}

        for name, item_vector in self.items.items():
            sim = self.cosine_similarity(vector, item_vector)
            scores[name] = sim

        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_items[:top_k]


if __name__ == "__main__":
    items = {
        "apple": [1, 0, 0],
        "banana": [0, 1, 0],
        "orange": [0, 0, 1],
        "pineapple": [0.9, 0.1, 0],
        "kiwi": [0.2, 0.8, 0]
    }

    engine = SimilarityEngine(items)

    query_vector = [1, 0, 0]

    results = engine.query(query_vector, top_k=3)

    print("Top 3 most similar items:")
    for name, score in results:
        print(f"{name}: {score:.3f}")
