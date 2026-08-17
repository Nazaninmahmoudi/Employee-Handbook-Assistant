class HybridRRF:

    def __init__(
        self,
        retrievers,
        weights,
        top_k=5
    ):
        self.retrievers = retrievers
        self.weights = weights
        self.top_k = top_k


    def invoke(self, query):

        results = []

        for retriever, weight in zip(
            self.retrievers,
            self.weights
        ):

            docs = retriever.invoke(query)

            for rank, doc in enumerate(docs):

                score = weight / (rank + 1)

                results.append(
                    (
                        doc,
                        score
                    )
                )


        results.sort(
            key=lambda x:x[1],
            reverse=True
        )


        return [
            doc
            for doc,score in results[:self.top_k]
        ]