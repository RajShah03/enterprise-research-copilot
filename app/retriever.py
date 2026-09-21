from flashrank import (
    Ranker,
    RerankRequest,
)

from app.ingestion import tokenize_for_bm25


class HybridRetriever:

    def __init__(
        self,
        index,
        bm25,
        leaf_nodes,
    ):
        self.index = index
        self.bm25 = bm25
        self.leaf_nodes = leaf_nodes

        # ----------------------------------------------------
        # Vector retriever
        # ----------------------------------------------------

        self.vector_retriever = (
            index.as_retriever(
                similarity_top_k=15
            )
        )

        # ----------------------------------------------------
        # FlashRank
        # ----------------------------------------------------

        print(
            "⚡ Loading FlashRank reranker..."
        )

        self.flashrank = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2"
        )

        print(
            "✅ FlashRank reranker ready."
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalizes text for exact duplicate detection.
        """

        if not text:
            return ""

        return " ".join(
            text.lower().split()
        )

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

    def _describe_node(
        self,
        node,
    ):
        metadata = getattr(
            node,
            "metadata",
            {},
        )

        file_name = (
            metadata.get("file_name")
            or metadata.get("filename")
            or "Unknown"
        )

        page = (
            metadata.get("page_label")
            or metadata.get("page")
            or "?"
        )

        text = (
            node.get_content()
            .replace("\n", " ")
            .strip()
        )

        preview = text[:220]

        if len(text) > 220:
            preview += "..."

        return (
            f"page={page} | "
            f"file={file_name} | "
            f"text={preview}"
        )

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_n: int = 6,
    ):
        """
        Hybrid retrieval pipeline:

        1. Vector retrieval
        2. BM25 retrieval
        3. Candidate union + deduplication
        4. FlashRank reranking
        5. Final top-N passages
        """

        if not query.strip():
            return []

        print(
            "\n" + "=" * 80
        )

        print(
            "🔍 HYBRID RETRIEVAL"
        )

        print(
            f"Query: {query}"
        )

        print(
            "=" * 80
        )

        # ====================================================
        # 1. Vector Retrieval
        # ====================================================

        print(
            "\n🔎 Running vector search..."
        )

        vector_results = (
            self.vector_retriever.retrieve(
                query
            )
        )

        print(
            f"✅ Vector results: "
            f"{len(vector_results)}"
        )

        # ====================================================
        # 2. BM25 Retrieval
        # ====================================================

        print(
            "\n🔎 Running BM25 search..."
        )

        tokenized_query = tokenize_for_bm25(
            query
        )

        bm25_scores = (
            self.bm25.get_scores(
                tokenized_query
            )
        )

        top_bm25_indices = sorted(
            range(
                len(bm25_scores)
            ),
            key=lambda index: (
                bm25_scores[index]
            ),
            reverse=True,
        )[:10]

        print(
            f"✅ BM25 results: "
            f"{len(top_bm25_indices)}"
        )

        # ====================================================
        # 3. Combine + Deduplicate
        # ====================================================

        candidate_nodes = {}
        seen_texts = set()

        def add_node(node):
            text = (
                node.get_content()
                .strip()
            )

            if not text:
                return

            normalized_text = (
                self._normalize_text(
                    text
                )
            )

            if normalized_text in seen_texts:
                return

            seen_texts.add(
                normalized_text
            )

            candidate_nodes[
                node.node_id
            ] = node

        # Vector candidates.
        for result in vector_results:
            add_node(
                result.node
            )

        # BM25 candidates.
        for index in top_bm25_indices:
            add_node(
                self.leaf_nodes[index]
            )

        if not candidate_nodes:

            print(
                "⚠️ No retrieval candidates found."
            )

            return []

        print(
            f"\n📦 Unique candidate pool: "
            f"{len(candidate_nodes)}"
        )

        # ====================================================
        # 4. FlashRank
        # ====================================================

        passages = [
            {
                "id": node_id,
                "text": node.get_content().strip(),
            }
            for node_id, node
            in candidate_nodes.items()
        ]

        print(
            "\n⚡ Reranking with FlashRank..."
        )

        rerank_request = RerankRequest(
            query=query,
            passages=passages,
        )

        reranked_results = (
            self.flashrank.rerank(
                rerank_request
            )
        )

        print(
            f"✅ FlashRank returned: "
            f"{len(reranked_results)} results"
        )

        # ====================================================
        # 5. Final Results
        # ====================================================

        node_lookup = {
            node.node_id: node
            for node
            in candidate_nodes.values()
        }

        final_results = []

        for result in reranked_results[:top_n]:

            node_id = result.get(
                "id"
            )

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            score = self._safe_float(
                result.get(
                    "score",
                    0.0,
                )
            )

            final_results.append(
                {
                    "id": node_id,
                    "text": result.get(
                        "text",
                        node.get_content(),
                    ),
                    "score": score,
                    "metadata": dict(
                        node.metadata
                    ),
                }
            )

        print(
            "\n" + "=" * 80
        )

        print(
            f"✅ Returning "
            f"{len(final_results)} "
            f"reranked passages."
        )

        print(
            "=" * 80
        )

        return final_results