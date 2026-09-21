import re

from app.config import llm


# ============================================================
# Context Evaluation
# ============================================================

def evaluate_relevance(
    query: str,
    context: str,
) -> str:
    """
    Determines whether the retrieved context is sufficient
    to answer the user's question.

    Returns:
        COMPLETE
        PARTIAL
        NONE
    """

    if not context.strip():
        return "NONE"

    prompt = f"""
You are a strict evidence evaluator for a financial RAG system.

User Query:
{query}

Retrieved Evidence:
{context}

Classify the evidence into exactly one category.

COMPLETE
- All major requested facts are supported.
- All important requirements are satisfied.
- Multiple sources may be combined.
- Comparisons require evidence for the metrics being compared.
- A requested ranking must be explicitly supported.
- A requested future or causal relationship must be explicitly supported.

PARTIAL
- Useful evidence exists, but one or more requested facts,
  metrics, comparisons, rankings, timeframes, or relationships
  are missing or unsupported.

NONE
- No useful evidence for the question is present.

Rules:
- Use only the retrieved evidence.
- Do not use outside knowledge.
- Do not infer missing facts or figures.
- Do not infer rankings from source order or retrieval score.
- Do not treat one financial metric as another.
- Historical evidence must not automatically become a future claim.
- Multiple sources may be combined.
- Be conservative.

Examples:
- Revenue growth is provided but operating-income growth is missing
  → PARTIAL.
- Diluted EPS growth is provided but net-income growth is missing
  → PARTIAL.
- Relevant risks are identified but the requested ranking is absent
  → PARTIAL.
- No relevant evidence is present
  → NONE.

Return ONLY:
COMPLETE
PARTIAL
NONE
"""

    response = (
        llm.complete(prompt)
        .text
        .strip()
        .upper()
    )

    if response in {
        "COMPLETE",
        "PARTIAL",
        "NONE",
    }:
        return response

    print(
        f"⚠️ Unexpected context evaluation response: {response}"
    )

    return "NONE"


# ============================================================
# Query Rewriting
# ============================================================

def rewrite_query(
    query: str,
) -> str:
    """
    Rewrites a failed query for corrective retrieval.

    This function is only called when the initial context
    is classified as NONE.
    """

    prompt = f"""
You are a search-query optimizer for a financial RAG system.

Rewrite the following question into a retrieval-focused query.

Rules:
- Preserve the original intent.
- Preserve all requested metrics.
- Preserve company, segment, and business-unit names.
- Preserve fiscal years and dates.
- Preserve comparison requirements.
- Preserve ranking terms such as most important, top, or significant.
- Include both amount and growth concepts when both are requested.
- Expand useful financial terminology where helpful.
- Do not invent facts or numbers.
- Do not answer the question.
- Return ONLY the rewritten query.
- Do not use quotation marks.

Original Query:
{query}

Rewritten Query:
"""

    rewritten = (
        llm.complete(prompt)
        .text
        .strip()
    )

    return rewritten or query


# ============================================================
# Citation Normalization
# ============================================================

def normalize_citation_format(
    answer: str,
) -> str:
    """
    Normalizes common citation variations into:

        [SOURCE 1]
        [SOURCE 2]
    """

    if not answer:
        return answer

    # 【SOURCE 1】 / 【Source 1】
    answer = re.sub(
        r"【\s*source\s+(\d+)\s*】",
        r"[SOURCE \1]",
        answer,
        flags=re.IGNORECASE,
    )

    # [Source 1] / [source 1]
    answer = re.sub(
        r"\[\s*source\s+(\d+)\s*\]",
        r"[SOURCE \1]",
        answer,
        flags=re.IGNORECASE,
    )

    # [ SOURCE 1 ] / [SOURCE 1]
    answer = re.sub(
        r"\[\s*SOURCE\s+(\d+)\s*\]",
        r"[SOURCE \1]",
        answer,
        flags=re.IGNORECASE,
    )

    return answer


# ============================================================
# Answer Generation
# ============================================================

def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generates a grounded answer using only retrieved evidence.
    """

    if not context.strip():
        return (
            "I could not find sufficient information in the "
            "provided documents to answer this question."
        )

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the retrieved evidence.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts, figures, percentages, calculations,
   rankings, or causal relationships.

3. Preserve the exact financial figures, units, and precision
   from the evidence.

4. Do not round or normalize financial figures unless the source
   itself does so.

5. Multiple sources may be combined when they provide
   complementary evidence.

6. If only part of the question is supported:
   - answer the supported part;
   - clearly identify what is missing;
   - do not pretend the answer is complete.

7. Do not substitute one financial metric for another.
   Revenue, operating income, net income, diluted EPS,
   gross margin, and cash flow are distinct metrics.

8. If diluted EPS growth is provided but net-income growth is not,
   explicitly say that net-income growth is not provided.

9. Do not infer segment revenue from company-level revenue.

10. Do not infer missing facts or figures.

11. Do not turn a historical observation into a future or causal
    claim unless the evidence explicitly supports that relationship.

12. Do not call factors "most important", "top", "primary",
    or "most significant" unless the evidence explicitly ranks them.

13. If the user requests a ranking but the evidence does not
    establish one, explicitly state that the documents do not
    provide a definitive ranking.

14. Do not make a claim stronger than the source supports.

15. Before calling the answer complete, verify that every explicit
    requirement in the question is supported.

Citation rules:

16. Use only these citation formats:
    [SOURCE 1]
    [SOURCE 2]
    [SOURCE 3]

17. Use only source numbers that exist in the retrieved evidence.

18. Place citations immediately after the claim they support.

19. Do not create citations for unsupported claims.

20. Avoid unnecessary duplicate citations.

Style:

21. Keep the answer concise but informative.

22. Prefer direct, evidence-faithful wording.

Retrieved Evidence:
{context}

User Query:
{query}

Final Answer:
"""

    answer = (
        llm.complete(prompt)
        .text
        .strip()
    )

    if not answer:
        return (
            "I could not generate an answer from the "
            "available document context."
        )

    return normalize_citation_format(
        answer
    )


# ============================================================
# Corrective RAG Agent
# ============================================================

class CorrectiveRAGAgent:

    def __init__(
        self,
        hybrid_retriever,
    ):
        self.retriever = hybrid_retriever

    # ========================================================
    # Build Context
    # ========================================================

    def _build_context(
        self,
        results,
    ):
        """
        Converts retrieved passages into:

        1. LLM context
        2. API source metadata

        Deduplication is handled by the retriever, so this
        function only formats the final retrieved results.
        """

        if not results:
            return "", []

        context_parts = []
        sources = []

        for result in results:

            text = (
                result.get(
                    "text",
                    "",
                )
                .strip()
            )

            if not text:
                continue

            metadata = result.get(
                "metadata",
                {},
            )

            score = result.get(
                "score"
            )

            source_id = len(
                sources
            ) + 1

            file_name = (
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("file_path")
                or "Unknown"
            )

            page = (
                metadata.get("page_label")
                or metadata.get("page")
            )

            source_header = (
                f"[SOURCE {source_id}]\n"
                f"Document: {file_name}\n"
            )

            if page is not None:
                source_header += (
                    f"Page: {page}\n"
                )

            source_header += (
                f"\n{text}"
            )

            context_parts.append(
                source_header
            )

            sources.append(
                {
                    "source_id": source_id,
                    "document": file_name,
                    "page": page,
                    "score": (
                        float(score)
                        if score is not None
                        else None
                    ),
                    "node_id": result.get(
                        "id"
                    ),
                    "text": text,
                }
            )

        context = (
            "\n\n---\n\n".join(
                context_parts
            )
        )

        return context, sources

    # ========================================================
    # Source Metadata
    # ========================================================

    @staticmethod
    def _build_source_evaluations(
        results,
    ):
        """
        Records which passages were selected by the
        hybrid retrieval + FlashRank pipeline.

        This is NOT an individual LLM relevance judgment.
        """

        evaluations = []

        for position, result in enumerate(
            results,
            start=1,
        ):

            evaluations.append(
                {
                    "retrieved_rank": position,
                    "node_id": result.get(
                        "id"
                    ),
                    "selection_method": (
                        "hybrid_retrieval_flashrank"
                    ),
                }
            )

        return evaluations

    # ========================================================
    # Main Pipeline
    # ========================================================

    def run(
        self,
        query: str,
    ):
        """
        Main Corrective RAG pipeline:

        1. Hybrid retrieval + FlashRank
        2. Context evaluation
        3. Corrective retrieval if NONE
        4. Grounded answer generation
        """

        query = query.strip()

        if not query:

            return {
                "status": "not_found",
                "answer": "Please provide a question.",
                "sources": [],
                "rewritten_query": None,
                "corrective_rag": False,
                "source_evaluations": [],
            }

        print(
            f"\n🔍 Searching context for query: "
            f"'{query}'"
        )

        # ====================================================
        # STEP 1 — Initial Retrieval
        # ====================================================

        results = self.retriever.search(
            query,
            top_n=6,
        )

        print(
            f"📚 Retrieved final passages: "
            f"{len(results)}"
        )

        # ====================================================
        # STEP 2 — Build Context
        # ====================================================

        context, sources = (
            self._build_context(
                results
            )
        )

        source_evaluations = (
            self._build_source_evaluations(
                results
            )
        )

        rewritten_query = None
        corrective_rag_used = False

        # ====================================================
        # STEP 3 — Context Evaluation
        # ====================================================

        print(
            "🧠 Evaluating retrieved context..."
        )

        coverage = evaluate_relevance(
            query,
            context,
        )

        print(
            f"🧠 Evidence coverage: {coverage}"
        )

        # ====================================================
        # STEP 4 — Corrective RAG
        # ====================================================

        if coverage == "NONE":

            corrective_rag_used = True

            print(
                "⚠️ No useful evidence found."
            )

            # ------------------------------------------------
            # Rewrite query
            # ------------------------------------------------

            rewritten_query = rewrite_query(
                query
            )

            print(
                f"🔄 Rewritten Query: "
                f"'{rewritten_query}'"
            )

            # ------------------------------------------------
            # Corrective retrieval
            # ------------------------------------------------

            results = self.retriever.search(
                rewritten_query,
                top_n=6,
            )

            print(
                f"📚 Corrective retrieval returned: "
                f"{len(results)} passages"
            )

            # ------------------------------------------------
            # Rebuild context
            # ------------------------------------------------

            context, sources = (
                self._build_context(
                    results
                )
            )

            source_evaluations = (
                self._build_source_evaluations(
                    results
                )
            )

            # ------------------------------------------------
            # Evaluate corrected context
            # ------------------------------------------------

            print(
                "🧠 Evaluating corrected context..."
            )

            coverage = evaluate_relevance(
                query,
                context,
            )

            print(
                f"🧠 Corrected evidence coverage: "
                f"{coverage}"
            )

            if coverage == "NONE":

                print(
                    "❌ Corrected retrieval also failed."
                )

                return {
                    "status": "not_found",
                    "answer": (
                        "I could not find relevant "
                        "information in the provided "
                        "documents."
                    ),
                    "sources": sources,
                    "rewritten_query": rewritten_query,
                    "corrective_rag": True,
                    "source_evaluations": (
                        source_evaluations
                    ),
                }

        else:

            print(
                f"✅ Retrieved context classified as "
                f"{coverage.lower()}."
            )

        # ====================================================
        # STEP 5 — Answer Generation
        # ====================================================

        print(
            "🤖 Generating answer via Groq..."
        )

        answer = generate_answer(
            query,
            context,
        )

        # ====================================================
        # STEP 6 — Final Response
        # ====================================================

        return {
            "status": coverage.lower(),
            "answer": answer,
            "sources": sources,
            "rewritten_query": rewritten_query,
            "corrective_rag": corrective_rag_used,
            "source_evaluations": (
                source_evaluations
            ),
        }
