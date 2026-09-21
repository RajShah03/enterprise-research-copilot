import re

from app.config import llm


# ============================================================
# Source-Level Relevance Evaluation
# ============================================================

def evaluate_source_relevance(
    query: str,
    source_text: str,
) -> bool:
    """
    Optional debugging utility.

    This function is intentionally NOT called for every source
    during the normal production pipeline because doing so creates
    many unnecessary LLM calls.

    The hybrid retriever + FlashRank already performs retrieval
    and reranking.
    """

    if not source_text.strip():
        return False

    prompt = f"""
You are evaluating one retrieved passage for a financial RAG system.

User Query:
{query}

Retrieved Source:
{source_text}

Determine whether this source contains useful evidence for ANY
meaningful part of the user's query.

Rules:
- The source does NOT need to answer the entire question.
- The source may contain one requested metric, date, percentage,
  comparison value, risk, explanation, or other relevant fact.
- Multiple sources may need to be combined.
- Reject only sources that are clearly unrelated or contain no useful
  evidence for the query.
- Do not use outside knowledge.
- Do not explain your reasoning.

Return exactly:
YES
or
NO
"""

    response = (
        llm.complete(prompt)
        .text
        .strip()
        .upper()
    )

    if response == "YES":
        return True

    if response == "NO":
        return False

    print(
        f"⚠️ Unexpected source relevance response: "
        f"{response}"
    )

    return False


# ============================================================
# Context-Level Evidence Coverage Evaluation
# ============================================================

def evaluate_relevance(
    query: str,
    context: str,
) -> str:
    """
    Evaluates whether the retrieved context satisfies the
    explicit requirements of the user's question.

    Returns:
        COMPLETE
        PARTIAL
        NONE
    """

    if not context.strip():
        return "NONE"

    prompt = f"""
You are a strict evidence-coverage evaluator for a financial RAG system.

User Query:
{query}

Retrieved Context:
{context}

Determine whether the retrieved context satisfies the user's
explicit requirements.

Return exactly ONE:

COMPLETE
- All major requested facts are directly supported.
- All explicit constraints are satisfied.
- If the question asks for multiple items, enough evidence exists
  for all requested items.
- If the question asks for a comparison, every metric being compared
  is supported.
- If the question asks for a ranking such as "most important",
  "top", "primary", or "most significant", the retrieved evidence
  must explicitly establish that ranking.
- If the question asks for a future effect or causal relationship,
  the evidence must explicitly support that relationship.

PARTIAL
- Useful evidence exists, but one or more requested facts,
  constraints, rankings, comparisons, metrics, timeframes,
  or relationships are missing or not fully supported.

NONE
- No useful evidence for the user's question is present.

Important rules:
- Use ONLY the retrieved context.
- Do not use outside knowledge.
- Do not infer missing facts, figures, rankings, or relationships.
- Do not treat source order, retrieval score, frequency of mention,
  or model judgment as evidence of importance.
- Multiple sources may be combined.
- A historical observation must not automatically be treated as
  evidence of a future effect.
- Be conservative when deciding COMPLETE.
- Judge the user's actual requirements, not merely topical relevance.

Examples:
- If the question asks for net income growth but the context only
  provides diluted EPS growth, return PARTIAL.
- If the question asks for three "most important" factors but the
  context only identifies relevant factors without ranking them,
  return PARTIAL.
- If the question asks for a future impact but the context only
  describes a historical fact, return PARTIAL unless the future
  relationship is explicitly stated.
- If only two of three requested facts are supported, return PARTIAL.

Return ONLY:
COMPLETE
PARTIAL
NONE

Answer:
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
        f"⚠️ Unexpected context evaluation response: "
        f"{response}"
    )

    return "NONE"


# ============================================================
# Query Rewriting
# ============================================================

def rewrite_query(
    query: str,
) -> str:
    """
    Rewrites a query for better vector + BM25 retrieval.

    This is only called during corrective retrieval.
    """

    prompt = f"""
You are a search-query optimizer for a financial RAG system.

Rewrite the user's question into a retrieval-focused query.

Goal:
Retrieve ALL factual evidence needed to answer the question.

Rules:
- Preserve the original intent.
- Identify every distinct fact requested.
- Include company, segment, and business-unit names when relevant.
- Include important financial metrics.
- Include fiscal years and dates.
- Expand useful financial terminology and acronyms when appropriate.
- Preserve ranking terms such as "most important", "top",
  "primary", or "most significant".
- Preserve comparison requirements.
- Include both amount and growth concepts when both are requested.
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
# Answer Generation
# ============================================================

def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generates a grounded answer from retrieved evidence.

    The answer is intentionally returned as Markdown.
    Source evidence is displayed separately by the frontend.
    """

    if not context.strip():
        return (
            "I could not find sufficient information in the "
            "provided documents to answer this question."
        )

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the provided evidence.

GROUNDING AND ACCURACY RULES

1. Do not use outside knowledge.

2. Do not invent facts, figures, percentages, calculations,
   rankings, or relationships.

3. Preserve the exact figures, units, and precision from the evidence.

4. Do not round, normalize, or change a financial figure unless
   the source itself does so.

5. Multiple sources may be combined when they provide
   complementary evidence.

6. If only part of the question is supported:
   - answer the supported part;
   - clearly state what requested information is missing;
   - do not present the answer as complete.

7. Never confuse different financial metrics.
   Revenue, operating income, net income, diluted EPS, gross margin,
   and cash flow are distinct metrics.

8. Never substitute one financial metric for another.
   For example, diluted EPS growth is NOT net-income growth.

9. If the question asks for net-income growth but the evidence only
   provides diluted EPS growth, explicitly say that net-income growth
   is not provided.

10. Do not infer a missing segment revenue from total company revenue.

11. Do not infer missing facts, figures, or relationships.

12. Do not convert a historical observation into a future-risk,
    future-growth, or causal claim unless the evidence explicitly
    makes that connection.

13. When synthesizing multiple sources, stay faithful to what they
    explicitly state. Reasonable synthesis is allowed, but do not
    make the synthesis stronger than the evidence.

RANKING AND PRIORITY RULES

14. Do not call factors "most important", "top", "primary",
    or "most significant" unless the retrieved evidence explicitly
    establishes that ranking.

15. Do not infer importance from source order, retrieval score,
    frequency of mention, or model judgment.

16. If the user asks for a ranked list but the evidence does not
    establish a ranking, explicitly state that the documents identify
    relevant factors but do not provide a definitive ranking.

17. If the user asks for a specific number of factors or items,
    make sure every requested item is actually supported before
    presenting the answer as complete.

QUESTION-REQUIREMENT RULES

18. Before claiming that the question is fully answered, verify that
    every explicit requirement is supported, including:
    - number of requested items;
    - ranking or priority;
    - comparison;
    - timeframe;
    - metric;
    - amount or percentage;
    - requested supporting evidence;
    - requested causal or future relationship.

19. If an explicit requirement is unsupported, say so clearly.

20. Do not claim that a source says something stronger than the
    source actually states.

ANSWER FORMAT RULES

21. Return ONLY the answer to the user's question.

22. Do NOT include source labels such as:
    [SOURCE 1]
    [SOURCE 2]
    [SOURCE 3]

23. Do NOT include citations or references inside the answer.

24. The application displays retrieved source evidence separately
    below the answer, so keep the answer clean and readable.

25. Markdown formatting is allowed and encouraged when useful.

26. You may use:
    - **bold**
    - *italics*
    - bullet lists
    - numbered lists
    - headings

27. Do not wrap the entire answer in quotation marks.

STYLE RULES

28. Keep the answer concise but informative.

29. Prefer direct, evidence-faithful wording over broad conclusions.

30. If the evidence is incomplete, do not hide the limitation behind
    confident wording.

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

    return answer


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
        include_score=False,
    ):
        """
        Converts retrieved results into:
            1. LLM context
            2. Source metadata

        Retrieval scores are retained in API metadata but omitted
        from the LLM context by default because they do not help
        answer the question and consume tokens.
        """

        if not results:
            return "", []

        context_parts = []
        sources = []

        seen_text = set()

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

            # ------------------------------------------------
            # Safety deduplication
            # ------------------------------------------------

            normalized_text = (
                " ".join(
                    text.lower().split()
                )
            )

            if normalized_text in seen_text:
                continue

            seen_text.add(
                normalized_text
            )

            metadata = result.get(
                "metadata",
                {},
            )

            score = result.get(
                "score"
            )

            source_number = len(
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
                f"[SOURCE {source_number}]\n"
                f"Document: {file_name}\n"
            )

            if page is not None:
                source_header += (
                    f"Page: {page}\n"
                )

            if include_score and score is not None:
                source_header += (
                    f"Retrieval Score: "
                    f"{float(score):.6f}\n"
                )

            source_header += (
                f"\n{text}"
            )

            context_parts.append(
                source_header
            )

            sources.append(
                {
                    "source_id": source_number,
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
    # Build Source Evaluations
    # ========================================================

    @staticmethod
    def _build_source_evaluations(
        results,
    ):
        """
        Records the final FlashRank-selected sources.

        These are not LLM-graded individually. Their inclusion is
        based on hybrid retrieval + FlashRank.
        """

        evaluations = []

        for position, result in enumerate(
            results,
            start=1,
        ):

            evaluations.append(
                {
                    "retrieved_rank": position,
                    "relevant": True,
                    "node_id": result.get(
                        "id"
                    ),
                    "evaluation_method": (
                        "hybrid_retrieval_and_flashrank"
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
        # STEP 1 — Initial Hybrid Retrieval + FlashRank
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
        # STEP 3 — Context-Level Evidence Coverage
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
                "⚠️ Retrieved context has no useful evidence."
            )

            print(
                "🔄 Rewriting query..."
            )

            rewritten_query = rewrite_query(
                query
            )

            print(
                f"🔄 Rewritten Query: "
                f"'{rewritten_query}'"
            )

            # ------------------------------------------------
            # Second retrieval
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

            second_coverage = (
                evaluate_relevance(
                    query,
                    context,
                )
            )

            print(
                f"🧠 Corrected evidence coverage: "
                f"{second_coverage}"
            )

            if second_coverage == "NONE":

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

            coverage = second_coverage

            print(
                "✅ Corrected context contains usable evidence."
            )

        else:

            print(
                f"✅ Retrieved context classified as "
                f"{coverage.lower()}."
            )

        # ====================================================
        # STEP 5 — Generate Final Answer
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