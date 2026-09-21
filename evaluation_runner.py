import csv
import json
import re
import statistics
import time
from pathlib import Path
from urllib import error, request


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

QUERY_ENDPOINT = f"{BASE_URL}/query"

DATASET_PATH = Path(
    "nvidia_rag_evaluation_dataset_v2.json"
)

OUTPUT_JSON = Path(
    "evaluation_results.json"
)

OUTPUT_CSV = Path(
    "evaluation_results.csv"
)

SUMMARY_JSON = Path(
    "evaluation_summary.json"
)

REQUEST_TIMEOUT = 180


# ============================================================
# HTTP
# ============================================================

def call_rag_api(
    query: str,
):
    payload = json.dumps(
        {
            "query": query
        }
    ).encode("utf-8")

    req = request.Request(
        QUERY_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()

    try:

        with request.urlopen(
            req,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            body = (
                response
                .read()
                .decode("utf-8")
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            return (
                response.status,
                json.loads(body),
                latency_ms,
                None,
            )

    except error.HTTPError as exc:

        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            exc.code,
            None,
            latency_ms,
            body,
        )

    except Exception as exc:

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            None,
            None,
            latency_ms,
            str(exc),
        )


# ============================================================
# Text Normalization
# ============================================================

def normalize(
    text,
):
    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace(
        "\u202f",
        " ",
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    text = text.replace(
        "“",
        '"',
    )

    text = text.replace(
        "”",
        '"',
    )

    text = text.replace(
        "’",
        "'",
    )

    # Remove citation syntax from simple fact matching.
    text = re.sub(
        r"\[\s*source\s+\d+\s*\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"【\s*source\s+\d+\s*】",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return " ".join(
        text.split()
    )


# ============================================================
# Status Evaluation
# ============================================================

def evaluate_status(
    expected_status,
    actual_response,
):
    actual_status = (
        actual_response.get(
            "status",
            "unknown",
        )
    )

    answer = normalize(
        actual_response.get(
            "answer",
            "",
        )
    )

    if expected_status == "not_found":

        return (
            actual_status == "not_found"
            or (
                not actual_response.get(
                    "sources"
                )
                and (
                    "could not find" in answer
                    or "not found" in answer
                    or "insufficient" in answer
                    or "not available" in answer
                )
            )
        )

    if expected_status == "partial":

        return (
            actual_status == "partial"
            or (
                actual_status == "success"
                and (
                    "partial" in answer
                    or "not provided" in answer
                    or "does not provide" in answer
                    or "cannot be completed" in answer
                    or "not ranked" in answer
                    or "cannot be determined" in answer
                )
            )
        )

    if expected_status == "complete":

        return (
            actual_status == "complete"
            or (
                actual_status == "success"
                and bool(answer)
                and bool(
                    actual_response.get(
                        "sources"
                    )
                )
            )
        )

    return False


# ============================================================
# Evidence Matching
# ============================================================

def source_matches_expected(
    source,
    expected,
):
    """
    Expected evidence can contain:

        document
        page
        keywords
        match_mode

    match_mode:
        all  -> all keywords required
        any  -> at least one keyword required
    """

    document = normalize(
        source.get(
            "document",
            "",
        )
    )

    page = str(
        source.get(
            "page",
            "",
        )
    ).strip()

    source_text = normalize(
        source.get(
            "text",
            "",
        )
    )

    expected_document = normalize(
        expected.get(
            "document",
            "",
        )
    )

    expected_page = str(
        expected.get(
            "page",
            "",
        )
    ).strip()

    if (
        expected_document
        and expected_document not in document
    ):
        return False

    if (
        expected_page
        and expected_page != page
    ):
        return False

    keywords = expected.get(
        "keywords",
        [],
    )

    if not keywords:
        return True

    normalized_keywords = [
        normalize(keyword)
        for keyword in keywords
    ]

    match_mode = (
        expected.get(
            "match_mode",
            "all",
        )
        .lower()
    )

    if match_mode == "any":

        return any(
            keyword in source_text
            for keyword in normalized_keywords
        )

    return all(
        keyword in source_text
        for keyword in normalized_keywords
    )


# ============================================================
# Evidence Coverage
# ============================================================

def evidence_coverage_at_k(
    expected_evidence,
    sources,
    k,
):
    if not expected_evidence:
        return None

    top_sources = sources[:k]

    matched = 0

    for expected in expected_evidence:

        found = any(
            source_matches_expected(
                source,
                expected,
            )
            for source in top_sources
        )

        if found:
            matched += 1

    return (
        matched
        / len(expected_evidence)
    )


def calculate_retrieval_metrics(
    expected_evidence,
    sources,
):
    if not expected_evidence:

        return {
            "evidence_count": 0,
            "evidence_recall_at_1": None,
            "evidence_recall_at_3": None,
            "evidence_recall_at_6": None,
            "first_complete_rank": None,
            "evidence_complete": True,
            "matching_source_ids": [],
            "evidence_matches": [],
        }

    evidence_matches = []

    for index, expected in enumerate(
        expected_evidence,
        start=1,
    ):

        matching_positions = []

        for position, source in enumerate(
            sources,
            start=1,
        ):

            if source_matches_expected(
                source,
                expected,
            ):

                matching_positions.append(
                    position
                )

        evidence_matches.append(
            {
                "evidence_index": index,
                "matching_positions": (
                    matching_positions
                ),
                "first_position": (
                    matching_positions[0]
                    if matching_positions
                    else None
                ),
            }
        )

    recall_at_1 = evidence_coverage_at_k(
        expected_evidence,
        sources,
        1,
    )

    recall_at_3 = evidence_coverage_at_k(
        expected_evidence,
        sources,
        3,
    )

    recall_at_6 = evidence_coverage_at_k(
        expected_evidence,
        sources,
        6,
    )

    first_complete_rank = None

    for k in range(
        1,
        len(sources) + 1,
    ):

        coverage = evidence_coverage_at_k(
            expected_evidence,
            sources,
            k,
        )

        if coverage == 1.0:

            first_complete_rank = k
            break

    matching_source_ids = []

    for source in sources:

        if any(
            source_matches_expected(
                source,
                expected,
            )
            for expected in expected_evidence
        ):

            matching_source_ids.append(
                source.get(
                    "source_id"
                )
            )

    return {
        "evidence_count": len(
            expected_evidence
        ),
        "evidence_recall_at_1": recall_at_1,
        "evidence_recall_at_3": recall_at_3,
        "evidence_recall_at_6": recall_at_6,
        "first_complete_rank": (
            first_complete_rank
        ),
        "evidence_complete": (
            first_complete_rank is not None
        ),
        "matching_source_ids": (
            matching_source_ids
        ),
        "evidence_matches": (
            evidence_matches
        ),
    }


# ============================================================
# Required Facts
# ============================================================

def evaluate_required_facts(
    answer,
    must_include=None,
    must_include_any=None,
    accepted_values=None,
):
    """
    Supports three fact-definition mechanisms:

    1. must_include
       Every exact fact must appear.

    2. must_include_any
       Each group requires at least one accepted phrase.

       Example:
       [
           ["215.9 billion", "215,938 million"],
           ["130.5 billion", "130,497 million"]
       ]

    3. accepted_values
       Dictionary of semantic fact groups.

       Example:
       {
           "revenue": [
               "$194 billion",
               "$193.7 billion"
           ],
           "growth": [
               "68%",
               "68 percent"
           ]
       }
    """

    normalized_answer = normalize(
        answer
    )

    matched = []
    missing = []
    coverage_groups = []

    # --------------------------------------------------------
    # Exact required facts
    # --------------------------------------------------------

    for fact in must_include or []:

        if normalize(fact) in normalized_answer:

            matched.append(
                fact
            )

            coverage_groups.append(
                True
            )

        else:

            missing.append(
                fact
            )

            coverage_groups.append(
                False
            )

    # --------------------------------------------------------
    # Must-include-any groups
    # --------------------------------------------------------

    for index, group in enumerate(
        must_include_any or [],
        start=1,
    ):

        matched_value = next(
            (
                value
                for value in group
                if normalize(value)
                in normalized_answer
            ),
            None,
        )

        if matched_value is not None:

            matched.append(
                {
                    "group": index,
                    "matched_value": matched_value,
                }
            )

            coverage_groups.append(
                True
            )

        else:

            missing.append(
                {
                    "group": index,
                    "accepted_values": group,
                }
            )

            coverage_groups.append(
                False
            )

    # --------------------------------------------------------
    # Accepted values
    # --------------------------------------------------------

    accepted_values = (
        accepted_values or {}
    )

    for group_name, values in (
        accepted_values.items()
    ):

        matched_value = next(
            (
                value
                for value in values
                if normalize(value)
                in normalized_answer
            ),
            None,
        )

        if matched_value is not None:

            matched.append(
                {
                    "group": group_name,
                    "matched_value": matched_value,
                }
            )

            coverage_groups.append(
                True
            )

        else:

            missing.append(
                {
                    "group": group_name,
                    "accepted_values": values,
                }
            )

            coverage_groups.append(
                False
            )

    if not coverage_groups:

        coverage = None

    else:

        coverage = (
            sum(coverage_groups)
            / len(coverage_groups)
        )

    return {
        "fact_coverage": coverage,
        "matched": matched,
        "missing": missing,
    }


# ============================================================
# Forbidden Claims
# ============================================================

def evaluate_forbidden_claims(
    answer,
    must_not_claim,
):
    if not must_not_claim:

        return {
            "violation": False,
            "violations": [],
        }

    normalized_answer = normalize(
        answer
    )

    violations = []

    for claim in must_not_claim:

        normalized_claim = normalize(
            claim
        )

        # Direct string match.
        if (
            normalized_claim
            and normalized_claim
            in normalized_answer
        ):

            violations.append(
                claim
            )

            continue

        # Specific ranking protection.
        if (
            "rank" in normalized_claim
            and (
                "most important"
                in normalized_answer
                or "top 3"
                in normalized_answer
                or "three most"
                in normalized_answer
                or "top three"
                in normalized_answer
            )
        ):

            violations.append(
                claim
            )

    return {
        "violation": bool(
            violations
        ),
        "violations": violations,
    }


# ============================================================
# Citations
# ============================================================

def extract_citations(
    answer,
):
    """
    Extract only the canonical citation format:

        [SOURCE 1]
        [SOURCE 2]
    """

    values = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        answer,
    )

    return sorted(
        {
            int(value)
            for value in values
        }
    )


def evaluate_citations(
    answer,
    sources,
    citation_required,
):
    citations = extract_citations(
        answer
    )

    valid_source_ids = {
        int(source["source_id"])
        for source in sources
        if source.get(
            "source_id"
        ) is not None
    }

    invalid_lowercase = bool(
        re.search(
            r"\[source\s+\d+\]",
            answer,
        )
    )

    invalid_mixed_case = bool(
        re.search(
            r"\[Source\s+\d+\]",
            answer,
        )
    )

    invalid_unicode = bool(
        re.search(
            r"【\s*SOURCE\s+\d+\s*】",
            answer,
            flags=re.IGNORECASE,
        )
    )

    invalid_numeric = bool(
        re.search(
            r"\[\d+\]",
            answer,
        )
    )

    format_valid = not any(
        [
            invalid_lowercase,
            invalid_mixed_case,
            invalid_unicode,
            invalid_numeric,
        ]
    )

    ids_valid = all(
        citation in valid_source_ids
        for citation in citations
    )

    if not citation_required:

        return {
            "citation_required": False,
            "citation_present": bool(
                citations
            ),
            "citation_format_valid": (
                format_valid
            ),
            "citation_ids": citations,
            "citation_ids_valid": ids_valid,
            "citation_valid": True,
        }

    citation_valid = (
        bool(citations)
        and format_valid
        and ids_valid
    )

    return {
        "citation_required": True,
        "citation_present": bool(
            citations
        ),
        "citation_format_valid": (
            format_valid
        ),
        "citation_ids": citations,
        "citation_ids_valid": ids_valid,
        "citation_valid": citation_valid,
        "citation_count": len(
            citations
        ),
    }


# ============================================================
# Abstention
# ============================================================

def evaluate_abstention(
    response,
    expected_abstention,
):
    if not expected_abstention:

        return None

    status = response.get(
        "status"
    )

    answer = normalize(
        response.get(
            "answer",
            "",
        )
    )

    sources = response.get(
        "sources",
        [],
    )

    explicit_abstention = any(
        phrase in answer
        for phrase in [
            "could not find",
            "not available",
            "insufficient",
            "cannot be determined",
            "not provided",
        ]
    )

    return (
        status in {
            "not_found",
            "partial",
        }
        or (
            not sources
            and explicit_abstention
        )
    )


# ============================================================
# Per-Question Evaluation
# ============================================================

def evaluate_question(
    item,
):
    print(
        f"\n{'=' * 80}\n"
        f"{item['id'].upper()}: "
        f"{item['question']}\n"
        f"{'=' * 80}"
    )

    (
        http_status,
        response,
        latency_ms,
        error_message,
    ) = call_rag_api(
        item["question"]
    )

    if response is None:

        print(
            "❌ API request failed:"
            f" {error_message}"
        )

        return {
            "id": item["id"],
            "question": item["question"],
            "http_status": http_status,
            "latency_ms": round(
                latency_ms,
                2,
            ),
            "error": error_message,
            "response": None,
            "evaluation": {
                "status_match": False,
                "fact_coverage": None,
                "retrieval": {},
                "citation": {},
                "forbidden_claims": {},
                "abstention_correct": None,
                "answer_pass": False,
                "retrieval_pass": False,
                "overall_pass": False,
            },
        }

    answer = response.get(
        "answer",
        "",
    )

    sources = response.get(
        "sources",
        [],
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status_match = evaluate_status(
        item["expected_status"],
        response,
    )

    # --------------------------------------------------------
    # Facts
    # --------------------------------------------------------

    fact_result = evaluate_required_facts(
        answer=answer,
        must_include=item.get(
            "must_include",
            [],
        ),
        must_include_any=item.get(
            "must_include_any",
            [],
        ),
        accepted_values=item.get(
            "accepted_values",
            {},
        ),
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_result = (
        calculate_retrieval_metrics(
            item.get(
                "expected_evidence",
                [],
            ),
            sources,
        )
    )

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    citation_result = evaluate_citations(
        answer,
        sources,
        item.get(
            "citation_required",
            False,
        ),
    )

    # --------------------------------------------------------
    # Forbidden claims
    # --------------------------------------------------------

    forbidden_result = (
        evaluate_forbidden_claims(
            answer,
            item.get(
                "must_not_claim",
                [],
            ),
        )
    )

    # --------------------------------------------------------
    # Abstention
    # --------------------------------------------------------

    abstention_correct = evaluate_abstention(
        response,
        item.get(
            "expected_abstention",
            False,
        ),
    )

    # --------------------------------------------------------
    # Answer-level pass
    # --------------------------------------------------------

    answer_components = [
        bool(status_match),
        not forbidden_result["violation"],
    ]

    if (
        fact_result["fact_coverage"]
        is not None
    ):

        answer_components.append(
            fact_result[
                "fact_coverage"
            ] == 1.0
        )

    if item.get(
        "citation_required",
        False,
    ):

        answer_components.append(
            citation_result.get(
                "citation_valid",
                False,
            )
        )

    if item.get(
        "expected_abstention",
        False,
    ):

        answer_components.append(
            bool(abstention_correct)
        )

    answer_pass = all(
        answer_components
    )

    # --------------------------------------------------------
    # Retrieval-level pass
    # --------------------------------------------------------

    if item.get(
        "expected_evidence"
    ):

        retrieval_pass = (
            retrieval_result.get(
                "evidence_complete",
                False,
            )
        )

    else:

        retrieval_pass = True

    # --------------------------------------------------------
    # Overall pass
    # --------------------------------------------------------

    overall_pass = (
        answer_pass
        and retrieval_pass
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"HTTP: {http_status}"
    )

    print(
        f"Latency: {latency_ms:.2f} ms"
    )

    print(
        "Status match: "
        f"{'✅' if status_match else '❌'}"
    )

    if (
        fact_result["fact_coverage"]
        is not None
    ):

        print(
            "Fact coverage: "
            f"{fact_result['fact_coverage']:.0%}"
        )

        if fact_result["missing"]:

            print(
                "Missing facts: "
                f"{fact_result['missing']}"
            )

    if item.get(
        "expected_evidence"
    ):

        print(
            "Evidence Recall@1: "
            f"{retrieval_result['evidence_recall_at_1']:.0%}"
        )

        print(
            "Evidence Recall@3: "
            f"{retrieval_result['evidence_recall_at_3']:.0%}"
        )

        print(
            "Evidence Recall@6: "
            f"{retrieval_result['evidence_recall_at_6']:.0%}"
        )

        print(
            "First complete rank: "
            f"{retrieval_result['first_complete_rank']}"
        )

    if item.get(
        "citation_required",
        False,
    ):

        print(
            "Citation: "
            f"{'✅' if citation_result.get('citation_valid') else '❌'}"
        )

    if forbidden_result[
        "violation"
    ]:

        print(
            "Forbidden claims: ❌ "
            f"{forbidden_result['violations']}"
        )

    print(
        "\nAnswer:"
    )

    print(
        answer
    )

    print(
        "\nAnswer evaluation: "
        f"{'✅ PASS' if answer_pass else '❌ REVIEW'}"
    )

    print(
        "Retrieval evaluation: "
        f"{'✅ PASS' if retrieval_pass else '❌ REVIEW'}"
    )

    print(
        "Overall evaluation: "
        f"{'✅ PASS' if overall_pass else '❌ REVIEW'}"
    )

    return {
        "id": item["id"],
        "question": item["question"],
        "http_status": http_status,
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "response": response,
        "evaluation": {
            "status_match": status_match,
            "fact_coverage": fact_result[
                "fact_coverage"
            ],
            "fact_matched": fact_result[
                "matched"
            ],
            "fact_missing": fact_result[
                "missing"
            ],
            "retrieval": retrieval_result,
            "citation": citation_result,
            "forbidden_claims": forbidden_result,
            "abstention_correct": abstention_correct,
            "answer_pass": answer_pass,
            "retrieval_pass": retrieval_pass,
            "overall_pass": overall_pass,
        },
    }


# ============================================================
# Safe Statistics
# ============================================================

def safe_mean(
    values,
):
    if not values:
        return None

    return statistics.mean(
        values
    )


# ============================================================
# Summary
# ============================================================

def calculate_summary(
    results,
):
    total = len(
        results
    )

    valid_results = [
        result
        for result in results
        if result.get(
            "response"
        ) is not None
    ]

    answer_passes = [
        result["evaluation"][
            "answer_pass"
        ]
        for result in valid_results
    ]

    retrieval_passes = [
        result["evaluation"][
            "retrieval_pass"
        ]
        for result in valid_results
    ]

    overall_passes = [
        result["evaluation"][
            "overall_pass"
        ]
        for result in valid_results
    ]

    status_matches = [
        result["evaluation"][
            "status_match"
        ]
        for result in valid_results
    ]

    recall_at_1 = [
        result["evaluation"][
            "retrieval"
        ]["evidence_recall_at_1"]
        for result in valid_results
        if result["evaluation"][
            "retrieval"
        ].get(
            "evidence_recall_at_1"
        ) is not None
    ]

    recall_at_3 = [
        result["evaluation"][
            "retrieval"
        ]["evidence_recall_at_3"]
        for result in valid_results
        if result["evaluation"][
            "retrieval"
        ].get(
            "evidence_recall_at_3"
        ) is not None
    ]

    recall_at_6 = [
        result["evaluation"][
            "retrieval"
        ]["evidence_recall_at_6"]
        for result in valid_results
        if result["evaluation"][
            "retrieval"
        ].get(
            "evidence_recall_at_6"
        ) is not None
    ]

    complete_ranks = [
        result["evaluation"][
            "retrieval"
        ]["first_complete_rank"]
        for result in valid_results
        if result["evaluation"][
            "retrieval"
        ].get(
            "first_complete_rank"
        ) is not None
    ]

    fact_coverages = [
        result["evaluation"][
            "fact_coverage"
        ]
        for result in valid_results
        if result["evaluation"][
            "fact_coverage"
        ] is not None
    ]

    latencies = [
        result["latency_ms"]
        for result in valid_results
    ]

    citation_results = [
        result["evaluation"][
            "citation"
        ]
        for result in valid_results
        if result["evaluation"][
            "citation"
        ].get(
            "citation_required",
            False,
        )
    ]

    citation_presence = [
        result.get(
            "citation_present",
            False,
        )
        for result in citation_results
    ]

    citation_format = [
        result.get(
            "citation_format_valid",
            False,
        )
        for result in citation_results
    ]

    citation_ids = [
        result.get(
            "citation_ids_valid",
            False,
        )
        for result in citation_results
    ]

    citation_valid = [
        result.get(
            "citation_valid",
            False,
        )
        for result in citation_results
    ]

    return {
        "total_questions": total,
        "successful_api_responses": len(
            valid_results
        ),

        "answer_pass_count": sum(
            answer_passes
        ),

        "answer_pass_rate": (
            sum(answer_passes)
            / len(answer_passes)
            if answer_passes
            else 0.0
        ),

        "retrieval_pass_count": sum(
            retrieval_passes
        ),

        "retrieval_pass_rate": (
            sum(retrieval_passes)
            / len(retrieval_passes)
            if retrieval_passes
            else 0.0
        ),

        "overall_pass_count": sum(
            overall_passes
        ),

        "overall_pass_rate": (
            sum(overall_passes)
            / len(overall_passes)
            if overall_passes
            else 0.0
        ),

        "status_match_rate": (
            sum(status_matches)
            / len(status_matches)
            if status_matches
            else 0.0
        ),

        "evidence_recall_at_1": safe_mean(
            recall_at_1
        ),

        "evidence_recall_at_3": safe_mean(
            recall_at_3
        ),

        "evidence_recall_at_6": safe_mean(
            recall_at_6
        ),

        "average_first_complete_rank": (
            safe_mean(
                complete_ranks
            )
        ),

        "mean_required_fact_coverage": (
            safe_mean(
                fact_coverages
            )
        ),

        "citation_presence_rate": (
            safe_mean(
                citation_presence
            )
        ),

        "citation_format_valid_rate": (
            safe_mean(
                citation_format
            )
        ),

        "citation_ids_valid_rate": (
            safe_mean(
                citation_ids
            )
        ),

        "citation_valid_rate": (
            safe_mean(
                citation_valid
            )
        ),

        "average_latency_ms": (
            safe_mean(
                latencies
            )
        ),

        "median_latency_ms": (
            statistics.median(
                latencies
            )
            if latencies
            else None
        ),

        "max_latency_ms": (
            max(latencies)
            if latencies
            else None
        ),
    }


# ============================================================
# CSV Output
# ============================================================

def write_csv(
    results,
):
    rows = []

    for result in results:

        evaluation = result.get(
            "evaluation",
            {},
        )

        retrieval = evaluation.get(
            "retrieval",
            {},
        )

        citation = evaluation.get(
            "citation",
            {},
        )

        rows.append(
            {
                "id": result.get(
                    "id"
                ),
                "question": result.get(
                    "question"
                ),
                "http_status": result.get(
                    "http_status"
                ),
                "latency_ms": result.get(
                    "latency_ms"
                ),
                "status_match": evaluation.get(
                    "status_match"
                ),
                "fact_coverage": evaluation.get(
                    "fact_coverage"
                ),
                "evidence_recall_at_1": retrieval.get(
                    "evidence_recall_at_1"
                ),
                "evidence_recall_at_3": retrieval.get(
                    "evidence_recall_at_3"
                ),
                "evidence_recall_at_6": retrieval.get(
                    "evidence_recall_at_6"
                ),
                "first_complete_rank": retrieval.get(
                    "first_complete_rank"
                ),
                "retrieval_pass": evaluation.get(
                    "retrieval_pass"
                ),
                "citation_present": citation.get(
                    "citation_present"
                ),
                "citation_format_valid": citation.get(
                    "citation_format_valid"
                ),
                "citation_ids_valid": citation.get(
                    "citation_ids_valid"
                ),
                "citation_valid": citation.get(
                    "citation_valid"
                ),
                "answer_pass": evaluation.get(
                    "answer_pass"
                ),
                "overall_pass": evaluation.get(
                    "overall_pass"
                ),
            }
        )

    fieldnames = [
        "id",
        "question",
        "http_status",
        "latency_ms",
        "status_match",
        "fact_coverage",
        "evidence_recall_at_1",
        "evidence_recall_at_3",
        "evidence_recall_at_6",
        "first_complete_rank",
        "retrieval_pass",
        "citation_present",
        "citation_format_valid",
        "citation_ids_valid",
        "citation_valid",
        "answer_pass",
        "overall_pass",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(
            file
        )

    if not isinstance(
        dataset,
        list,
    ):

        raise ValueError(
            "Dataset must contain a JSON array."
        )

    print(
        f"Loaded {len(dataset)} benchmark questions."
    )

    results = []

    for item in dataset:

        results.append(
            evaluate_question(
                item
            )
        )

    summary = calculate_summary(
        results
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_csv(
        results
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "📊 EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    for key, value in summary.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        "\n✅ Created:"
    )

    print(
        f"   {OUTPUT_JSON}"
    )

    print(
        f"   {OUTPUT_CSV}"
    )

    print(
        f"   {SUMMARY_JSON}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()