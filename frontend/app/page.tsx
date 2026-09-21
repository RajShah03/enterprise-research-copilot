"use client";

import {
  FormEvent,
  KeyboardEvent,
  useState,
} from "react";

interface Source {
  source_id: number;
  document: string;
  page: string | number | null;
  score: number | null;
  node_id: string | null;
  text: string;
}

interface QueryResponse {
  status: string;
  query: string;
  answer: string;
  sources: Source[];
  rewritten_query: string | null;
  corrective_rag: boolean;
  source_evaluations: unknown[];
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

const EXAMPLE_QUERIES = [
  "What were NVIDIA's Data Center revenues in fiscal 2026?",
  "How did NVIDIA's Data Center revenue change year over year?",
  "What risks did NVIDIA identify in its annual review?",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] =
    useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  async function submitQuery() {
    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setCopied(false);

    try {
      const response = await fetch(
        `${API_URL}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: trimmedQuery,
          }),
        }
      );

      let data: QueryResponse | { detail?: string };

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The research service returned an invalid response."
        );
      }

      if (!response.ok) {
        throw new Error(
          "detail" in data && data.detail
            ? data.detail
            : "Unable to process the question."
        );
      }

      setResult(data as QueryResponse);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while processing your question."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    submitQuery();
  }

  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey)
    ) {
      event.preventDefault();
      submitQuery();
    }
  }

  function newQuestion() {
    setQuery("");
    setResult(null);
    setError("");
    setCopied(false);
  }

  async function copyAnswer() {
    if (!result?.answer) {
      return;
    }

    try {
      await navigator.clipboard.writeText(
        result.answer
      );

      setCopied(true);

      window.setTimeout(() => {
        setCopied(false);
      }, 2000);
    } catch {
      setCopied(false);
    }
  }

  function scrollToSource(sourceId: number) {
    document
      .getElementById(`source-${sourceId}`)
      ?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
  }

  function selectExample(example: string) {
    setQuery(example);
    setResult(null);
    setError("");
    setCopied(false);
  }

  function renderAnswer(answer: string) {
    const paragraphs = answer
      .split(/\n\s*\n/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean);

    return paragraphs.map(
      (paragraph, paragraphIndex) => {
        const parts = paragraph.split(
          /(\[SOURCE \d+\])/g
        );

        return (
          <p
            key={paragraphIndex}
            className={
              paragraphIndex > 0
                ? "mt-5"
                : ""
            }
          >
            {parts.map((part, index) => {
              const match = part.match(
                /^\[SOURCE (\d+)\]$/
              );

              if (!match) {
                return (
                  <span key={index}>
                    {part}
                  </span>
                );
              }

              const sourceId = Number(
                match[1]
              );

              return (
                <button
                  key={index}
                  type="button"
                  onClick={() =>
                    scrollToSource(sourceId)
                  }
                  aria-label={`View source ${sourceId}`}
                  className="mx-1 inline-flex translate-y-[-1px] items-center rounded-md border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-blue-700 transition hover:border-blue-300 hover:bg-blue-100"
                >
                  {sourceId}
                </button>
              );
            })}
          </p>
        );
      }
    );
  }

  function statusMeta() {
    if (!result) {
      return null;
    }

    if (result.status === "complete") {
      return {
        label: "Well supported",
        className:
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        dot:
          "bg-emerald-500",
      };
    }

    if (result.status === "partial") {
      return {
        label: "Partially supported",
        className:
          "border-amber-200 bg-amber-50 text-amber-700",
        dot:
          "bg-amber-500",
      };
    }

    return {
      label: "Limited evidence",
      className:
        "border-slate-200 bg-slate-50 text-slate-600",
      dot:
        "bg-slate-400",
    };
  }

  const status = statusMeta();

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-slate-900">

      {/* ====================================================
          Header
          ==================================================== */}

      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 sm:px-8">

          <button
            type="button"
            onClick={newQuestion}
            className="flex items-center gap-3"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-950 text-[11px] font-bold tracking-tight text-white">
              ER
            </div>

            <div className="text-left">
              <div className="text-sm font-semibold tracking-tight text-slate-900">
                Enterprise Research Copilot
              </div>

              <div className="hidden text-[11px] text-slate-400 sm:block">
                Document-grounded research assistant
              </div>
            </div>
          </button>

          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-medium text-emerald-700 sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              System ready
            </div>

            <button
              type="button"
              onClick={newQuestion}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
            >
              New question
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-5 pb-20 sm:px-8">

        {/* ====================================================
            Hero
            ==================================================== */}

        {!result && !loading && !error && (
          <section className="mx-auto max-w-3xl pt-16 text-center sm:pt-20">

            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[11px] font-medium text-slate-500 shadow-sm">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
              AI-powered document research
            </div>

            <h1 className="text-4xl font-semibold tracking-[-0.035em] text-slate-950 sm:text-5xl">
              Research with evidence,
              <span className="block text-blue-600">
                not just answers.
              </span>
            </h1>

            <p className="mx-auto mt-5 max-w-2xl text-sm leading-6 text-slate-500 sm:text-base">
              Ask questions about your indexed documents.
              Receive grounded answers with supporting
              evidence and page references.
            </p>
          </section>
        )}

        {/* ====================================================
            Search
            ==================================================== */}

        <section className="mx-auto max-w-3xl pt-10 sm:pt-12">

          <form onSubmit={handleSubmit}>
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_8px_32px_rgba(15,23,42,0.06)] transition focus-within:border-blue-300 focus-within:ring-4 focus-within:ring-blue-100">

              <textarea
                value={query}
                onChange={(event) =>
                  setQuery(event.target.value)
                }
                onKeyDown={handleKeyDown}
                disabled={loading}
                rows={4}
                placeholder="Ask a question about your research documents..."
                className="w-full resize-none bg-transparent px-5 py-5 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed disabled:opacity-60"
              />

              <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3">

                <div className="hidden items-center gap-2 text-[11px] text-slate-400 sm:flex">
                  <span className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-medium">
                    Ctrl
                  </span>

                  <span>+</span>

                  <span className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-medium">
                    Enter
                  </span>

                  <span>to ask</span>
                </div>

                <button
                  type="submit"
                  disabled={
                    loading || !query.trim()
                  }
                  className="ml-auto inline-flex items-center gap-2 rounded-xl bg-slate-950 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      Researching
                    </>
                  ) : (
                    <>
                      Ask Copilot
                      <ArrowUpIcon />
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* Example queries */}
          {!result && !loading && !error && (
            <div className="mt-5">
              <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                Example questions
              </div>

              <div className="flex flex-wrap gap-2">
                {EXAMPLE_QUERIES.map(
                  (example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() =>
                        selectExample(example)
                      }
                      className="rounded-full border border-slate-200 bg-white px-3.5 py-2 text-xs text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
                    >
                      {example}
                    </button>
                  )
                )}
              </div>
            </div>
          )}
        </section>

        {/* ====================================================
            Loading
            ==================================================== */}

        {loading && (
          <section className="mx-auto mt-8 max-w-3xl rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-blue-600" />

              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Researching your question
                </p>

                <p className="mt-0.5 text-xs text-slate-400">
                  Retrieving and evaluating relevant evidence...
                </p>
              </div>
            </div>
          </section>
        )}

        {/* ====================================================
            Error
            ==================================================== */}

        {error && !loading && (
          <section className="mx-auto mt-8 max-w-3xl rounded-2xl border border-red-200 bg-red-50 p-5">
            <div className="flex gap-3">
              <div className="mt-0.5 text-red-500">
                <AlertIcon />
              </div>

              <div>
                <p className="text-sm font-semibold text-red-700">
                  We couldn't process that question
                </p>

                <p className="mt-1 text-sm leading-5 text-red-600">
                  {error}
                </p>
              </div>
            </div>
          </section>
        )}

        {/* ====================================================
            Results
            ==================================================== */}

        {result && !loading && !error && (
          <section className="mt-10">

            {/* Question */}
            <div className="mx-auto max-w-3xl">
              <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                Your question
              </div>

              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <h2 className="text-lg font-medium leading-7 text-slate-900">
                  {result.query}
                </h2>

                {status && (
                  <div
                    className={`flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold ${status.className}`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${status.dot}`}
                    />
                    {status.label}
                  </div>
                )}
              </div>
            </div>

            {/* Answer + Sources */}
            <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">

              {/* ==================================================
                  Answer
                  ================================================== */}

              <article className="min-w-0 rounded-2xl border border-slate-200 bg-white shadow-sm">

                <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">

                  <div className="flex items-center gap-2.5">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
                      <SparkIcon />
                    </div>

                    <div>
                      <div className="text-sm font-semibold text-slate-900">
                        Answer
                      </div>

                      <div className="text-[11px] text-slate-400">
                        Grounded in retrieved evidence
                      </div>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={copyAnswer}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
                  >
                    {copied ? (
                      <>
                        <CheckIcon />
                        Copied
                      </>
                    ) : (
                      <>
                        <CopyIcon />
                        Copy
                      </>
                    )}
                  </button>
                </div>

                <div className="px-6 py-7 sm:px-8 sm:py-8">
                  <div className="text-[15px] leading-7 text-slate-700">
                    {renderAnswer(result.answer)}
                  </div>
                </div>
              </article>

              {/* ==================================================
                  Sources
                  ================================================== */}

              <aside>
                <div className="mb-4">
                  <h3 className="text-sm font-semibold text-slate-900">
                    Sources
                  </h3>

                  <p className="mt-1 text-xs leading-5 text-slate-400">
                    Evidence retrieved for this answer.
                  </p>
                </div>

                <div className="space-y-3">
                  {result.sources.map(
                    (source) => (
                      <article
                        id={`source-${source.source_id}`}
                        key={`${source.source_id}-${source.node_id}`}
                        className="scroll-mt-24 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition hover:border-slate-300"
                      >
                        <div className="flex items-start gap-3">
                          <span className="flex h-6 min-w-6 items-center justify-center rounded-md bg-blue-50 px-1.5 text-[10px] font-bold text-blue-700">
                            {source.source_id}
                          </span>

                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-[11px] font-medium text-slate-400">
                                Source
                              </span>

                              {source.page !==
                                null && (
                                <>
                                  <span className="text-slate-300">
                                    ·
                                  </span>

                                  <span className="text-[11px] font-medium text-slate-400">
                                    Page{" "}
                                    {source.page}
                                  </span>
                                </>
                              )}
                            </div>

                            <div className="mt-1.5 break-words text-xs font-semibold leading-5 text-slate-800">
                              {source.document}
                            </div>
                          </div>
                        </div>

                        <div className="mt-3 border-t border-slate-100 pt-3">
                          <p className="line-clamp-7 text-xs leading-5 text-slate-500">
                            {source.text}
                          </p>
                        </div>
                      </article>
                    )
                  )}
                </div>
              </aside>
            </div>

            {/* ==================================================
                Technical Details
                ================================================== */}

            <div className="mt-8">
              <details className="rounded-2xl border border-slate-200 bg-white shadow-sm">

                <summary className="cursor-pointer list-none px-5 py-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-sm font-semibold text-slate-800">
                        Technical details
                      </div>

                      <div className="mt-1 text-xs text-slate-400">
                        How this answer was produced.
                      </div>
                    </div>

                    <ChevronIcon />
                  </div>
                </summary>

                <div className="border-t border-slate-100 px-5 py-5">

                  <div className="grid gap-5 sm:grid-cols-4">

                    <TechnicalItem
                      label="Retrieval"
                      value="Vector + BM25"
                    />

                    <TechnicalItem
                      label="Reranking"
                      value="FlashRank"
                    />

                    <TechnicalItem
                      label="Evidence"
                      value={
                        result.status
                          .toUpperCase()
                      }
                    />

                    <TechnicalItem
                      label="Corrective RAG"
                      value={
                        result.corrective_rag
                          ? "Triggered"
                          : "Not triggered"
                      }
                    />
                  </div>

                  {result.rewritten_query && (
                    <div className="mt-5 border-t border-slate-100 pt-5">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                        Corrective query
                      </div>

                      <div className="mt-2 rounded-xl bg-slate-50 p-3 text-sm leading-6 text-slate-600">
                        {result.rewritten_query}
                      </div>
                    </div>
                  )}
                </div>
              </details>
            </div>
          </section>
        )}

        {/* ====================================================
            Empty-state features
            ==================================================== */}

        {!result && !loading && !error && (
          <section className="mx-auto mt-16 max-w-3xl">
            <div className="grid gap-4 sm:grid-cols-3">

              <FeatureCard
                icon={<DocumentIcon />}
                title="Grounded answers"
                text="Responses are generated from retrieved document evidence."
              />

              <FeatureCard
                icon={<SourceIcon />}
                title="Traceable evidence"
                text="See the document passages and page references behind an answer."
              />

              <FeatureCard
                icon={<ShieldIcon />}
                title="Evidence aware"
                text="The system can identify when retrieved evidence is insufficient."
              />
            </div>
          </section>
        )}

        {/* ====================================================
            Footer
            ==================================================== */}

        <footer className="mt-16 border-t border-slate-200 pt-6 text-center text-[11px] text-slate-400">
          Enterprise Research Copilot · FastAPI · Next.js · Hybrid Retrieval · Corrective RAG
        </footer>
      </div>
    </main>
  );
}


/* ============================================================
   Reusable Components
   ============================================================ */

function TechnicalItem({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
        {label}
      </div>

      <div className="mt-1.5 text-sm text-slate-700">
        {value}
      </div>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg bg-slate-50 text-slate-600">
        {icon}
      </div>

      <h3 className="text-sm font-semibold text-slate-800">
        {title}
      </h3>

      <p className="mt-2 text-xs leading-5 text-slate-500">
        {text}
      </p>
    </div>
  );
}


/* ============================================================
   Icons
   ============================================================ */

function ArrowUpIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 19V5" />
      <path d="m6 11 6-6 6 6" />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m12 3-1.3 5.2L6 10l4.7 1.8L12 17l1.3-5.2L18 10l-4.7-1.8L12 3Z" />
      <path d="m19 16-.6 2.4L16 19l2.4.6L19 22l.6-2.4L22 19l-2.4-.6L19 16Z" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect
        x="9"
        y="9"
        width="11"
        height="11"
        rx="2"
      />

      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m5 12 4 4L19 6" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
      <path d="M10.3 3.7 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3l-7.5-13.3a2 2 0 0 0-3.4 0Z" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
      <path d="M14 2v6h6" />
      <path d="M8 13h8" />
      <path d="M8 17h5" />
    </svg>
  );
}

function SourceIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 12.5 11 15l4.5-5" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3 19 6v5c0 4.7-2.8 8-7 10-4.2-2-7-5.3-7-10V6l7-3Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}