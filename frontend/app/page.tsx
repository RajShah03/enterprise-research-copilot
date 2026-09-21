"use client";

import {
  ChangeEvent,
  DragEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

const MAX_FILE_SIZE = 25 * 1024 * 1024;

type Role = "user" | "assistant";

type Source = {
  source_id: number;
  document?: string | null;
  page?: string | number | null;
  score?: number | null;
  node_id?: string | null;
  text?: string;
};

type SourceEvaluation = {
  retrieved_rank?: number;
  relevant?: boolean;
  node_id?: string | null;
  evaluation_method?: string;
};

type Message = {
  id: string;
  role: Role;
  content: string;
  status?: string;
  sources?: Source[];
  sourceEvaluations?: SourceEvaluation[];
  rewrittenQuery?: string | null;
  correctiveRag?: boolean;
};

export default function Home() {
  const [documentId, setDocumentId] =
    useState<string | null>(null);

  const [filename, setFilename] =
    useState("");

  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [messages, setMessages] =
    useState<Message[]>([]);

  const [query, setQuery] =
    useState("");

  const [uploading, setUploading] =
    useState(false);

  const [asking, setAsking] =
    useState(false);

  const [error, setError] =
    useState("");

  const [darkMode, setDarkMode] =
    useState(false);

  const fileInputRef =
    useRef<HTMLInputElement | null>(null);

  const hasDocument =
    Boolean(documentId);

  // ==========================================================
  // Load Saved Theme
  // ==========================================================

  useEffect(() => {
    const savedTheme =
      window.localStorage.getItem(
        "enterprise-copilot-theme"
      );

    if (savedTheme === "dark") {
      setDarkMode(true);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      "enterprise-copilot-theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  const currentDocumentLabel =
    useMemo(() => {
      if (!filename) {
        return "No document selected";
      }

      return filename;
    }, [filename]);

  // ==========================================================
  // Reset
  // ==========================================================

  function resetForNewDocument() {
    setDocumentId(null);
    setFilename("");
    setSelectedFile(null);
    setMessages([]);
    setQuery("");
    setError("");
  }

  // ==========================================================
  // File Selection
  // ==========================================================

  function handleFileSelection(
    file: File | null
  ) {
    if (!file) {
      return;
    }

    setError("");

    if (
      file.type !== "application/pdf" &&
      !file.name
        .toLowerCase()
        .endsWith(".pdf")
    ) {
      setError(
        "Please select a PDF file."
      );
      return;
    }

    if (
      file.size > MAX_FILE_SIZE
    ) {
      setError(
        "PDF size must be 25 MB or smaller."
      );
      return;
    }

    setSelectedFile(file);
  }

  function handleInputChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file =
      event.target.files?.[0] ||
      null;

    handleFileSelection(file);
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    const file =
      event.dataTransfer.files?.[0] ||
      null;

    handleFileSelection(file);
  }

  function handleDragOver(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();
  }

  // ==========================================================
  // Upload PDF
  // ==========================================================

  async function uploadFile(
    file: File | null
  ) {
    if (!file) {
      return;
    }

    setUploading(true);
    setError("");

    setDocumentId(null);
    setMessages([]);
    setQuery("");

    try {
      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      const response =
        await fetch(
          `${API_URL}/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

      let body: any = null;

      try {
        body =
          await response.json();
      } catch {
        body = null;
      }

      if (!response.ok) {
        throw new Error(
          body?.detail ||
            body?.message ||
            "Failed to upload and process the PDF."
        );
      }

      if (!body?.document_id) {
        throw new Error(
          "The server did not return a document ID."
        );
      }

      setDocumentId(
        body.document_id
      );

      setFilename(
        body.filename ||
          file.name
      );

      setSelectedFile(file);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while uploading the PDF."
      );
    } finally {
      setUploading(false);
    }
  }

  async function handleUploadClick() {
    if (!selectedFile) {
      fileInputRef.current?.click();
      return;
    }

    await uploadFile(
      selectedFile
    );
  }

  function handleChangePdf() {
    setError("");
    fileInputRef.current?.click();
  }

  // ==========================================================
  // Ask Question
  // ==========================================================

  async function askQuestion(
    questionOverride?: string
  ) {
    const currentQuery = (
      questionOverride !== undefined
        ? questionOverride
        : query
    ).trim();

    if (
      !currentQuery ||
      !documentId ||
      asking
    ) {
      return;
    }

    setError("");

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: currentQuery,
    };

    setMessages(
      (previous) => [
        ...previous,
        userMessage,
      ]
    );

    setQuery("");
    setAsking(true);

    try {
      const response =
        await fetch(
          `${API_URL}/query`,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body: JSON.stringify({
              document_id:
                documentId,
              query: currentQuery,
            }),
          }
        );

      let body: any = null;

      try {
        body =
          await response.json();
      } catch {
        body = null;
      }

      if (!response.ok) {
        throw new Error(
          body?.detail ||
            body?.message ||
            "Failed to generate an answer."
        );
      }

      const assistantMessage: Message =
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            body?.answer ||
            "I could not generate an answer for this question.",
          status:
            body?.status,
          sources:
            Array.isArray(
              body?.sources
            )
              ? body.sources
              : [],
          sourceEvaluations:
            Array.isArray(
              body?.source_evaluations
            )
              ? body.source_evaluations
              : [],
          rewrittenQuery:
            body?.rewritten_query ||
            null,
          correctiveRag:
            Boolean(
              body?.corrective_rag
            ),
        };

      setMessages(
        (previous) => [
          ...previous,
          assistantMessage,
        ]
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while generating the answer."
      );
    } finally {
      setAsking(false);
    }
  }

  // ==========================================================
  // Keyboard
  // ==========================================================

  function handleComposerKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      void askQuestion();
    }
  }

  // ==========================================================
  // Copy Answer
  // ==========================================================

  async function copyAnswer(
    answer: string
  ) {
    try {
      await navigator.clipboard.writeText(
        answer
      );
    } catch {
      setError(
        "Could not copy the answer."
      );
    }
  }

  // ==========================================================
  // Status Label
  // ==========================================================

  function getStatusLabel(
    status?: string
  ) {
    if (!status) {
      return null;
    }

    if (
      status === "complete"
    ) {
      return "Complete evidence";
    }

    if (
      status === "partial"
    ) {
      return "Partial evidence";
    }

    if (
      status === "not_found"
    ) {
      return "Insufficient evidence";
    }

    return status;
  }

  // ==========================================================
  // Main UI
  // ==========================================================

  return (
    <div
      className={
        darkMode
          ? "dark"
          : ""
      }
    >
      <main className="min-h-screen bg-[#f7f8fa] text-slate-900 transition-colors duration-200 dark:bg-slate-950 dark:text-slate-100">

        <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8">

          {/* ================================================== */}
          {/* Header                                             */}
          {/* ================================================== */}

          <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">

            <div>
              <div className="mb-1 flex items-center gap-2">

                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-sm font-bold text-white dark:bg-white dark:text-slate-900">
                  AI
                </div>

                <span className="text-sm font-semibold uppercase tracking-[0.14em] text-slate-500 dark:text-slate-400">
                  Research Copilot
                </span>

              </div>

              <h1 className="text-2xl font-semibold tracking-tight text-slate-950 dark:text-white sm:text-3xl">
                Chat with any PDF
              </h1>

              <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-400">
                Upload a document and ask questions using
                hybrid retrieval, reranking, corrective RAG,
                and grounded source evidence.
              </p>
            </div>

            <div className="flex items-center gap-2">

              {/* Theme Toggle */}

              <button
                type="button"
                onClick={() =>
                  setDarkMode(
                    (previous) =>
                      !previous
                  )
                }
                aria-label={
                  darkMode
                    ? "Switch to light mode"
                    : "Switch to dark mode"
                }
                title={
                  darkMode
                    ? "Light mode"
                    : "Dark mode"
                }
                className="flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 bg-white text-lg text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                {darkMode
                  ? "☀"
                  : "☾"}
              </button>

              {hasDocument && (
                <button
                  type="button"
                  onClick={
                    resetForNewDocument
                  }
                  className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  New PDF
                </button>
              )}

            </div>
          </header>

          {/* ================================================== */}
          {/* Upload Screen                                      */}
          {/* ================================================== */}

          {!hasDocument ? (

            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 sm:p-8">

              <div className="mx-auto max-w-3xl">

                <div
                  onDrop={
                    handleDrop
                  }
                  onDragOver={
                    handleDragOver
                  }
                  className="rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-5 py-12 text-center transition hover:border-slate-400 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-950 dark:hover:border-slate-600 dark:hover:bg-slate-900 sm:px-10"
                >

                  <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-white text-2xl shadow-sm ring-1 ring-slate-200 dark:bg-slate-900 dark:ring-slate-700">
                    PDF
                  </div>

                  <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                    Upload a PDF
                  </h2>

                  <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-600 dark:text-slate-400">
                    Drop your PDF here or choose a file
                    from your computer. The document will be
                    processed and indexed before you start
                    chatting.
                  </p>

                  <div className="mt-6 flex flex-col items-center justify-center gap-3 sm:flex-row">

                    <button
                      type="button"
                      onClick={() =>
                        fileInputRef.current?.click()
                      }
                      className="rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200"
                    >
                      Choose PDF
                    </button>

                    {selectedFile && (
                      <button
                        type="button"
                        onClick={
                          handleUploadClick
                        }
                        disabled={
                          uploading
                        }
                        className="rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-800 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                      >
                        {uploading
                          ? "Processing PDF..."
                          : "Upload & Process"}
                      </button>
                    )}

                  </div>

                  <input
                    ref={
                      fileInputRef
                    }
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={
                      handleInputChange
                    }
                    className="hidden"
                  />

                  {selectedFile && (
                    <div className="mx-auto mt-6 max-w-lg rounded-xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm dark:border-slate-700 dark:bg-slate-900">

                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                        Selected document
                      </div>

                      <div className="mt-1 truncate text-sm font-medium text-slate-900 dark:text-white">
                        {
                          selectedFile.name
                        }
                      </div>

                      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                        {(
                          selectedFile.size /
                          1024 /
                          1024
                        ).toFixed(2)}{" "}
                        MB
                      </div>

                    </div>
                  )}

                  <p className="mt-5 text-xs text-slate-500 dark:text-slate-500">
                    PDF only · Maximum size 25 MB
                  </p>

                </div>

                {error && (
                  <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
                    {error}
                  </div>
                )}

                <div className="mt-8 grid gap-4 sm:grid-cols-3">

                  <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
                    <div className="text-sm font-semibold text-slate-900 dark:text-white">
                      Hybrid retrieval
                    </div>

                    <p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">
                      Dense vector search combined with BM25
                      lexical retrieval.
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
                    <div className="text-sm font-semibold text-slate-900 dark:text-white">
                      Corrective RAG
                    </div>

                    <p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">
                      Evidence coverage is checked before
                      final answer generation.
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
                    <div className="text-sm font-semibold text-slate-900 dark:text-white">
                      Source evidence
                    </div>

                    <p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">
                      Retrieved passages are available
                      below each answer.
                    </p>
                  </div>

                </div>

              </div>
            </section>

          ) : (

            /* ================================================== */
            /* Chat Screen                                        */
            /* ================================================== */

            <section className="flex min-h-[calc(100vh-190px)] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">

              {/* Document Bar */}

              <div className="border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900 sm:px-6">

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">

                  <div className="min-w-0">
                    <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">
                      Current document
                    </div>

                    <div className="mt-1 truncate text-sm font-semibold text-slate-900 dark:text-white">
                      {
                        currentDocumentLabel
                      }
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={
                      handleChangePdf
                    }
                    disabled={
                      uploading
                    }
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                  >
                    Change PDF
                  </button>

                  <input
                    ref={
                      fileInputRef
                    }
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={
                      async (
                        event
                      ) => {
                        const file =
                          event.target.files?.[0] ||
                          null;

                        if (file) {
                          handleFileSelection(
                            file
                          );

                          await uploadFile(
                            file
                          );
                        }

                        event.target.value =
                          "";
                      }
                    }
                    className="hidden"
                  />

                </div>
              </div>

              {/* Messages */}

              <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">

                {messages.length === 0 ? (

                  <div className="mx-auto max-w-3xl py-16 text-center">

                    <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-sm font-bold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      Q&A
                    </div>

                    <h2 className="mt-5 text-xl font-semibold text-slate-900 dark:text-white">
                      Ask questions about your document
                    </h2>

                    <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-600 dark:text-slate-400">
                      Ask for facts, comparisons, trends,
                      explanations, figures, or other
                      information contained in the uploaded PDF.
                    </p>

                    <div className="mt-8 grid gap-3 sm:grid-cols-2">

                      {[
                        "What are the key financial highlights?",
                        "What factors affected revenue or margins?",
                        "What were the major risks mentioned?",
                        "Summarize the most important findings.",
                      ].map(
                        (example) => (
                          <button
                            key={
                              example
                            }
                            type="button"
                            onClick={() =>
                              void askQuestion(
                                example
                              )
                            }
                            className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm text-slate-700 transition hover:border-slate-300 hover:bg-white dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:border-slate-700 dark:hover:bg-slate-900"
                          >
                            {
                              example
                            }
                          </button>
                        )
                      )}

                    </div>
                  </div>

                ) : (

                  <div className="mx-auto max-w-4xl space-y-6">

                    {messages.map(
                      (message) => (
                        <article
                          key={
                            message.id
                          }
                        >

                          {/* User */}

                          {message.role ===
                          "user" ? (

                            <div className="flex justify-end">

                              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-slate-900 px-4 py-3 text-sm leading-6 text-white dark:bg-white dark:text-slate-900">
                                {
                                  message.content
                                }
                              </div>

                            </div>

                          ) : (

                            /* Assistant */

                            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950 sm:p-5">

                              {/* Assistant Header */}

                              <div className="flex flex-wrap items-center justify-between gap-3">

                                <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500 dark:text-slate-400">
                                  Assistant
                                </div>

                                <div className="flex items-center gap-2">

                                  {getStatusLabel(
                                    message.status
                                  ) && (
                                    <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                                      {
                                        getStatusLabel(
                                          message.status
                                        )
                                      }
                                    </span>
                                  )}

                                  <button
                                    type="button"
                                    onClick={() =>
                                      void copyAnswer(
                                        message.content
                                      )
                                    }
                                    className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
                                  >
                                    Copy
                                  </button>

                                </div>
                              </div>

                              {/* Markdown Answer */}

                              <div className="prose mt-4 max-w-none text-sm leading-7 text-slate-700 dark:text-slate-300">

                                <ReactMarkdown
                                  components={{
                                    p: ({
                                      children,
                                    }) => (
                                      <p className="mb-3 last:mb-0">
                                        {
                                          children
                                        }
                                      </p>
                                    ),

                                    ul: ({
                                      children,
                                    }) => (
                                      <ul className="mb-3 ml-5 list-disc space-y-1">
                                        {
                                          children
                                        }
                                      </ul>
                                    ),

                                    ol: ({
                                      children,
                                    }) => (
                                      <ol className="mb-3 ml-5 list-decimal space-y-1">
                                        {
                                          children
                                        }
                                      </ol>
                                    ),

                                    li: ({
                                      children,
                                    }) => (
                                      <li className="leading-7">
                                        {
                                          children
                                        }
                                      </li>
                                    ),

                                    strong: ({
                                      children,
                                    }) => (
                                      <strong className="font-semibold text-slate-900 dark:text-white">
                                        {
                                          children
                                        }
                                      </strong>
                                    ),

                                    em: ({
                                      children,
                                    }) => (
                                      <em className="italic">
                                        {
                                          children
                                        }
                                      </em>
                                    ),

                                    h1: ({
                                      children,
                                    }) => (
                                      <h1 className="mb-2 mt-4 text-xl font-semibold text-slate-900 first:mt-0 dark:text-white">
                                        {
                                          children
                                        }
                                      </h1>
                                    ),

                                    h2: ({
                                      children,
                                    }) => (
                                      <h2 className="mb-2 mt-4 text-lg font-semibold text-slate-900 first:mt-0 dark:text-white">
                                        {
                                          children
                                        }
                                      </h2>
                                    ),

                                    h3: ({
                                      children,
                                    }) => (
                                      <h3 className="mb-2 mt-4 text-base font-semibold text-slate-900 first:mt-0 dark:text-white">
                                        {
                                          children
                                        }
                                      </h3>
                                    ),

                                    blockquote: ({
                                      children,
                                    }) => (
                                      <blockquote className="my-3 border-l-4 border-slate-300 pl-4 italic text-slate-600 dark:border-slate-700 dark:text-slate-400">
                                        {
                                          children
                                        }
                                      </blockquote>
                                    ),

                                    code: ({
                                      children,
                                    }) => (
                                      <code className="rounded bg-slate-200 px-1.5 py-0.5 text-[0.9em] text-slate-800 dark:bg-slate-800 dark:text-slate-200">
                                        {
                                          children
                                        }
                                      </code>
                                    ),
                                  }}
                                >
                                  {
                                    message.content
                                  }
                                </ReactMarkdown>

                              </div>

                              {/* Corrective RAG */}

                              {message.correctiveRag && (
                                <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 dark:border-amber-900/60 dark:bg-amber-950/30">

                                  <div className="text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
                                    Corrective retrieval used
                                  </div>

                                  {message.rewrittenQuery && (
                                    <div className="mt-1 text-sm text-amber-900 dark:text-amber-300">
                                      <span className="font-medium">
                                        Rewritten query:
                                      </span>{" "}
                                      {
                                        message.rewrittenQuery
                                      }
                                    </div>
                                  )}

                                </div>
                              )}

                              {/* Collapsed Sources */}

                              {message.sources &&
                                message.sources.length >
                                  0 && (

                                  <details className="mt-6 overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">

                                    <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 dark:text-slate-200 dark:hover:bg-slate-800">

                                      <div className="flex items-center justify-between gap-3">

                                        <span>
                                          Sources ·{" "}
                                          {
                                            message
                                              .sources
                                              .length
                                          }
                                        </span>

                                        <span className="text-xs font-medium text-slate-400">
                                          Click to expand
                                        </span>

                                      </div>

                                    </summary>

                                    <div className="border-t border-slate-200 p-4 dark:border-slate-800">

                                      <div className="space-y-3">

                                        {message.sources.map(
                                          (source) => (

                                            <div
                                              key={`${message.id}-${source.source_id}`}
                                              className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950"
                                            >

                                              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">

                                                <div>
                                                  <div className="text-sm font-semibold text-slate-900 dark:text-white">
                                                    Source{" "}
                                                    {
                                                      source.source_id
                                                    }
                                                  </div>

                                                  <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                                                    {
                                                      source.document ||
                                                      "Document"
                                                    }

                                                    {source.page !==
                                                      null &&
                                                      source.page !==
                                                        undefined && (
                                                        <>
                                                          {" "}
                                                          · Page{" "}
                                                          {
                                                            source.page
                                                          }
                                                        </>
                                                      )}
                                                  </div>
                                                </div>

                                                {source.score !==
                                                  null &&
                                                  source.score !==
                                                    undefined && (
                                                    <span className="text-[11px] font-medium text-slate-400">
                                                      Score{" "}
                                                      {Number(
                                                        source.score
                                                      ).toFixed(
                                                        4
                                                      )}
                                                    </span>
                                                  )}

                                              </div>

                                              <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600 dark:text-slate-400">
                                                {
                                                  source.text
                                                }
                                              </p>

                                            </div>

                                          )
                                        )}

                                      </div>

                                    </div>
                                  </details>
                                )}

                              {/* Collapsed Retrieval Details */}

                              {message.sourceEvaluations &&
                                message.sourceEvaluations.length >
                                  0 && (

                                  <details className="mt-4 overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">

                                    <summary className="cursor-pointer list-none px-4 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500 transition hover:bg-slate-50 dark:text-slate-400 dark:hover:bg-slate-800">

                                      <div className="flex items-center justify-between gap-3">

                                        <span>
                                          Retrieval details
                                        </span>

                                        <span className="text-xs font-medium normal-case tracking-normal text-slate-400">
                                          Advanced
                                        </span>

                                      </div>

                                    </summary>

                                    <div className="border-t border-slate-200 p-4 dark:border-slate-800">

                                      <div className="overflow-x-auto">

                                        <table className="min-w-full text-left text-xs">

                                          <thead>
                                            <tr className="border-b border-slate-200 dark:border-slate-800">

                                              <th className="px-2 py-2 font-semibold text-slate-500 dark:text-slate-400">
                                                Rank
                                              </th>

                                              <th className="px-2 py-2 font-semibold text-slate-500 dark:text-slate-400">
                                                Relevant
                                              </th>

                                              <th className="px-2 py-2 font-semibold text-slate-500 dark:text-slate-400">
                                                Method
                                              </th>

                                              <th className="px-2 py-2 font-semibold text-slate-500 dark:text-slate-400">
                                                Node
                                              </th>

                                            </tr>
                                          </thead>

                                          <tbody>

                                            {message.sourceEvaluations.map(
                                              (
                                                evaluation,
                                                index
                                              ) => (

                                                <tr
                                                  key={`${message.id}-eval-${index}`}
                                                  className="border-b border-slate-100 last:border-0 dark:border-slate-800"
                                                >

                                                  <td className="px-2 py-2 text-slate-700 dark:text-slate-300">
                                                    {
                                                      evaluation.retrieved_rank ??
                                                      "-"
                                                    }
                                                  </td>

                                                  <td className="px-2 py-2 text-slate-700 dark:text-slate-300">
                                                    {
                                                      evaluation.relevant
                                                        ? "Yes"
                                                        : "No"
                                                    }
                                                  </td>

                                                  <td className="px-2 py-2 text-slate-700 dark:text-slate-300">
                                                    {
                                                      evaluation.evaluation_method ||
                                                      "-"
                                                    }
                                                  </td>

                                                  <td className="max-w-[240px] truncate px-2 py-2 font-mono text-[11px] text-slate-500 dark:text-slate-400">
                                                    {
                                                      evaluation.node_id ||
                                                      "-"
                                                    }
                                                  </td>

                                                </tr>

                                              )
                                            )}

                                          </tbody>

                                        </table>

                                      </div>

                                    </div>

                                  </details>
                                )}

                            </div>
                          )}
                        </article>
                    )
                  )}

                  {/* Loading */}

                  {asking && (
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950">

                      <div className="flex items-center gap-3 text-sm text-slate-600 dark:text-slate-400">

                        <div className="flex gap-1">

                          <span className="h-2 w-2 animate-pulse rounded-full bg-slate-400" />

                          <span className="h-2 w-2 animate-pulse rounded-full bg-slate-400 [animation-delay:120ms]" />

                          <span className="h-2 w-2 animate-pulse rounded-full bg-slate-400 [animation-delay:240ms]" />

                        </div>

                        Searching the document and generating a grounded answer...

                      </div>

                    </div>
                  )}

                  {/* Error */}

                  {error && (
                    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
                      {
                        error
                      }
                    </div>
                  )}

                </div>
              )}
            </div>

            {/* Composer */}

            <div className="border-t border-slate-200 bg-white px-4 py-4 dark:border-slate-800 dark:bg-slate-900 sm:px-6">

              <div className="mx-auto max-w-4xl">

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-2 shadow-sm dark:border-slate-700 dark:bg-slate-950">

                  <textarea
                    value={query}
                    onChange={(
                      event
                    ) =>
                      setQuery(
                        event.target.value
                      )
                    }
                    onKeyDown={
                      handleComposerKeyDown
                    }
                    rows={3}
                    placeholder="Ask a question about the uploaded PDF..."
                    disabled={
                      asking ||
                      uploading
                    }
                    className="w-full resize-none border-0 bg-transparent px-3 py-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
                  />

                  <div className="flex items-center justify-between gap-3 px-2 pb-1 pt-2">

                    <div className="text-xs text-slate-400">
                      Enter to send · Shift+Enter for a new line
                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        void askQuestion()
                      }
                      disabled={
                        asking ||
                        !query.trim() ||
                        !documentId
                      }
                      className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200"
                    >
                      {asking
                        ? "Thinking..."
                        : "Ask"}
                    </button>

                  </div>

                </div>

              </div>

            </div>

          </section>
        )}
      </div>
    </main>
    </div>
  );
}