import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Bot, Menu, Send, Sparkles, User, X } from "lucide-react";
import { Fragment, useState, type ReactNode } from "react";
import { z } from "zod";
import { ApiError, fetchHeatlensAnswer, fetchWardDetail, type HeatlensResponse } from "@/lib/api";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

// Lightweight renderer for HeatLens AI's structured Markdown-style
// answers (### headers, - bullets, **bold**). Intentionally not a
// full Markdown library - the model's output format is constrained
// by the system prompt to just these three constructs.
function renderInline(text: string, keyPrefix: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <b key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</b>
    ) : (
      <Fragment key={`${keyPrefix}-${i}`}>{part}</Fragment>
    ),
  );
}

function MarkdownLite({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let bulletBuffer: string[] = [];

  const flushBullets = (key: string) => {
    if (bulletBuffer.length === 0) return;
    blocks.push(
      <ul key={key}>
        {bulletBuffer.map((item, i) => <li key={i}>{renderInline(item, `${key}-${i}`)}</li>)}
      </ul>,
    );
    bulletBuffer = [];
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    const key = `line-${index}`;

    if (!line) {
      flushBullets(`${key}-bullets`);
      return;
    }

    if (line.startsWith("### ")) {
      flushBullets(`${key}-bullets`);
      blocks.push(<h4 key={key}>{line.slice(4)}</h4>);
      return;
    }

    if (line.startsWith("- ") || line.startsWith("* ")) {
      bulletBuffer.push(line.slice(2));
      return;
    }

    flushBullets(`${key}-bullets`);
    blocks.push(<p key={key}>{renderInline(line, key)}</p>);
  });

  flushBullets("trailing-bullets");

  return <div className="heatlens-answer">{blocks}</div>;
}

const searchSchema = z.object({
  ward: z.string().optional(),
});

export const Route = createFileRoute("/heatlens")({
  validateSearch: searchSchema,
  head: () => ({
    meta: [
      { title: "HeatLens AI" },
      { name: "description", content: "Ask HeatLens AI about Bengaluru's heat analytics, grounded in the connected datasets." },
    ],
  }),
  component: HeatlensPage,
});

interface ConversationEntry {
  question: string;
  response?: HeatlensResponse;
  error?: string;
}

const CITY_QUESTIONS = [
  "Which wards have persistent heat pockets?",
  "How many wards show a statistically significant rising heat trend?",
  "What does future heat probability mean?",
  "What does persistence mean?",
];

function WardQuestions(wardLabel: string) {
  return [
    `Why is ${wardLabel} high priority?`,
    `Explain the historical trend for ${wardLabel}.`,
    `Why are these interventions recommended for ${wardLabel}?`,
  ];
}

function HeatlensPage() {
  const { ward } = Route.useSearch();
  const navigate = Route.useNavigate();
  const toggleSidebar = useSidebarToggle();
  const [input, setInput] = useState("");
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);

  const wardContext = useQuery({
    queryKey: ["wardDetail", ward],
    queryFn: () => fetchWardDetail(ward as string),
    enabled: Boolean(ward),
    retry: false,
  });

  const mutation = useMutation({
    mutationFn: (question: string) => fetchHeatlensAnswer(question, ward),
  });

  const ask = (question: string) => {
    if (!question.trim()) return;
    setInput("");
    mutation.mutate(question, {
      onSuccess: (response) => setConversation((c) => [...c, { question, response }]),
      onError: (error) =>
        setConversation((c) => [
          ...c,
          { question, error: error instanceof ApiError ? error.detail || error.message : "HeatLens is unavailable right now." },
        ]),
    });
  };

  const wardLabel = wardContext.data ? `${wardContext.data.ward.ward_name} (${wardContext.data.ward.ward_key})` : ward;
  const suggested = ward ? WardQuestions(wardLabel ?? ward) : CITY_QUESTIONS;

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <h1>HeatLens AI</h1>
          <p>Explain the data. Ask questions grounded in the connected risk, persistence, trend, XAI, and policy datasets.</p>
        </div>
      </header>

      {ward && (
        <div className="heatlens-context">
          <Sparkles />
          <span>Ward context: <strong>{wardLabel}</strong></span>
          <button type="button" onClick={() => navigate({ search: {} })}><X /></button>
        </div>
      )}

      <div className="heatlens-layout">
        <section className="heatlens-conversation">
          {conversation.length === 0 && (
            <div className="heatlens-empty">
              <Bot />
              <p>Ask a question about Bengaluru's heat analytics — city-wide, or about a specific ward.</p>
            </div>
          )}
          {conversation.map((entry, index) => (
            <div key={index} className="heatlens-turn">
              <div className="heatlens-bubble user"><User /><p>{entry.question}</p></div>
              <div className="heatlens-bubble assistant">
                <Bot />
                <div>
                  {entry.response ? (
                    <>
                      <MarkdownLite text={entry.response.answer} />
                      {entry.response.ward_key && <small>Detected ward: {entry.response.ward_key}</small>}
                    </>
                  ) : (
                    <p className="heatlens-error">{entry.error}</p>
                  )}
                </div>
              </div>
            </div>
          ))}
          {mutation.isPending && (
            <div className="heatlens-turn">
              <div className="heatlens-bubble user"><User /><p>{mutation.variables}</p></div>
              <div className="heatlens-bubble assistant"><Bot /><p className="heatlens-loading">Thinking, grounded in the connected datasets…</p></div>
            </div>
          )}
        </section>

        <aside className="heatlens-suggested">
          <h2>Suggested questions</h2>
          {suggested.map((question) => (
            <button key={question} type="button" onClick={() => ask(question)}>{question}</button>
          ))}
        </aside>
      </div>

      <form
        className="heatlens-input"
        onSubmit={(event) => {
          event.preventDefault();
          ask(input);
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={ward ? `Ask about ${wardLabel}…` : "Ask HeatLens AI…"}
        />
        <button type="submit" aria-label="Send" disabled={mutation.isPending}><Send /></button>
      </form>
    </AppShell>
  );
}
