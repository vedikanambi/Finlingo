import { memo, useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink, AlertTriangle, CheckCircle2 } from 'lucide-react'

const pct = (value) => (value == null ? 'N/A' : `${Math.round(value * 100)}%`)

function ClauseCard({ clause }) {
  const [open, setOpen] = useState(false)
  const risky = !['Safe', 'Needs Review'].includes(clause.risk_label)
  const simplificationNeedsReview = clause.simplification_status === 'needs_review'
  return (
    <article className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full p-4 text-left flex items-center justify-between gap-4"
      >
        <div>
          <div className="flex flex-wrap gap-2 items-center">
            <span className="font-semibold">{clause.clause_id}</span>
            <span
              className={`badge ${risky ? 'badge-risk' : clause.risk_label === 'Safe' ? 'badge-safe' : 'badge-review'}`}
            >
              {clause.risk_label}
            </span>
            <span className="text-xs text-gray-500">risk {clause.risk_score}/5</span>
            {simplificationNeedsReview && (
              <span className="badge badge-review">simplification review</span>
            )}
          </div>
          <p className="text-sm text-gray-300 mt-2 line-clamp-2">{clause.simplified_text}</p>
        </div>
        {open ? <ChevronUp /> : <ChevronDown />}
      </button>
      {open && (
        <div className="border-t border-gray-800 p-4 space-y-5">
          <Section title="Original clause">
            <p>{clause.original_text}</p>
            <p className="muted">
              Parser: {clause.extraction_method || 'N/A'} · page {clause.page_number ?? 'N/A'}
            </p>
          </Section>
          <Section title="Plain-language version">
            <p>{clause.simplified_text}</p>
            <p className="muted">
              FK grade: {clause.readability_grade ?? 'N/A'} · semantic preservation:{' '}
              {pct(clause.semantic_preservation_score)} · attempts:{' '}
              {clause.simplification_attempts ?? 'N/A'}
            </p>
            {simplificationNeedsReview && (
              <p className="text-amber-300 mt-2">
                This candidate did not satisfy both readability and semantic-preservation gates.
              </p>
            )}
          </Section>
          <Section title="Risk decision">
            <p>{clause.risk_explanation}</p>
            <p className="muted">
              Confidence: {pct(clause.risk_confidence)} · regulatory support:{' '}
              {pct(clause.mean_risk_regulatory_support)}
            </p>
            {(clause.secondary_risk_labels || []).length > 0 && (
              <p className="muted">Secondary risks: {clause.secondary_risk_labels.join(', ')}</p>
            )}
            {(clause.risk_explanation_support || []).map((item, index) => (
              <div key={index} className="rounded-lg bg-gray-950 p-3 mt-2">
                <div className="flex gap-2">
                  {item.supported ? (
                    <CheckCircle2 className="text-emerald-400" size={17} />
                  ) : (
                    <AlertTriangle className="text-red-400" size={17} />
                  )}
                  <p>{item.sentence}</p>
                </div>
                <p className="muted">Regulatory support: {pct(item.regulatory_support)}</p>
                {item.attribution && <EvidenceLink chunk={item.attribution} />}
              </div>
            ))}
          </Section>
          <Section title="Sentence-level verification">
            {(clause.faithfulness || []).length === 0 ? (
              <p className="muted">Verifier not run.</p>
            ) : (
              (clause.faithfulness || []).map((item, index) => (
                <div key={index} className="rounded-lg bg-gray-950 p-3 mb-2">
                  <div className="flex gap-2">
                    {item.unsupported ? (
                      <AlertTriangle className="text-red-400" size={17} />
                    ) : (
                      <CheckCircle2 className="text-emerald-400" size={17} />
                    )}
                    <p>{item.sentence}</p>
                  </div>
                  <div className="text-xs text-gray-400 mt-2">
                    Primary Fi ({item.faithfulness_premise}): {pct(item.faithfulness_score)} ·
                    source preservation: {pct(item.source_faithfulness)} · retrieved-chunk support:{' '}
                    {pct(item.regulatory_support)}
                  </div>
                  {item.unsupported_reason && (
                    <p className="text-xs text-amber-300 mt-1">{item.unsupported_reason}</p>
                  )}
                  {item.attribution ? (
                    <EvidenceLink chunk={item.attribution} />
                  ) : (
                    item.attribution_candidate && (
                      <p className="text-xs text-gray-500 mt-2">
                        Best candidate {item.attribution_candidate.chunk_id} scored{' '}
                        {pct(item.attribution_candidate_score)} but did not meet the attribution
                        threshold.
                      </p>
                    )
                  )}
                </div>
              ))
            )}
          </Section>
          <Section title="Retrieved evidence">
            {(clause.evidence || []).length === 0 ? (
              <p className="muted">No evidence passed the retrieval/reranking gate.</p>
            ) : (
              (clause.evidence || []).map((chunk) => (
                <div key={chunk.chunk_id} className="rounded-lg bg-gray-950 p-3 mb-2">
                  <EvidenceLink chunk={chunk} />
                  <p className="text-xs text-gray-500 mt-1">
                    {chunk.chunk_id} · retrieval {chunk.retrieval_score?.toFixed?.(4) ?? 'N/A'} ·
                    rerank {chunk.rerank_score?.toFixed?.(4) ?? 'N/A'}
                  </p>
                  <p className="text-sm text-gray-300 mt-2">{chunk.text}</p>
                </div>
              ))
            )}
          </Section>
        </div>
      )}
    </article>
  )
}

export default memo(ClauseCard)

function EvidenceLink({ chunk }) {
  return (
    <a
      className="text-xs text-sky-400 inline-flex items-center gap-1 mt-2"
      href={chunk.source_url}
      target="_blank"
      rel="noreferrer"
    >
      {chunk.source} {chunk.section || ''} · {chunk.chunk_id}
      <ExternalLink size={12} />
    </a>
  )
}

function Section({ title, children }) {
  return (
    <section>
      <h3 className="text-sm font-semibold text-gray-200 mb-2">{title}</h3>
      <div className="text-sm text-gray-300">{children}</div>
    </section>
  )
}
