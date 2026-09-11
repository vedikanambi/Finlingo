import { Download, RotateCcw, AlertTriangle } from 'lucide-react'
import ClauseCard from './ClauseCard.jsx'

const pct = (value) => (value == null ? 'Not measured' : `${Math.round(value * 100)}%`)
const number = (value) => (value == null ? 'Not measured' : Number(value).toFixed(2))
const compactHash = (value) => (value ? `${value.slice(0, 12)}…` : 'Not recorded')

export default function ResultsPanel({ result, onReset }) {
  const metrics = result?.metrics || {}
  const clauses = result?.clauses || []
  const models = result?.models || {}
  const download = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${(result.filename || 'document').replace(/\.[^.]+$/, '')}_finlingo_report.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="text-gray-100">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold">FinLingo++ report</h1>
          <p className="text-gray-400 mt-1">{result.filename}</p>
        </div>
        <div className="flex gap-2">
          <button onClick={download} className="btn">
            <Download size={16} />
            Export JSON
          </button>
          <button onClick={onReset} className="btn">
            <RotateCcw size={16} />
            New document
          </button>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        <Metric label="Clauses" value={metrics.clause_count ?? clauses.length} />
        <Metric label="Risky clauses" value={metrics.risky_clause_count} />
        <Metric label="Needs risk review" value={metrics.needs_review_count} />
        <Metric label="Simplifications to review" value={metrics.simplification_review_count} />
        <Metric
          label="Primary retrieved-chunk support"
          value={pct(metrics.mean_primary_faithfulness)}
        />
        <Metric label="Source-clause preservation" value={pct(metrics.mean_source_faithfulness)} />
        <Metric
          label="Supported attribution coverage"
          value={pct(metrics.supported_attribution_coverage)}
        />
        <Metric label="Unsupported flag rate" value={pct(metrics.unsupported_flag_rate)} />
        <Metric
          label="Risk explanation support"
          value={pct(metrics.mean_risk_regulatory_support)}
        />
        <Metric label="Retrieval coverage" value={pct(metrics.retrieval_coverage)} />
        <Metric label="Mean FK grade" value={number(metrics.mean_fk_grade)} />
        <Metric label="Semantic gate passed" value={pct(metrics.pct_semantic_target_met)} />
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 mb-6">
        <h2 className="font-semibold mb-2">Run provenance</h2>
        <div className="grid md:grid-cols-2 gap-1 text-sm text-gray-300">
          <span>
            Simplifier: {models.simplifier_model}
            {models.simplifier_adapter
              ? ` + ${models.simplifier_adapter}`
              : ' (no adapter reported)'}
          </span>
          <span>
            Verifier: {models.verifier_model}
            {models.verifier_adapter ? ` + ${models.verifier_adapter}` : ' (no adapter reported)'}
          </span>
          <span>
            Embedding: {models.embedding_model} ({models.embedding_dimensions} dimensions)
          </span>
          <span>Reranker: {models.reranker_model}</span>
          <span>Primary premise: {models.faithfulness_primary_premise || 'Not recorded'}</span>
          <span>
            Faithfulness τ / attribution τ: {models.faithfulness_tau} / {models.attribution_tau}
          </span>
          <span>Document SHA-256: {compactHash(models.document_sha256)}</span>
          <span>Regulatory corpus SHA-256: {compactHash(models.regulatory_corpus_sha256)}</span>
          <span>Config fingerprint: {models.config_fingerprint}</span>
        </div>
      </div>

      {(result.warnings || []).map((warning, index) => (
        <div key={index} className="warning">
          <AlertTriangle size={18} />
          {warning}
        </div>
      ))}
      <div className="space-y-4 mt-5">
        {clauses.map((clause) => (
          <ClauseCard key={clause.clause_id} clause={clause} />
        ))}
      </div>

      <div className="mt-8 rounded-xl border border-amber-800/50 bg-amber-950/30 p-4 text-sm text-amber-200 flex gap-3">
        <AlertTriangle size={20} className="shrink-0" />
        <p>{result.disclaimer}</p>
      </div>
    </div>
  )
}

function Metric({ label, value }) {
  const missing = value == null || value === 'Not measured'
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${missing ? 'text-gray-500' : 'text-white'}`}>
        {value ?? 'Not measured'}
      </div>
    </div>
  )
}
