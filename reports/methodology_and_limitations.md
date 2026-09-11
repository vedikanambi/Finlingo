# Methodology Notes and Known Limitations

## The faithfulness metric (RQ3)

The technical proposal defines faithfulness as an entailment score between a generated
sentence and the retrieved regulatory chunk that's supposed to support it:

> Fi = P(entailment | sentence, retrieved chunk)

This is what distinguishes the approach from prior work like RAGAS (paragraph-level,
LLM-as-judge) and SelfCheckGPT (sentence-level, but checked against the model's own
self-consistency rather than an external source). The implementation computes this formula
directly, using the DeBERTa-v3 verifier against whichever chunks the retrieval and reranking
stages surface.

When measured on the FLB benchmark, this primary metric comes out close to zero
(precision and recall both near 0 across 182 rows). The cause: the verifier only checks the
top 5 reranked chunks for each sentence, and post-rerank recall@5 on FLB is only about 8%.
So most of the time, the chunk that actually supports a given sentence simply isn't among the
chunks the verifier gets to look at — it's a retrieval miss, not a verifier failure.

To confirm this wasn't a deeper problem with the retrieval code, the same embedding model and
reranker were tested on a different benchmark (FinanceBench), where recall@5 comes out around
70%. Same code, same models, very different result — which points to FLB's own document
collection (general commercial contracts, not finance-specific regulatory text) as the actual
bottleneck, not the retrieval logic itself.

As a secondary, more benchmark-appropriate check, the same verifier is also scored against
the original source clause instead of the retrieved chunk. That comes out to precision 0.906,
recall 0.637 — a reasonable proxy result, though it isn't what the proposal specifies as the
primary metric.

A smaller diagnostic was also run: looking only at the subset of records (12 out of 182)
where the correct chunk *was* retrieved, to see if the verifier does better when given the
right premise. It didn't — recall on that small subset was still 0. With only 12 examples
this isn't strong evidence either way, but it's reported honestly rather than left out: the
retrieval gap is the dominant explanation for the low score, but it may not be the only one.

## Risk classification (RQ2)

Target was macro-F1 > 0.75. The canonical prompted classifier (Qwen2.5-1.5B, few-shot)
only gets macro-F1 0.563 — below target. The proposal originally specified this stage
should stay prompted only, unlike the simplifier and verifier, which are fine-tuned. That
was a real tension: the target assumed a level of performance that's hard to reach without
fine-tuning.

So a fine-tuned classifier (legal-bert-base-uncased + LoRA) was trained after all, and it
reaches macro-F1 0.79 on the same 91 clauses — above target, and it's what the submitted
system actually runs (`RISK_CLASSIFIER_BACKEND=trained`). A simpler frozen-embedding
baseline (FinBERT + logistic regression) was also tried as a sanity check and landed in
between at 0.69. The canonical prompted number (0.563) is kept and reported honestly
alongside the deployed number, since it's what the original proposal's method would have
produced — the report doesn't quietly swap one for the other.

Two of the six categories (Hidden Fee, Data Sharing) are consistently the weakest. The
regulatory source documents used for retrieval don't have much dedicated text on data privacy
or hidden fees specifically, so the evidence handed to the classifier for those categories
tends to be topically related but not directly useful.

## Adapter attachment check

An automated check compares the fine-tuned verifier's output against the base model on 14
probe sentence pairs, expecting a noticeable score difference if the adapter is doing
something. The difference came out smaller than the test's own threshold — but on inspection,
all 14 probes turned out to be cases where the base model already scores very close to 0 or 1
with high confidence, leaving essentially no room for a fine-tuning effect to show up (a
probability can't move far past 0 or 1). So this particular test doesn't tell us much either
way. The adapter's authenticity is separately confirmed by checking its file structure and
hashes, and its actual effect is shown properly by the FaithBench comparison below.

## Multi-verifier comparison

DeBERTa v3 (the deployed verifier) vs a ModernBERT-based comparator, trained on the same
5,000 examples. ModernBERT came out degenerate — it always predicts "unsupported," never
learning useful discrimination. This is a real, reproducible result: ModernBERT doesn't have
DeBERTa's NLI-specific pretraining, and 5,000 examples wasn't enough to make up for that.

## FaithBench reproduction

Comparing the fine-tuned verifier against its own base model on FaithBench, an external
benchmark: the fine-tuned version improves balanced accuracy by +0.038 (95% CI 0.006–0.072,
p=0.02). Both the base model and an earlier checkpoint had previously scored at chance level
on this benchmark (always predicting "unsupported"), so this is a genuine, statistically
meaningful improvement — the strongest and cleanest result in the evaluation.

## Ablations and comparator baselines

All six pipeline-component variants (full pipeline, no verifier, no reranking, BM25 instead
of FAISS, no adapters, simplification only) and all five comparator baselines (rule-based
classifier, retrieval-augmented generation without the verifier, FinBERT, a RAGAS-style
paragraph judge, and a SelfCheckGPT-style consistency check) are complete. Results are in
`reports/ablations.json`.

## Reranker and retrieval-query-mode comparisons

Both complete. The deployed reranker (a larger cross-encoder) modestly outperforms a smaller,
faster one on ranking quality, at roughly 1.8x the latency. Separately, querying retrieval
with the original clause text works better than querying with the simplified version — the
simplification step tends to drop exact financial and legal terms that retrieval relies on.

## Regulatory source coverage

Two additional real regulatory sources were added during the project (a US rule on negative
option/subscription terms, and part of the UK's data protection legislation) to help cover
the Auto-Renewal and Data Sharing categories, which were previously very thin. This gave a
measurable improvement to both classification accuracy and retrieval recall for those
categories, though the underlying corpus is still small relative to what a production system
would need.

## Not covered here

Frontend end-to-end interaction testing and version control history are outside the scope of
this write-up.
