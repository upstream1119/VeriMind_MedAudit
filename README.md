# MedAudit-RAG

[中文说明](./README.zh-CN.md)

Rule-aware evidence auditing RAG system for safer pediatric medication QA.

MedAudit-RAG is a research prototype for auditing whether pediatric medication answers are supported by guideline evidence. It is not designed to replace doctors, generate prescriptions, or provide clinical treatment advice. The system retrieves evidence, generates constrained answers, audits answer faithfulness, and uses a TrustScore gate to decide whether to answer, refuse, or request human review.

> Status: research prototype. The current implementation is a vector RAG + TrustScore baseline with FastAPI, LangGraph, ChromaDB, and a React frontend. Graph-enhanced evidence auditing and expert validation are planned future work.

## What It Does

- Routes pediatric medication questions into `DETAIL`, `CONCEPT`, and `CONTEXT` intent types.
- Retrieves evidence from multi-granularity guideline indexes.
- Generates answers only from retrieved evidence.
- Audits retrieval relevance, answer faithfulness, and source authority.
- Applies TrustScore gating for supported answers, review-required cases, insufficient evidence, and boundary refusal.
- Displays answer status, TrustScore breakdown, citations, source pages, and evidence snippets in the frontend.

## Why It Matters

Pediatric medication QA is a high-risk and low-tolerance scenario. Incorrect dosage, frequency, route, age or weight boundary, off-label usage, and drug-combination claims may lead to unsafe suggestions. Standard RAG can reduce hallucination, but it can still retrieve weak evidence, repeat irrelevant snippets, or generate confident answers unsupported by the retrieved content.

MedAudit-RAG focuses on answer auditing, not answer generation alone.

## Architecture

```text
User Query
    |
    v
Router
    |
    v
Retriever
    |
    v
Constrained Generator
    |
    v
Evidence Auditor
    |
    v
TrustScore Gate
    |
    +--> answer_supported
    +--> review_required
    +--> insufficient_evidence
    +--> boundary_refusal
```

TrustScore is based on retrieval relevance, answer faithfulness, and source authority:

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

## Tech Stack

- Backend: FastAPI, Python
- RAG workflow: LangGraph
- Vector database: ChromaDB
- Frontend: React, Ant Design, Vite
- Streaming: Server-Sent Events
- Testing: pytest

## Current Repository Scope

The repository currently includes:

- backend API routes for health checks, audit queries, and SSE streaming
- router, retriever, generator, and auditor nodes
- TrustScore calculation and source-authority weighting
- guideline source admission scripts and manifest tracking
- React frontend for displaying audit status and evidence
- unit tests for parser, source preparation, and TrustScore behavior

The current baseline does not claim completed GraphRAG, clinical deployment, or expert medical validation.

## Knowledge Base and Source Admission

Guideline PDFs are not committed to Git. Sources are tracked through a manifest before entering the formal index.

Formal index path:

```text
backend/data/chroma_db/
```

Index status:

```text
backend/data/chroma_db/index_status.json
```

Source manifest:

```text
data/guidelines/source_manifest.json
```

## Quick Start

Install backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Run backend tests:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

Rebuild the vector index:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

Start the backend:

```powershell
$env:PYTHONPATH='backend'
python -m uvicorn app.main:app --reload
```

Start the frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

## API and Output Shape

Main endpoints:

```text
GET  /api/health
POST /api/audit/query
POST /api/audit/query/stream
```

The audit response is designed to expose:

- normalized query and intent type
- answer text or refusal message
- TrustScore and score breakdown
- retrieved evidence snippets
- citation source and page
- final decision such as supported answer, review required, insufficient evidence, or boundary refusal

## Evaluation Plan

The planned benchmark is guideline-grounded pediatric medication safety QA. Each sample should include gold evidence, expected decision, allowed answer scope, and forbidden claims.

Planned metrics include:

- hallucination rate
- unsupported claim rate
- unsafe suggestion rate
- refusal correctness
- claim-evidence alignment precision and recall
- evidence-source mismatch rate

Any future claim about reducing these errors should be backed by raw outputs, audit traces, confidence intervals, and statistical tests.

## Medical Safety Boundary

This project is for research, method validation, and medical evidence auditing only.

It does not provide clinical diagnosis, individualized prescription, or treatment advice. All generated medical content must be grounded in retrieved guideline, consensus, catalog, or drug-label evidence. When evidence is insufficient, incomplete, mismatched, or outside the allowed answer boundary, the system should refuse or request human review.

## Roadmap

- Expand public and authoritative pediatric medication sources.
- Build a guideline-grounded benchmark with gold evidence.
- Compare vanilla LLM, naive RAG, multi-granularity RAG, TrustScore Gate, and future graph-enhanced methods.
- Save raw outputs, audit traces, failure cases, confidence intervals, and statistical tests for paper writing.
