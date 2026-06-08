# VeriMind-MedAudit

**面向儿科用药科普的可信问答与证据审计系统**  
*A pediatric-medication evidence auditing prototype for safer medical AI responses.*

> 本项目不提供临床诊断、处方建议或个体化治疗决策。它面向医学知识核验、用药风险提示、教学演示、竞赛展示和科研原型验证。

## 1. Project Positioning

普通医学问答系统关注的是：**如何回答问题**。  
VeriMind-MedAudit 更关注的是：**这个回答是否应该被相信**。

在儿科用药这种低容错场景中，剂量、年龄限制、禁忌症、适应症、给药方式和指南来源一旦出错，风险会被大模型“自信回答”的表达方式放大。因此，本项目不把目标定义为“让 AI 替代医生回答用药问题”，而是把目标定义为：

- 判断回答是否有明确医学证据支撑；
- 判断回答是否忠实于检索到的指南、共识或目录来源；
- 判断回答是否越过医疗安全边界；
- 在证据不足、来源不匹配或知识库异常时，触发拒答、风险提示或人工复核。

简单说，MedAudit 的核心不是 **medical QA generation**，而是 **medical answer auditing**。

## 2. Core Safety Principle: Fail-closed

医疗场景里，“检索不到但继续编一个答案”是不可接受的。

VeriMind-MedAudit 采用 **fail-closed** 策略：

- 知识库未就绪时，不允许伪装成有依据；
- 检索不到可靠证据时，不输出确定性医学结论；
- 来源权威性不足、页码错配或证据污染时，触发风险提示；
- 回答与证据不一致时，进入人工复核或拒答路径。

> 宁可拒答，也不输出弱证据下的确定性结论。

## 3. Evidence-grounded Workflow

系统采用“检索 → 生成 → 审计 → 门控”的闭环。Router、Retriever、Generator、Auditor 不是为了强调概念包装，而是为了把医学问答中的关键责任拆开，让每一步都可以被检查和回放。

```text
User Query
    |
    v
Router
- normalize query
- identify medication / disease / dosage / safety intent
    |
    v
Retriever
- retrieve guideline / consensus / catalog evidence
- return source, page and chunk metadata
    |
    v
Generator
- draft an answer only from retrieved evidence
    |
    v
Auditor
- check retrieval relevance
- check answer faithfulness
- check source authority
- check medical safety boundary
    |
    v
Trust Gate
    +--> trusted answer with evidence
    +--> risk warning / human review
    +--> evidence-insufficient refusal
```

## 4. What the Auditor Checks

| Audit dimension | Main question | Possible decision |
| --- | --- | --- |
| Evidence support | Is there reliable evidence for this answer? | answer / review / refuse |
| Faithfulness | Does the answer stay within the evidence? | pass / revise / reject |
| Source authority | Is the source a guideline, expert consensus, or official catalog? | weighted trust score |
| Medical safety boundary | Does the answer imply diagnosis, prescription, or personalized treatment? | warning / refusal |
| Knowledge integrity | Is the index ready and free from obvious parsing or scanning failure? | allow retrieval / fail-closed |

Core trust scoring:

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

Where:

- `S_ret`: relevance between the user query and retrieved medical evidence;
- `S_faith`: faithfulness between the generated answer and retrieved evidence;
- `W_authority`: authority weight of the evidence source.

## 5. Current Capabilities

- **Multi-granularity medical indexing**: 128 / 512 / 1024 chunk ChromaDB indexes.
- **PDF governance before indexing**: checks whether documents are machine-readable before they enter the formal index.
- **Dual-track PDF parsing**: `pymupdf4llm` for text extraction and `pdfplumber` for tables.
- **Evidence metadata tracing**: source file, page, chunk and retrieval score are preserved for answer review.
- **Trust-Score gating**: combines retrieval relevance, answer faithfulness and source authority.
- **SSE workflow visualization**: frontend displays node-level streaming, evidence panels and audit results.
- **Regression scripts**: manual regression and index audit scripts are available for repeatable checks.

## 6. Current Knowledge Base

Formal evidence sources are stored in `data/guidelines/`.

| File | Purpose | Status |
| --- | --- | --- |
| `中国儿科超药品说明书用药专家共识.pdf` | Pediatric off-label medication evidence | text-readable |
| `儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf` | MPP / severe MPP / azithromycin-related evidence | text-readable |
| `儿童社区获得性肺炎诊疗规范（2019年版）.pdf` | CAP, mycoplasma pneumonia and anti-infective treatment evidence | text-readable |
| `国家基本药物目录（2018年版）.pdf` | Official essential medicine identity and dosage-form catalog | text-readable |

Scanned or unreliable files are moved to:

```text
data/guidelines/_archive_scanned/
```

Index integrity is recorded in:

```text
backend/data/chroma_db/index_status.json
```

The retriever should only operate normally when the index status is ready. Otherwise, downstream modules should receive empty or insufficient evidence and trigger the fail-closed path.

## 7. Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | FastAPI, LangGraph, Pydantic |
| Retrieval | ChromaDB, multi-granularity vector indexes |
| Document processing | PyMuPDF, `pymupdf4llm`, `pdfplumber` |
| Frontend | React, TypeScript, Vite, Ant Design |
| Interaction | SSE node streaming, evidence panel, Trust-Score display |
| Testing / audit | pytest, manual regression, index audit scripts |

## 8. Running and Validation

Backend tests:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

Rebuild knowledge index:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

Audit index status:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/audit_index.py
```

Run manual regression:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/run_manual_regression.py
```

## 9. Evaluation Roadmap

The next stage is to move from a working prototype to a more complete evidence-auditing research workflow:

- **Trap-QA test set**: pediatric medication questions covering dosage, contraindication, evidence insufficiency, guideline conflicts and unsafe prompts.
- **Claim-evidence alignment**: decompose generated answers into claims and align each claim to guideline snippets, page numbers and sources.
- **Audit label taxonomy**: classify generated claims into supported, unsupported, contradicted and insufficient-evidence cases.
- **Structured audit logs**: record query intent, retrieved evidence, generated answer, audit score and final gate decision.
- **Replayable audit traces**: replay Router, Retriever, Generator, Auditor and Trust Gate decisions.
- **Failure taxonomy**: distinguish retrieval miss, evidence pollution, unsupported claim, over-refusal, under-refusal and page mismatch.

## 10. Project Boundary

- This project is not a clinical decision system.
- It does not replace doctors, pharmacists or medical institutions.
- It should not be used to make real treatment or prescription decisions.
- All medical outputs should be treated as auxiliary evidence-checking information.
- When evidence is insufficient, mismatched or unavailable, the system should prefer refusal or human review over confident generation.
