# Medaudit-RAG

Medaudit-RAG 是一个面向儿科用药问答的指南约束证据审计研究原型。项目关注的不是让大模型给出更像医生的答案，而是评估医学回答是否能够追溯到公开权威资料、是否忠实于证据，以及在证据不足或触及处方边界时是否应拒答或提示人工复核。

> 本项目仅用于科研实验、方法验证和医学证据审计研究，不构成真实临床诊断、处方或治疗建议。任何医学结论都必须回到已读取的指南、共识、目录或说明书证据；证据不足时应保守处理。

## 1. 研究动机

儿科用药问答属于高风险、低容错场景。剂量、频次、给药途径、年龄/体重边界、超说明书用药和药物联用等信息一旦被模型编造或错误外推，就可能形成 unsafe suggestion。普通 RAG 能缓解幻觉，但仍可能出现检索片段不相关、证据重复、回答与证据不一致、证据不足却强行回答等问题。

Medaudit-RAG 的研究目标是构建一条 evidence-grounded auditing workflow，将医学问答拆解为“问题路由 -> 证据检索 -> 受约束生成 -> 证据审计 -> TrustScore 门控”，并为后续 Graph-Enhanced Evidence Auditing 论文实验提供可复现的代码基线。

## 2. 当前系统能力

当前仓库实现的是多智能体证据审计原型，主要能力包括：

- Router：标准化儿科用药问题，识别 `DETAIL`、`CONCEPT`、`CONTEXT` 三类意图。
- Retriever：基于 128 / 512 / 1024 三粒度向量索引召回证据片段。
- Generator：要求候选回答严格基于检索证据，不允许脱离证据生成确定性医学结论。
- Auditor：计算检索相关性、证据忠实度和来源权威度，输出 TrustScore 与门控结论。
- Frontend：展示问答结果、审计状态、TrustScore 分解、证据来源与页码。
- Source admission：通过 `source_manifest.json` 和准入脚本记录资料来源、状态和可解析性，避免未审核资料污染正式知识库。

当前实现仍是 vector RAG + TrustScore baseline，尚未声称完成 GraphRAG 或专家验证。Graph-enhanced evidence auditing 是后续研究方向。

## 3. 知识库与资料准入

正式资料位于 `data/guidelines/`，PDF 原文不进入 Git 版本控制。当前已使用或计划准入的资料必须经过以下检查：

1. 来源是否为公开权威资料。
2. PDF 是否可抽取文本，是否存在扫描件/OCR 高风险。
3. 是否记录 `title`、`source_type`、`year`、`publisher`、`url`、`status`、`notes` 等元信息。
4. 是否适合进入正式索引，或仅作为 staging/候选资料。

正式索引目录为：

```text
backend/data/chroma_db/
```

索引完整性状态写入：

```text
backend/data/chroma_db/index_status.json
```

资料准入清单：

```text
data/guidelines/source_manifest.json
```

## 4. 系统架构

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
Generator
    |
    v
Auditor
    |
    v
TrustScore Gate
    |
    +--> answer_supported
    +--> review_required
    +--> insufficient_evidence
    +--> boundary_refusal
```

TrustScore 的基本形式为：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

其中：

- `S_ret`：检索证据与问题的相关性。
- `S_faith`：回答是否忠实于证据片段。
- `W_authority`：资料来源权威度权重。

## 5. 研究评测方向

后续论文实验将围绕 guideline-grounded pediatric medication safety QA benchmark 展开。该 benchmark 不宣称是真实临床数据集，而是基于公开指南、共识、说明书和目录构建的证据约束评测集。

每道样本应至少包含：

- `sample_id`
- `question`
- `risk_labels`
- `gold_source`
- `gold_page`
- `gold_span`
- `expected_decision`
- `allowed_answer_scope`
- `forbidden_claims`

核心指标包括：

- hallucination rate
- unsupported claim rate
- unsafe suggestion rate
- refusal correctness
- claim-evidence alignment precision / recall
- evidence-source mismatch rate

论文中若声称显著降低上述错误率，必须提供统计检验、置信区间和可复现的原始结果记录。

## 6. 运行方式

后端测试：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

重建知识库：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

审计索引：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/audit_index.py
```

前端构建：

```powershell
Set-Location frontend
npm run build
```

一键启动脚本：

```powershell
.\start-dev.ps1
```

如果本机 Python 环境不是默认环境，可先设置：

```powershell
$env:MEDAUDIT_PYTHON='你的 python.exe 路径'
.\start-dev.ps1
```

## 7. 医学安全边界

- 本项目不提供临床诊断。
- 本项目不生成个体化处方。
- 本项目不替代医生、药师或医疗机构的专业判断。
- 所有医学输出都只能作为证据审计实验结果。
- 当证据不足、来源不匹配、知识库不完整或用户请求越过处方边界时，系统应拒答或提示人工复核。

## 8. 当前开发重点

1. 扩展公开权威儿科用药资料库，并通过 manifest 记录资料准入状态。
2. 构建 guideline-grounded benchmark，保证每道题都有 gold evidence 和 expected decision。
3. 建立 Vanilla LLM、Naive RAG、Multi-granularity RAG、TrustScore Gate 和未来 Graph-enhanced method 的对比实验。
4. 保存原始实验结果、统计检验、置信区间和失败案例，为论文写作提供可审计证据。
