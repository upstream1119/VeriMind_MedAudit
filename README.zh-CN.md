# MedAudit-RAG

[English README](./README.md)

面向儿科用药问答的规则感知证据审计 RAG 系统。

MedAudit-RAG 是一个研究原型，用于审计儿科用药回答是否受到指南、共识、目录或说明书证据支持。它不是医疗聊天机器人，不用于替代医生、生成处方或提供临床治疗建议。系统会检索证据、生成受约束回答、审计回答忠实度，并通过 TrustScore 门控决定回答、拒答或提示人工复核。

> 当前状态：研究原型。当前实现是 vector RAG + TrustScore baseline，包含 FastAPI、LangGraph、ChromaDB 和 React 前端。Graph-enhanced evidence auditing 和专家验证是后续研究方向。

## 核心能力

- 将儿科用药问题路由为 `DETAIL`、`CONCEPT`、`CONTEXT` 三类意图。
- 从多粒度指南索引中召回证据片段。
- 要求回答只基于检索证据生成。
- 审计检索相关性、回答忠实度和来源权威度。
- 用 TrustScore 门控区分支持回答、需要复核、证据不足和边界拒答。
- 在前端展示回答状态、TrustScore 分解、引用来源、页码和证据片段。

## 为什么需要这个项目

儿科用药问答属于高风险、低容错场景。剂量、频次、给药途径、年龄或体重边界、超说明书用药和药物联用等信息，一旦被模型编造或错误外推，就可能形成 unsafe suggestion。普通 RAG 可以缓解幻觉，但仍可能检索到弱证据、重复无关片段，或在证据不足时生成看似确定的回答。

MedAudit-RAG 关注的是回答审计，而不只是回答生成。

## 系统架构

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

TrustScore 基于检索相关性、回答忠实度和来源权威度：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

## 技术栈

- 后端：FastAPI, Python
- RAG 编排：LangGraph
- 向量数据库：ChromaDB
- 前端：React, Ant Design, Vite
- 流式输出：Server-Sent Events
- 测试：pytest

## 当前仓库范围

当前仓库包含：

- 健康检查、审计问答和 SSE 流式接口
- router、retriever、generator、auditor 节点
- TrustScore 计算和来源权威度加权
- 指南资料准入脚本和 manifest 记录
- 用于展示审计状态和证据的 React 前端
- parser、source preparation 和 TrustScore 相关测试

当前 baseline 不声称已经完成 GraphRAG、临床部署或专家医学验证。

## 知识库与资料准入

指南 PDF 原文不进入 Git。资料进入正式索引前需要通过 manifest 记录来源和准入状态。

正式索引目录：

```text
backend/data/chroma_db/
```

索引状态：

```text
backend/data/chroma_db/index_status.json
```

资料准入清单：

```text
data/guidelines/source_manifest.json
```

## 快速启动

安装后端依赖：

```powershell
pip install -r backend/requirements.txt
```

运行后端测试：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

重建向量索引：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

启动后端：

```powershell
$env:PYTHONPATH='backend'
python -m uvicorn app.main:app --reload
```

启动前端：

```powershell
Set-Location frontend
npm install
npm run dev
```

## API 与输出结构

主要接口：

```text
GET  /api/health
POST /api/audit/query
POST /api/audit/query/stream
```

审计返回结构用于展示：

- 标准化问题和意图类型
- 回答文本或拒答说明
- TrustScore 与分项得分
- 检索证据片段
- 引用来源和页码
- 最终门控结论，例如支持回答、需要复核、证据不足或边界拒答

## 评测计划

后续 benchmark 方向是 guideline-grounded pediatric medication safety QA。每个样本应包含 gold evidence、expected decision、allowed answer scope 和 forbidden claims。

计划关注的指标包括：

- hallucination rate
- unsupported claim rate
- unsafe suggestion rate
- refusal correctness
- claim-evidence alignment precision and recall
- evidence-source mismatch rate

未来如果声称降低上述错误率，需要提供原始输出、审计轨迹、置信区间和统计检验。

## 医学安全边界

本项目仅用于科研实验、方法验证和医学证据审计研究。

它不提供临床诊断、个体化处方或治疗建议。所有医学输出都必须回到已检索到的指南、共识、目录或说明书证据。当证据不足、不完整、不匹配，或用户请求越过允许回答边界时，系统应拒答或提示人工复核。

## 路线图

- 扩展公开权威儿科用药资料库。
- 构建带 gold evidence 的指南约束评测集。
- 对比 vanilla LLM、naive RAG、multi-granularity RAG、TrustScore Gate 和未来 graph-enhanced 方法。
- 保存原始输出、审计轨迹、失败案例、置信区间和统计检验结果，为论文写作提供可审计证据。
