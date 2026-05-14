# VeriMind-MedAudit

面向儿科用药场景的多智能体证据审计系统，结合医学指南检索、Trust-Score 门控、来源溯源和人工复核提示，辅助识别潜在用药风险。

> 本系统不替代医生诊断或临床决策，仅用于医学知识核验、用药风险提示、教学演示和竞赛原型展示。

## 1. 项目简介

儿科用药属于低容错场景。剂量、年龄禁忌、适应症、给药方式和指南来源一旦出错，普通“总是回答”的大模型交互方式就会带来明显风险。

VeriMind-MedAudit 的目标不是生成更长的医学回答，而是让系统在证据不足、来源不匹配或知识库不完整时，能够触发保守输出、拒答或人工复核。项目采用“检索-推理-审计”多智能体协同架构，对药品剂量、禁忌症、适应症和证据来源进行自动核验。

## 2. Evidence-grounded Workflow

系统核心链路如下：

1. Router 标准化医学问题并识别意图。
2. Retriever 从多粒度医学知识库召回证据片段。
3. Generator 严格基于证据片段生成候选回答。
4. Auditor 评估检索相关性、回答忠实度和来源权威度。
5. Trust-Score 将回答分流为可回答、需人工复核或证据不足。

核心信任评分：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

其中：

- `S_ret`：检索证据与问题的相关性。
- `S_faith`：回答是否忠实于证据片段。
- `W_authority`：证据来源的权威度权重。

## 3. 当前能力

- 多粒度医学知识库：128 / 512 / 1024 三层 ChromaDB 索引。
- 双轨 PDF 解析：`pymupdf4llm` 提取正文，`pdfplumber` 提取表格。
- 扫描件识别：入库前抽样检测 PDF 是否可抽文本，避免扫描件静默污染索引。
- 多 Agent 审计链：Router -> Retriever -> Generator -> Auditor。
- Trust-Score 门控：综合检索相关性、回答忠实度和来源权威度。
- 前端工作站：React + Ant Design，支持 SSE 节点流、证据面板和 Trust-Score 展示。

## 4. 当前知识库

正式知识库位于 `data/guidelines/`，当前可用来源为：

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `中国儿科超药品说明书用药专家共识.pdf` | 儿科超说明书用药依据 | 可抽文本 |
| `儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf` | MPP / 重症 MPP / 阿奇霉素治疗依据 | 可抽文本 |
| `儿童社区获得性肺炎诊疗规范（2019年版）.pdf` | CAP、支原体肺炎、抗感染治疗依据 | 可抽文本 |
| `国家基本药物目录（2018年版）.pdf` | 官方基本药物身份 / 剂型目录依据 | 可抽文本 |

旧扫描件已移入 `data/guidelines/_archive_scanned/`，不参与正式索引。

索引完整性状态写入：

```powershell
backend\data\chroma_db\index_status.json
```

当前验收状态：`ready=true`。

## 5. 技术架构

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
Trust-Score Gate
    |
    +--> Answer
    +--> Review Required
    +--> Evidence Insufficient
```

主要技术栈：

- 后端：FastAPI、LangGraph、Pydantic
- 检索：ChromaDB、多粒度向量索引
- 文档解析：PyMuPDF / `pymupdf4llm`、`pdfplumber`
- 前端：React、TypeScript、Vite、Ant Design
- 交互：SSE 节点流、证据面板、Trust-Score 展示

## 6. 运行与验证

后端测试：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe -m pytest backend/tests -q
```

重建知识库：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe backend\rebuild_index.py
```

审计索引：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe backend\audit_index.py
```

运行人工回归：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe backend\run_manual_regression.py
```

## 7. 后续评测计划

项目下一步会从可演示原型升级为更完整的 evidence-grounded workflow：

- Trap-QA 测试集：覆盖剂量、禁忌症、证据不足、指南冲突等高风险问题。
- Claim-evidence alignment：将回答中的关键 claim 对齐到具体指南片段、页码和来源。
- Structured audit logs：记录每次请求的意图、证据、回答、分数和最终决策。
- Replayable audit traces：支持回放 Router、Retriever、Generator、Auditor 的链路决策。
- Failure taxonomy：区分 retrieval miss、evidence pollution、unsupported claim、over-refusal、under-refusal、page mismatch 等失败类型。

## 8. 使用边界

- 本项目不提供临床诊断。
- 本项目不替代医生、药师或医疗机构的专业判断。
- 所有医学输出都应被视为辅助核验信息。
- 证据不足、来源不匹配或知识库不完整时，系统应优先提示人工复核。
