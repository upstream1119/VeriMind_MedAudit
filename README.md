# MedAudit-RAG: Rule-Aware Evidence Auditing for Low-Hallucination Pediatric Medication QA

**基于规则感知证据审计的儿科用药低幻觉问答系统**

> 本项目不提供临床诊断、处方建议或个体化治疗决策。项目面向医学知识核验、用药风险提示、教学演示、竞赛展示和科研原型验证。

## 1. 项目定位

普通医学问答系统关注的是：**如何回答问题**。  
MedAudit-RAG 更关注的是：**这个回答是否应该被相信**。

在儿科用药这种低容错场景中，剂量、年龄限制、禁忌症、适应症、给药方式和指南来源一旦出错，风险会被大模型“自信回答”的表达方式放大。因此，本项目不把目标定义为“让 AI 替代医生回答用药问题”，而是把目标定义为：

- 判断回答是否有明确医学证据支撑；
- 判断回答是否忠实于检索到的指南、共识或目录来源；
- 判断回答是否越过医疗安全边界；
- 在证据不足、来源不匹配或知识库异常时，触发拒答、风险提示或人工复核。

简单说，MedAudit 的核心不是 **医学问答生成**，而是 **医学回答审计**。

## 2. 核心安全策略：Fail-closed

医疗场景里，“检索不到但继续编一个答案”是不可接受的。

MedAudit-RAG 采用 **fail-closed** 策略：

- 知识库未就绪时，不允许伪装成有依据；
- 检索不到可靠证据时，不输出确定性医学结论；
- 来源权威性不足、页码错配或证据污染时，触发风险提示；
- 回答与证据不一致时，进入人工复核或拒答路径。

> 宁可拒答，也不输出弱证据下的确定性结论。

## 3. 证据驱动工作流

系统采用“检索 → 生成 → 审计 → 门控”的闭环。Router、Retriever、Generator、Auditor 不是为了强调概念包装，而是为了把医学问答中的关键责任拆开，让每一步都可以被检查和回放。

```text
用户问题
    |
    v
Router 路由器
- 规范化医学问题
- 识别药品、疾病、剂量、安全边界等意图
    |
    v
Retriever 检索器
- 从指南、专家共识、官方目录中召回证据
- 返回来源、页码、片段和检索分数
    |
    v
Generator 生成器
- 仅基于检索证据生成候选回答
    |
    v
Auditor 审计器
- 检查检索相关性
- 检查回答忠实度
- 检查来源权威度
- 检查医疗安全边界
    |
    v
Trust Gate 信任门控
    +--> 带证据的可信回答
    +--> 风险提示 / 人工复核
    +--> 证据不足拒答
```

## 4. 审计器检查什么

| 审计维度 | 核心问题 | 可能决策 |
| --- | --- | --- |
| 证据支撑 | 回答是否有可靠医学证据支持？ | 回答 / 复核 / 拒答 |
| 回答忠实度 | 回答是否严格停留在证据范围内？ | 通过 / 修正 / 拒绝 |
| 来源权威度 | 来源是否为指南、专家共识或官方目录？ | 加权信任评分 |
| 医疗安全边界 | 回答是否暗示诊断、处方或个体化治疗？ | 风险提示 / 拒答 |
| 知识库完整性 | 索引是否就绪，是否存在扫描件污染或解析失败？ | 允许检索 / fail-closed |

核心信任评分：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

其中：

- `S_ret`：用户问题与检索证据之间的相关性；
- `S_faith`：生成回答对检索证据的忠实程度；
- `W_authority`：证据来源的权威度权重。

## 5. 当前能力

- **多粒度医学索引**：支持 128 / 512 / 1024 三层 ChromaDB 向量索引。
- **入库前 PDF 治理**：正式建库前检测文档是否可抽取文本，避免扫描件静默污染索引。
- **双轨 PDF 解析**：使用 `pymupdf4llm` 提取正文，使用 `pdfplumber` 提取表格。
- **证据元数据追踪**：保留来源文件、页码、文本片段和检索分数，方便后续核验。
- **Trust-Score 门控**：综合检索相关性、回答忠实度和来源权威度。
- **SSE 流式可视化**：前端展示节点流、证据面板和审计结果。
- **回归与审计脚本**：提供人工回归、索引审计和测试脚本，支持重复验证。

## 6. 当前知识库

正式证据来源存放于 `data/guidelines/`。

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `中国儿科超药品说明书用药专家共识.pdf` | 儿科超说明书用药依据 | 可抽文本 |
| `儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf` | 肺炎支原体肺炎、重症肺炎支原体肺炎、阿奇霉素相关依据 | 可抽文本 |
| `儿童社区获得性肺炎诊疗规范（2019年版）.pdf` | 儿童社区获得性肺炎、支原体肺炎、抗感染治疗依据 | 可抽文本 |
| `国家基本药物目录（2018年版）.pdf` | 国家基本药物身份、剂型目录依据 | 可抽文本 |

扫描件或不可靠文件会移入：

```text
data/guidelines/_archive_scanned/
```

索引完整性状态记录在：

```text
backend/data/chroma_db/index_status.json
```

只有当索引状态为就绪时，检索器才应正常工作。否则，下游模块应收到空证据或证据不足结果，并触发 fail-closed 路径。

## 7. 技术栈

| 层级 | 工具 |
| --- | --- |
| 后端 | FastAPI、LangGraph、Pydantic |
| 检索 | ChromaDB、多粒度向量索引 |
| 文档处理 | PyMuPDF、`pymupdf4llm`、`pdfplumber` |
| 前端 | React、TypeScript、Vite、Ant Design |
| 交互 | SSE 节点流、证据面板、Trust-Score 展示 |
| 测试与审计 | pytest、人工回归脚本、索引审计脚本 |

## 8. 运行与验证

后端测试：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

重建知识库索引：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

审计索引状态：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/audit_index.py
```

运行人工回归：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/run_manual_regression.py
```

## 9. 评测路线

下一阶段会从可演示原型升级为更完整的证据审计研究工作流：

- **Trap-QA 测试集**：覆盖剂量、禁忌症、证据不足、指南冲突和不安全提问等高风险问题。
- **Claim-evidence alignment**：将生成回答拆解为关键 claim，并对齐到指南片段、页码和来源。
- **审计标签体系**：将生成内容标注为 supported、unsupported、contradicted、insufficient evidence 等类别。
- **结构化审计日志**：记录问题意图、检索证据、生成回答、审计分数和最终门控决策。
- **可回放审计轨迹**：支持回放 Router、Retriever、Generator、Auditor 和 Trust Gate 的链路决策。
- **失败类型分析**：区分 retrieval miss、evidence pollution、unsupported claim、over-refusal、under-refusal、page mismatch 等失败类型。

## 10. 使用边界

- 本项目不是临床决策系统。
- 本项目不替代医生、药师或医疗机构的专业判断。
- 本项目不能用于真实治疗或处方决策。
- 所有医学输出都应被视为辅助核验信息。
- 当证据不足、来源不匹配或知识库不可用时，系统应优先拒答或提示人工复核，而不是自信生成。
