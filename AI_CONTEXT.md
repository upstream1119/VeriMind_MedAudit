# VeriMind-Med 🤖 AI 开发上下文同步文档 (AI_CONTEXT)

> **⚠️ 核心约束守则 (AI 必读)** 
> 1. 每一次接到新任务前，必须首要阅读本项目下的 `AI_CONTEXT.md` 和 `.gemini/antigravity/GEMINI.md`。
> 2. 对任何文件进行创建、重构、大量修改之后，必须同步更新本文档的“当前结构与状态”部分。
> 3. 不可仅仅依赖对话历史，依赖本文件的物理记录来追踪数据流和状态流。

---

## 一、当前项目发展阶段
- **当前所处位置**：阶段一、二、二点五 以及 **阶段三（核心 Agent 引擎）** 均已全部开发并测试完成。我们利用 LangGraph 构建了包含 Router、Retriever、Generator 和 Auditor 的完整裁判流水线，并结合 Pydantic 结构化输出（加入极致精简与严格约束补丁），在批量自动化测试（10 道 Trap-QA）中达成了 10/10 的 100% REJECTED 幻觉拦截率！
- **正在进军**：阶段四 前端 React 工作站构建与后端流式接口（SSE/FastAPI）对接。

---

## 二、当前后端物理目录结构与角色说明 (`backend/`)

### 1. 根级别基础设施
- `.env`：本地环境变量与 API Key，**禁止打印或写入公开日志**。
- `requirements.txt`：Python 依赖清单（pymupdf4llm, pdfplumber, chromadb, pytest等）。

### 2. `app/` (核心业务逻辑)
- `config.py`：全局配置中枢，包含所有超参数（三粒度 size/overlap，Trust-Score $\alpha/\beta/\tau$ 阈值，权威度权重 $W_{authority}$），使用 Pydantic Settings 自动读取 `.env`。
- `main.py`：FastAPI 入口，处理 CORS 和挂载路由。

#### `app/models/` (数据与类型契约)
- `schemas.py`：Pydantic 模型，存储所有跨模块传输结构（`RetrievedChunk`, `TrustScoreDetail`, `IntentType`, `TrustLevel` 枚举），防止 dict 到处传引发 Key 错误。

#### `app/knowledge/` (大基建：知识管道)
- `parser.py`：`DualTrackMedicalParser`。双轨解析器，轨道A负责正文 (`pymupdf4llm`)，轨道B负责精准榨取危险剂量表格 (`pdfplumber`)。
- `chunker.py`：`SemanticChunker`。动态语义切分器，按医学标点硬截断，使用双指针滑动窗口实现 128 / 512 / 1024 三粒度切分，并继承 PDF Metadata。
- `indexer.py`：`VectorIndexer`。向量入库封装，包含 `ZhipuEmbeddingFunction`，执行 ChromaDB 的批量写入，使用 `hash_g{granularity}` 组合键防止去重 Bug。
- `retriever.py`：`MultiGranularityRetriever`。多粒度向量检索器。将检索结果的 L2 距离转换为相关性 $S_{ret}$，并施加 $W_{authority}$ 权重进行综合排序。

#### `app/services/` (独立业务计算服务)
- `trust_score.py`：单纯的数学组件。实现了论文公式 $Eq.4$，完成 $(S_{ret} \times \alpha + S_{faith} \times \beta) \times W_{authority}$ 计算，强制输入钳制并在三档门控之间路由。
- `llm_client.py`：大语言模型封装，适配原生 AsyncOpenAI 与 LangChain 的 `ChatOpenAI`，包含了非常关键的带有限制机制的 `generate_structured_output`。

#### `app/agents/` (系统大脑 - LangGraph 引擎)
- `state.py`：`AuditState` 定义了在有向无环图中流转的全局内存字典。
- `graph.py`：核心编排文件，用 `StateGraph` 连接了四大节点，构筑防守闭环。
- `nodes/router.py`：知识对齐与意图分流，将白话转换为带颗粒度的意图。
- `nodes/retriever_node.py`：调用底层多粒度模块。
- `nodes/generator.py`：具备极致精简限制的医嘱生成器。
- `nodes/auditor.py`：铁面裁判 Judge，基于 0~10 的忠实度打分得出最终的 `TrustScoreDetail`。

### 3. `tests/` (质量防御阵地)
- `test_trust_score.py`：13 个 pytest 单元测试，拦截任何对门控公式的错误重构。
- `trap_qa.json`：当前 10 道书本基线版陷阱题，涵盖超说明书、体重折算、年龄禁忌。**等待临床真实案例替换扩充**。

### 4. `data/` (数据存储)
- `guidelines/`：核心 PDF 存放地（专家共识、病房内科分册、处方集）。
- `chroma_db/`：ChromaDB 持久化 SQLite，不要随手删除。

---

## 三、当前已知的技术债与待办任务

1. **大文件 Embedding 消耗控制**：140MB 的《基本药物处方集》和 44MB 的《小儿内科分册》尚未入库，需要研究按特定“儿科重症章节”截断入库的节流逻辑，防止 170 万 tokens 被一次性击穿。
2. **前后端流式流转管道搭建**：目前后端逻辑跑在终端和异步断点脚本（`run_trap_qa.py`）里。需要重写 `/api/audit/query/stream`，将 Agent 执行的过程步骤实时 yield 推送给 React 前端。
3. **专家协作回收**：等待 Trap-QA 临床医生吐槽版原型的采集（已输出 Markdown 打分板），这决定了系统在“实战场景”对评委的最终杀伤力。

---

> [Update Time: 2026-04-18]

## 四、冲刺窗口（新增计划层，保留历史不删除）

### 4.1 时间与资源约束
- **V1 内部封版截止**：2026-05-02（可运行、可演示、可复现实验）
- **校赛冲刺窗口**：2026-05-03 ~ 2026-05-14（按 5 月中旬校赛节奏预留）
- **期中考时间约束**：2026-04-22、2026-04-23 以复习为主，主开发窗口从 2026-04-24 开始
- **V1 可投入预算**：约 18~27 小时（按每天 2~3 小时估算）

### 4.2 项目优先级权重（已确认）
- **整体推进权重**：安全 / 演示 / 评测 = **50 / 30 / 20**
- **校赛冲刺权重**：稳定性与观感 / 新能力 = **65 / 35**

### 4.3 医生反馈驱动的高风险场景（新增重点）
1. **药物联用风险**（尤其与抗生素联用时）
2. **剂量风险**（剂量大小、单位换算、频次、体重折算）

---

## 五、V1 验收标准（2026-05-02）

### 5.1 演示策略
- **演示模式**：仅预置题稳定演示（不做完全开放自由提问）
- **双主演示链路**：
  1. 高危拦截演示（联用 / 剂量）
  2. 完整审计链演示（检索证据 -> 生成回答 -> Trust-Score 分解 -> 最终门控）

### 5.2 量化指标
- **预置题通过率**：`>= 90%`
- 输出结果必须可追溯：包含关键证据片段与分项评分字段（`s_ret/s_faith/w_authority/trust_level`）

---

## 六、校赛冲刺目标（2026-05-03 ~ 2026-05-14）
1. **65%（稳定性与观感）**：演示流畅度、失败兜底、现场脚本稳定、展示信息密度优化
2. **35%（新能力）**：联用风险覆盖扩展、剂量场景细化、陷阱题集增强

---

## 七、更新日志（增量）
- **2026-04-19**：修复计划层乱码；保留历史成果原文，新增赛程导向的 V1 / 冲刺计划与验收口径。
