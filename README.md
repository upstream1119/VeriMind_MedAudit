# VeriMind-MedAudit

基于多智能体协同与证据链溯源的儿科用药合规审计系统。

## 1. 项目简介

VeriMind-MedAudit 面向儿科用药这一高风险场景，构建“检索-推理-审计”多智能体协同架构，对药品剂量、禁忌症、适应症和证据来源进行自动核验，辅助发现潜在用药风险。

本系统不替代医生诊断，仅作为医学知识核验、用药风险提示和教学演示工具。

## 2. 当前能力

- 多粒度医学知识库：128 / 512 / 1024 三层 ChromaDB 索引
- 双轨 PDF 解析：`pymupdf4llm` 提取正文，`pdfplumber` 提取表格
- 扫描件识别：入库前抽样检测 PDF 是否可抽文本，避免扫描件静默污染索引
- 多 Agent 审计链：Router -> Retriever -> Generator -> Auditor
- Trust-Score 门控：综合检索相关性、回答忠实度和来源权威度
- 前端工作站：React + Ant Design，支持 SSE 节点流、证据面板和 Trust-Score 展示

## 3. 当前知识库

正式知识库位于 `data/guidelines/`，当前可用来源为：

| 文件 | 用途 | 状态 |
|---|---|---|
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

## 4. 技术架构

系统由四个 LangGraph 节点组成：

1. Router：将自然语言问题标准化为医学检索查询和意图类型
2. Retriever：执行多粒度检索，并根据来源计算 `W_authority`
3. Generator：严格基于证据片段生成简短回答
4. Auditor：评估回答忠实度并计算 Trust-Score

核心信任评分：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

当证据不足、来源不匹配或信任分不足时，系统会提示拒答或人工复核。

## 5. 运行与验证

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

前端构建：

```powershell
cd frontend
npm run build
```

## 6. 最新回归结果

最新人工回归目录：

```text
docs/test_runs/2026-05-13_13-13-54/
```

重点结果：

- 阿奇霉素静滴问题：命中《儿童肺炎支原体肺炎诊疗指南（2023年版）》第 14 页，给出重症 MPP 静点阿奇霉素相关依据，门控结果为 `WARNING`
- 氨溴索静脉给药问题：当前知识库未找到直接依据，系统输出“现有文献依据不足”，避免强行回答

## 7. 项目边界声明

本项目仅用于科研、教学、比赛展示和辅助审计演示，不构成真实临床诊断或处方建议。所有输出结果均需由专业医务人员复核。
