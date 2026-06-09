# Medaudit-RAG Frontend

这是 Medaudit-RAG 的前端证据审计工作站，基于 React、TypeScript、Vite 和 Ant Design 实现。

## 功能

- 输入儿科用药问答审计问题。
- 通过 SSE 接收后端 Router、Retriever、Generator、Auditor 节点状态。
- 展示 TrustScore、检索相关性、证据忠实度和来源权威度。
- 展示可追溯证据片段、来源文件和页码。

## 运行

```powershell
npm install
npm run dev -- --host=127.0.0.1 --port=5173
```

## 构建

```powershell
npm run build
```

## 边界

前端只展示后端证据审计结果，不承担医学判断。所有医学输出均应回到后端检索证据和审计门控。
