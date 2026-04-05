/**
 * VeriMind-Med API 服务层
 * 封装与后端的所有 HTTP 通信
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 30000,
    headers: { 'Content-Type': 'application/json' },
});

// ── 类型定义 ──

export type TrustLevel = 'TRUSTED' | 'WARNING' | 'REJECTED';
export type IntentType = 'DETAIL' | 'CONCEPT' | 'CONTEXT';

export interface TrustScoreDetail {
    s_ret: number;
    s_faith: number;
    w_authority: number;
    trust_score: number;
    trust_level: TrustLevel;
    alpha: number;
    beta: number;
}

export interface RetrievedChunk {
    content: string;
    source: string;
    page: number | null;
    chunk_id: string;
    similarity_score: number;
    evidence_level: string;
    authority_weight: number;
}

export interface AuditQueryResponse {
    query: string;
    normalized_query: string;
    intent: IntentType;
    answer: string;
    trust_score: TrustScoreDetail;
    evidence: RetrievedChunk[];
    processing_time: number;
    timestamp: string;
}

export interface HealthResponse {
    status: string;
    app_name: string;
    version: string;
    llm_provider: string;
    models: Record<string, string>;
}

// ── API 调用 ──

/** 健康检查 */
export const checkHealth = () => api.get<HealthResponse>('/health');

/** 审计查询 (非流式) */
export const auditQuery = (query: string, sessionId?: string) =>
    api.post<AuditQueryResponse>('/audit/query', { query, session_id: sessionId });

/**
 * 审计查询 (SSE 流式)
 * @param query 用户提问
 * @param onToken 每收到一个 token 的回调
 * @param onDone 流式完成回调
 * @param onError 错误回调
 */
export const auditQueryStream = (
    query: string,
    onToken: (token: string) => void,
    onDone: (answer: string) => void,
    onError?: (error: string) => void,
) => {
    const eventSource = new EventSource(
        `${API_BASE}/audit/query/stream?` + new URLSearchParams({ query }),
    );

    // SSE 需要 POST, EventSource 只支持 GET, 所以这里用 fetch 替代
    fetch(`${API_BASE}/audit/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    }).then(async (response) => {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) return;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'token') onToken(data.content);
                        else if (data.type === 'done') onDone(data.answer);
                        else if (data.type === 'error') onError?.(data.message);
                    } catch { /* skip malformed lines */ }
                }
            }
        }
    }).catch((err) => onError?.(err.message));

    eventSource.close();
};

export default api;
