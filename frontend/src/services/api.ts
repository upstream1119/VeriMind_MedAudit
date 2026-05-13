/**
 * VeriMind-MedAudit API 服务层
 * 封装与后端的所有 HTTP 通信
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 60000,
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
    alpha?: number;
    beta?: number;
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
    timestamp?: string;
}

export interface HealthResponse {
    status: string;
    app_name: string;
    version: string;
    llm_provider: string;
    models: Record<string, string>;
}

/** SSE 节点更新事件的负载结构 */
export interface NodeUpdateEvent {
    type: 'start' | 'node_update' | 'token' | 'done' | 'error';
    node?: string;
    query?: string;
    intent?: string;
    normalized_query?: string;
    evidence_count?: number;
    evidence?: any[];
    answer?: string;
    content?: string; // 用于 token 传递
    trust_score?: TrustScoreDetail;
    message?: string;
}

// ── API 调用 ──

/** 健康检查 */
export const checkHealth = () => api.get<HealthResponse>('/health');

/** 审计查询 (非流式) */
export const auditQuery = (query: string, sessionId?: string) =>
    api.post<AuditQueryResponse>('/audit/query', { query, session_id: sessionId });

/**
 * 审计查询 (SSE 流式)
 * 适配后端 LangGraph astream 产出的 node_update 事件
 */
export const auditQueryStream = (
    query: string,
    onNodeUpdate: (event: NodeUpdateEvent) => void,
    onDone: () => void,
    onError?: (error: string) => void,
) => {
    const controller = new AbortController();

    fetch(`${API_BASE}/audit/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: controller.signal,
    }).then(async (response) => {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) return;

        let isFinished = false;
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // 保留最后一个可能不完整的行
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data: NodeUpdateEvent = JSON.parse(line.slice(6));
                        if (data.type === 'done') {
                            isFinished = true;
                            onDone();
                        } else if (data.type === 'error') {
                            isFinished = true;
                            onError?.(data.message || '未知错误');
                        } else {
                            onNodeUpdate(data);
                        }
                    } catch { /* skip malformed lines */ }
                }
            }
            // 如果已经接收到结束信号，主动打断底层网络流读取
            if (isFinished) break;
        }

        // 终极兜底：如果底层 TCP 连接断开/后端崩溃，导致没有收到明确的 done，强行解除界面卡死
        if (!isFinished) {
            onError?.('网络流异常中断（可能因 API 超时或网络波动）');
        }

    }).catch((err) => {
        if (err.name !== 'AbortError') {
            onError?.(`请求异常: ${err.message}`);
        }
    });

    return controller;
};

export default api;
