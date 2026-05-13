/**
 * VeriMind-MedAudit 全局状态管理 (Zustand)
 */

import { create } from 'zustand';
import type { TrustLevel, TrustScoreDetail, RetrievedChunk } from '../services/api';

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    trustLevel?: TrustLevel;
    trustScore?: TrustScoreDetail;
    evidence?: RetrievedChunk[];
    intent?: string;
    normalizedQuery?: string;
    evidenceCount?: number;
    currentNode?: string;
    isStreaming?: boolean;
}

interface AppState {
    // ── 对话 ──
    messages: ChatMessage[];
    isLoading: boolean;
    addMessage: (msg: ChatMessage) => void;
    updateLastAssistant: (patch: Partial<ChatMessage>) => void;
    setLoading: (loading: boolean) => void;
    clearMessages: () => void;

    // ── 证据面板：当前选中的消息 ──
    selectedMessageId: string | null;
    setSelectedMessageId: (id: string | null) => void;

    // ── 主题 ──
    isDarkMode: boolean;
    toggleDarkMode: () => void;
}

export const useAppStore = create<AppState>((set) => ({
    // ── 对话 ──
    messages: [],
    isLoading: false,
    addMessage: (msg) =>
        set((state) => ({ messages: [...state.messages, msg] })),
    updateLastAssistant: (patch: Partial<ChatMessage> & { appendContent?: string }) =>
        set((state) => {
            const messages = [...state.messages];
            // 从尾部找到最后一条 assistant 消息
            for (let i = messages.length - 1; i >= 0; i--) {
                if (messages[i].role === 'assistant') {
                    const currentMsg = messages[i];
                    messages[i] = { ...currentMsg, ...patch };
                    // 处理流式追加
                    if (patch.appendContent) {
                        messages[i].content = (currentMsg.content || '') + patch.appendContent;
                        // 清除原补丁中的 appendContent 防止泄漏到 state
                        delete (messages[i] as any).appendContent;
                    }
                    break;
                }
            }
            return { messages };
        }),
    setLoading: (loading) => set({ isLoading: loading }),
    clearMessages: () => set({ messages: [] }),

    // ── 证据面板 ──
    selectedMessageId: null,
    setSelectedMessageId: (id) => set({ selectedMessageId: id }),

    // ── 主题 ──
    isDarkMode: true,
    toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
}));
