/**
 * VeriMind-Med 全局状态管理 (Zustand)
 */

import { create } from 'zustand';
import type { AuditQueryResponse, TrustLevel } from '../services/api';

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    trustLevel?: TrustLevel;
    auditResult?: AuditQueryResponse;
    isStreaming?: boolean;
}

interface AppState {
    // ── 对话 ──
    messages: ChatMessage[];
    isLoading: boolean;
    addMessage: (msg: ChatMessage) => void;
    updateLastMessage: (content: string) => void;
    setLoading: (loading: boolean) => void;
    clearMessages: () => void;

    // ── 证据面板 ──
    selectedAuditResult: AuditQueryResponse | null;
    setSelectedAuditResult: (result: AuditQueryResponse | null) => void;

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
    updateLastMessage: (content) =>
        set((state) => {
            const messages = [...state.messages];
            if (messages.length > 0) {
                messages[messages.length - 1] = {
                    ...messages[messages.length - 1],
                    content,
                };
            }
            return { messages };
        }),
    setLoading: (loading) => set({ isLoading: loading }),
    clearMessages: () => set({ messages: [] }),

    // ── 证据面板 ──
    selectedAuditResult: null,
    setSelectedAuditResult: (result) => set({ selectedAuditResult: result }),

    // ── 主题 ──
    isDarkMode: true,
    toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
}));
