import { useState, useEffect, useRef } from 'react';
import {
    Layout,
    Typography,
    Input,
    Button,
    List,
    Card,
    Tag,
    Space,
    Spin,
    Progress,
    Descriptions,
    Empty,
    message,
    Modal,
} from 'antd';
import {
    SendOutlined,
    SafetyCertificateOutlined,
    WarningOutlined,
    StopOutlined,
    DeleteOutlined,
    ThunderboltOutlined,
} from '@ant-design/icons';
import { useAppStore } from '../stores/useAppStore';
import type { ChatMessage } from '../stores/useAppStore';
import { checkHealth, auditQueryStream } from '../services/api';
import type { NodeUpdateEvent, TrustScoreDetail } from '../services/api';

const { Header, Content, Sider } = Layout;
const { Title, Text, Paragraph } = Typography;

/** 节点名到中文标签 */
const NODE_LABELS: Record<string, string> = {
    router: '🧭 知识对齐 / 意图路由',
    retriever: '📚 三粒度证据检索',
    generator: '✍️ 推理生成',
    auditor: '⚖️ 审计门控',
};

export default function AuditChat() {
    const [inputText, setInputText] = useState('');
    const [backendStatus, setBackendStatus] = useState<string>('checking...');
    const [activeNode, setActiveNode] = useState<string | null>(null);
    const [viewingEvidence, setViewingEvidence] = useState<any | null>(null); // 新增：控制溯源查看器弹窗
    const chatBottomRef = useRef<HTMLDivElement>(null);

    const {
        messages,
        isLoading,
        addMessage,
        updateLastAssistant,
        setLoading,
        clearMessages,
        selectedMessageId,
        setSelectedMessageId,
    } = useAppStore();

    // 自动滚动到底部
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // 挂载时检查后端
    useEffect(() => {
        checkHealth()
            .then((res) => setBackendStatus(`已连接 (${res.data.llm_provider})`))
            .catch(() => {
                setBackendStatus('断开连接');
                message.error('无法连接到后端服务');
            });
    }, []);

    const handleSend = () => {
        if (!inputText.trim() || isLoading) return;
        const query = inputText.trim();

        // 添加用户消息
        addMessage({
            id: Date.now().toString(),
            role: 'user',
            content: query,
            timestamp: new Date(),
        });

        setInputText('');
        setLoading(true);
        setActiveNode(null);

        // 添加 AI 占位消息
        const assistantId = (Date.now() + 1).toString();
        addMessage({
            id: assistantId,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isStreaming: true,
            currentNode: 'start',
        });

        // 发起 SSE 请求
        auditQueryStream(
            query,
            // onNodeUpdate：每当 LangGraph 某个节点跑完
            (event: NodeUpdateEvent) => {
                if (event.type === 'start') {
                    updateLastAssistant({ currentNode: 'start' });
                    return;
                }
                if (event.type === 'token' && event.content) {
                    updateLastAssistant({ appendContent: event.content } as any);
                    return;
                }

                const patch: Partial<ChatMessage> = {};

                if (event.node) {
                    patch.currentNode = event.node;
                    setActiveNode(event.node);
                }
                if (event.intent) patch.intent = event.intent;
                if (event.normalized_query) patch.normalizedQuery = event.normalized_query;
                if (event.evidence_count !== undefined) patch.evidenceCount = event.evidence_count;
                if (event.evidence) patch.evidence = event.evidence;
                if (event.answer) patch.content = event.answer;
                if (event.trust_score) {
                    patch.trustScore = event.trust_score;
                    patch.trustLevel = event.trust_score.trust_level;
                }

                updateLastAssistant(patch);
            },
            // onDone
            () => {
                setLoading(false);
                setActiveNode(null);
                updateLastAssistant({ isStreaming: false });
                // 自动选中最新的 assistant 消息，展示证据面板
                setSelectedMessageId(assistantId);
            },
            // onError
            (err) => {
                setLoading(false);
                setActiveNode(null);
                updateLastAssistant({ isStreaming: false, content: `❌ 审计出错: ${err}` });
            }
        );
    };

    const renderTrustBadge = (level?: string) => {
        if (!level) return null;
        switch (level) {
            case 'TRUSTED':
                return <Tag color="success" icon={<SafetyCertificateOutlined />}>已核实</Tag>;
            case 'WARNING':
                return <Tag color="warning" icon={<WarningOutlined />}>需注意</Tag>;
            case 'REJECTED':
                return <Tag color="error" icon={<StopOutlined />}>已拦截</Tag>;
            default:
                return null;
        }
    };

    // 获取当前选中消息的审计数据（用于右侧面板）
    const selectedMsg = messages.find((m) => m.id === selectedMessageId);

    const renderTrustGauge = (ts: TrustScoreDetail) => {
        const percent = Math.round(ts.trust_score * 10);
        const color = ts.trust_level === 'TRUSTED' ? '#52c41a'
            : ts.trust_level === 'WARNING' ? '#faad14' : '#ff4d4f';
        return (
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Progress
                    type="dashboard"
                    percent={percent}
                    strokeColor={color}
                    format={() => (
                        <div>
                            <div style={{ fontSize: 24, fontWeight: 700 }}>{ts.trust_score.toFixed(2)}</div>
                            <div style={{ fontSize: 12, color: '#999' }}>/ 10.0</div>
                        </div>
                    )}
                />
                <div style={{ marginTop: 8 }}>{renderTrustBadge(ts.trust_level)}</div>
            </div>
        );
    };

    return (
        <Layout style={{ height: '100vh' }}>
            <Header
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'linear-gradient(135deg, #0a1628 0%, #162a4a 100%)',
                    padding: '0 24px',
                    borderBottom: '1px solid #1a3a5c',
                }}
            >
                <Space>
                    <SafetyCertificateOutlined style={{ color: '#4fc3f7', fontSize: 24 }} />
                    <Title level={4} style={{ color: 'white', margin: 0 }}>
                        VeriMind-MedAudit 审计工作站
                    </Title>
                </Space>
                <Space>
                    {activeNode && (
                        <Tag icon={<ThunderboltOutlined />} color="processing">
                            {NODE_LABELS[activeNode] || activeNode}
                        </Tag>
                    )}
                    <Tag color={backendStatus.includes('已连接') ? 'success' : 'error'}>
                        {backendStatus}
                    </Tag>
                    <Button
                        type="text"
                        icon={<DeleteOutlined />}
                        onClick={clearMessages}
                        style={{ color: '#aaa' }}
                        title="清空对话"
                    />
                </Space>
            </Header>

            <Layout>
                {/* 对话区 */}
                <Content style={{ padding: '16px 24px', display: 'flex', flexDirection: 'column', background: '#f5f7fa' }}>
                    <Card
                        style={{ flex: 1, overflowY: 'auto', marginBottom: 12, borderRadius: 12 }}
                        styles={{ body: { padding: '12px 20px' } }}
                    >
                        {messages.length === 0 ? (
                            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Empty description="输入用药问题开始审计" />
                            </div>
                        ) : (
                            <List
                                itemLayout="horizontal"
                                dataSource={messages}
                                renderItem={(msg) => (
                                    <List.Item
                                        style={{ borderBottom: 'none', padding: '8px 0', cursor: msg.role === 'assistant' ? 'pointer' : 'default' }}
                                        onClick={() => msg.role === 'assistant' && setSelectedMessageId(msg.id)}
                                    >
                                        <div
                                            style={{
                                                width: '100%',
                                                display: 'flex',
                                                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                                            }}
                                        >
                                            <div
                                                style={{
                                                    maxWidth: '80%',
                                                    background: msg.role === 'user'
                                                        ? 'linear-gradient(135deg, #1890ff, #096dd9)'
                                                        : msg.id === selectedMessageId ? '#e6f4ff' : '#fff',
                                                    color: msg.role === 'user' ? 'white' : '#333',
                                                    padding: '12px 16px',
                                                    borderRadius: 12,
                                                    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                                                    border: msg.id === selectedMessageId ? '2px solid #1890ff' : '1px solid #f0f0f0',
                                                }}
                                            >
                                                <Space direction="vertical" style={{ width: '100%' }}>
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                        <Text strong style={{ color: msg.role === 'user' ? 'white' : 'inherit', fontSize: 13 }}>
                                                            {msg.role === 'user' ? '👨‍⚕️ 提问者' : '🤖 审计系统'}
                                                        </Text>
                                                        {renderTrustBadge(msg.trustLevel)}
                                                    </div>
                                                    <Paragraph style={{ margin: 0, color: 'inherit', whiteSpace: 'pre-wrap', fontSize: 14 }}>
                                                        {msg.content || (msg.isStreaming ? '正在审计中...' : '')}
                                                        {msg.isStreaming && <Spin size="small" style={{ marginLeft: 8 }} />}
                                                    </Paragraph>
                                                    {msg.role === 'assistant' && msg.intent && !msg.isStreaming && (
                                                        <Space size={4}>
                                                            <Tag color="blue">{msg.intent}</Tag>
                                                            {msg.evidenceCount !== undefined && (
                                                                <Tag color="cyan">证据 {msg.evidenceCount} 条</Tag>
                                                            )}
                                                        </Space>
                                                    )}
                                                </Space>
                                            </div>
                                        </div>
                                    </List.Item>
                                )}
                            />
                        )}
                        <div ref={chatBottomRef} />
                    </Card>

                    <div style={{ display: 'flex', gap: 8 }}>
                        <Input.TextArea
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            onPressEnter={(e) => {
                                if (!e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            placeholder="输入用药审核问题... (Enter 发送, Shift+Enter 换行)"
                            autoSize={{ minRows: 2, maxRows: 4 }}
                            style={{ borderRadius: 10 }}
                        />
                        <Button
                            type="primary"
                            icon={<SendOutlined />}
                            onClick={handleSend}
                            loading={isLoading}
                            style={{ height: 'auto', borderRadius: 10, minWidth: 100 }}
                        >
                            审计
                        </Button>
                    </div>
                </Content>

                {/* 右侧：Trust-Score 与证据面板 */}
                <Sider width={380} style={{ padding: '16px', borderLeft: '1px solid #e8e8e8', background: '#fff', overflowY: 'auto' }}>
                    <Title level={5} style={{ marginBottom: 16 }}>⚖️ 审计详情面板</Title>

                    {selectedMsg?.trustScore ? (
                        <Space direction="vertical" style={{ width: '100%' }} size="middle">
                            {/* Trust-Score 仪表盘 */}
                            {renderTrustGauge(selectedMsg.trustScore)}

                            {/* 分项指标 */}
                            <Descriptions column={1} size="small" bordered>
                                <Descriptions.Item label="检索相关性 S_ret">
                                    <Progress
                                        percent={Math.round(selectedMsg.trustScore.s_ret * 10)}
                                        size="small"
                                        strokeColor="#1890ff"
                                    />
                                </Descriptions.Item>
                                <Descriptions.Item label="忠实度 S_faith">
                                    <Progress
                                        percent={Math.round(selectedMsg.trustScore.s_faith * 10)}
                                        size="small"
                                        strokeColor="#722ed1"
                                    />
                                </Descriptions.Item>
                                <Descriptions.Item label="权威度 W_authority">
                                    <Text strong>{selectedMsg.trustScore.w_authority.toFixed(2)}</Text>
                                </Descriptions.Item>
                            </Descriptions>

                            {/* 意图 & 标准化查询 */}
                            {selectedMsg.intent && (
                                <Card size="small" title="意图识别" style={{ borderRadius: 8 }}>
                                    <Tag color="geekblue">{selectedMsg.intent}</Tag>
                                    {selectedMsg.normalizedQuery && (
                                        <Paragraph type="secondary" style={{ marginTop: 8, fontSize: 13 }}>
                                            标准化查询：{selectedMsg.normalizedQuery}
                                        </Paragraph>
                                    )}
                                </Card>
                            )}

                            {/* 证据统计与片段 */}
                            {selectedMsg.evidence && selectedMsg.evidence.length > 0 ? (
                                <Card size="small" title={`📑 检索证据 (${selectedMsg.evidenceCount || selectedMsg.evidence.length})`} style={{ borderRadius: 8 }}>
                                    {selectedMsg.evidence.map((ev, idx) => (
                                        <div key={idx} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: idx < selectedMsg.evidence!.length - 1 ? '1px dashed #e8e8e8' : 'none' }}>
                                            <Space direction="vertical" size={4} style={{ width: '100%' }}>
                                                <Space wrap>
                                                    <Tag color="purple">依据 {idx + 1}</Tag>
                                                    <Text type="secondary" style={{ fontSize: 12 }}>{ev.source}</Text>
                                                    {ev.page !== undefined && ev.page !== null && <Tag>第 {ev.page} 页</Tag>}
                                                </Space>
                                                <Paragraph style={{ fontSize: 13, margin: '4px 0 0 0', color: '#555' }} ellipsis={{ rows: 3 }}>
                                                    {ev.content}
                                                </Paragraph>
                                                <Button
                                                    type="link"
                                                    size="small"
                                                    style={{ padding: 0, fontSize: 12 }}
                                                    onClick={() => setViewingEvidence(ev)}
                                                >
                                                    🔍 查看完整原文
                                                </Button>
                                            </Space>
                                        </div>
                                    ))}
                                </Card>
                            ) : selectedMsg.evidenceCount !== undefined && (
                                <Card size="small" title="📑 检索证据" style={{ borderRadius: 8 }}>
                                    <Text>共检索到 <Text strong>{selectedMsg.evidenceCount}</Text> 条相关证据片段</Text>
                                </Card>
                            )}
                        </Space>
                    ) : (
                        <div style={{ padding: '60px 0', textAlign: 'center' }}>
                            <Empty description="点击左侧对话气泡查看审计详情" />
                        </div>
                    )}
                </Sider>
            </Layout>

            {/* 独立的 PDF 溯源原文查看器 */}
            <Modal
                title={
                    <Space>
                        <SafetyCertificateOutlined style={{ color: '#52c41a' }} />
                        证据原文查验
                    </Space>
                }
                open={!!viewingEvidence}
                onCancel={() => setViewingEvidence(null)}
                footer={[
                    <Button key="close" type="primary" onClick={() => setViewingEvidence(null)}>
                        关闭
                    </Button>
                ]}
                width={700}
                centered
            >
                {viewingEvidence && (
                    <div>
                        <Card size="small" style={{ marginBottom: 16, background: '#fafafa' }}>
                            <Descriptions column={2} size="small">
                                <Descriptions.Item label="文献出处" span={2}>
                                    <Text strong>{viewingEvidence.source}</Text>
                                </Descriptions.Item>
                                <Descriptions.Item label="系统定位页码">
                                    {viewingEvidence.page !== undefined && viewingEvidence.page !== null
                                        ? `第 ${viewingEvidence.page} 页`
                                        : '未知'}
                                </Descriptions.Item>
                                <Descriptions.Item label="检索片段 ID">
                                    <Text type="secondary" style={{ fontSize: 12 }}>{viewingEvidence.chunk_id || 'N/A'}</Text>
                                </Descriptions.Item>
                            </Descriptions>
                        </Card>
                        <Title level={5}>原文节选内容：</Title>
                        <div style={{
                            background: '#f5f5f5',
                            padding: '16px',
                            borderRadius: '8px',
                            maxHeight: '400px',
                            overflowY: 'auto',
                            border: '1px solid #e8e8e8'
                        }}>
                            <Paragraph style={{ whiteSpace: 'pre-wrap', fontSize: 15, lineHeight: 1.8 }}>
                                {viewingEvidence.content}
                            </Paragraph>
                        </div>
                    </div>
                )}
            </Modal>
        </Layout>
    );
}
