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
    Alert,
} from 'antd';
import {
    SendOutlined,
    SafetyCertificateOutlined,
    WarningOutlined,
    StopOutlined,
    DeleteOutlined,
    ThunderboltOutlined,
    LoginOutlined,
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

const WORKFLOW_NODES = ['router', 'retriever', 'generator', 'auditor'];

export default function AuditChat() {
    const [inputText, setInputText] = useState('');
    const [backendStatus, setBackendStatus] = useState<string>('checking...');
    const [activeNode, setActiveNode] = useState<string | null>(null);
    const [hasEntered, setHasEntered] = useState(false);
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
        setSelectedMessageId(assistantId);

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

    const getDecisionCopy = (msg?: ChatMessage) => {
        if (!msg?.trustScore) {
            return {
                color: '#2f6fed',
                title: '审计进行中',
                detail: '系统正在执行意图识别、证据检索、回答生成和审计评分。',
            };
        }
        if (msg.trustScore.trust_level === 'TRUSTED') {
            return {
                color: '#237804',
                title: '结论：有证据支持',
                detail: '当前回答与检索证据一致，可作为教学或审核演示中的参考结果。',
            };
        }
        if (msg.trustScore.trust_level === 'WARNING') {
            return {
                color: '#ad6800',
                title: '结论：需注意 / 建议复核',
                detail: '当前回答存在剂量、频次、证据充分性或临床边界风险，应由医生或药师进一步复核。',
            };
        }
        return {
            color: '#a8071a',
            title: '结论：已拦截',
            detail: '证据不足、请求越过处方边界，或回答与证据不一致，系统不应强行给出治疗方案。',
        };
    };

    const renderWorkflowStatus = (msg?: ChatMessage) => {
        const currentNode = activeNode || msg?.currentNode;
        const currentIndex = WORKFLOW_NODES.indexOf(currentNode || '');
        return (
            <Card size="small" title="审计链路" style={{ borderRadius: 10 }}>
                <Space size={6} wrap>
                    {WORKFLOW_NODES.map((node, index) => {
                        const isActive = node === currentNode;
                        const isDone = currentIndex >= 0 && index < currentIndex;
                        return (
                            <Tag
                                key={node}
                                color={isActive ? 'processing' : isDone ? 'success' : 'default'}
                                style={{ marginInlineEnd: 0 }}
                            >
                                {NODE_LABELS[node]}
                            </Tag>
                        );
                    })}
                </Space>
                <Paragraph type="secondary" style={{ fontSize: 12, margin: '8px 0 0' }}>
                    流程说明：先检索证据，再生成谨慎回答，最后由 TrustScore Gate 判断通过、提醒或拦截。
                </Paragraph>
            </Card>
        );
    };

    if (!hasEntered) {
        return (
            <Layout
                style={{
                    minHeight: '100vh',
                    background:
                        'radial-gradient(circle at 18% 18%, rgba(79,195,247,0.20), transparent 28%), linear-gradient(135deg, #071426 0%, #10294a 48%, #f5f7fa 48%, #eef4fb 100%)',
                    padding: 32,
                }}
            >
                <div
                    style={{
                        minHeight: 'calc(100vh - 64px)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                    }}
                >
                    <Card
                        style={{
                            width: 720,
                            borderRadius: 24,
                            boxShadow: '0 24px 80px rgba(7, 20, 38, 0.18)',
                            overflow: 'hidden',
                        }}
                        styles={{ body: { padding: 0 } }}
                    >
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
                            <div
                                style={{
                                    padding: 32,
                                    background: 'linear-gradient(160deg, #0a1628, #173a63)',
                                    color: 'white',
                                    minHeight: 360,
                                }}
                            >
                                <SafetyCertificateOutlined style={{ color: '#7dd3fc', fontSize: 34 }} />
                                <Title level={2} style={{ color: 'white', margin: '18px 0 8px' }}>
                                    VeriMind-MedAudit
                                </Title>
                                <Paragraph style={{ color: 'rgba(255,255,255,0.78)', fontSize: 15 }}>
                                    面向儿科用药场景的证据链审计与风险提醒系统
                                </Paragraph>
                                <Space direction="vertical" size={10} style={{ marginTop: 28 }}>
                                    <Tag color="blue">检索 - 推理 - 审计</Tag>
                                    <Tag color="cyan">证据来源与页码溯源</Tag>
                                    <Tag color="gold">证据不足优先人工复核</Tag>
                                </Space>
                            </div>
                            <div style={{ padding: 32, background: '#fff' }}>
                                <Title level={4} style={{ marginBottom: 8 }}>
                                    审计工作站登录
                                </Title>
                                <Paragraph type="secondary" style={{ marginBottom: 24 }}>
                                    比赛演示模式，无需真实账号。进入后可进行儿科用药问题审计。
                                </Paragraph>
                                <Alert
                                    type="warning"
                                    showIcon
                                    message="医疗安全声明"
                                    description="本系统仅用于科研、教学和比赛展示，不提供真实诊断或处方建议。"
                                    style={{ borderRadius: 12, marginBottom: 24 }}
                                />
                                <Button
                                    type="primary"
                                    size="large"
                                    icon={<LoginOutlined />}
                                    block
                                    onClick={() => setHasEntered(true)}
                                    style={{ height: 48, borderRadius: 12 }}
                                >
                                    进入审计工作站
                                </Button>
                                <Paragraph type="secondary" style={{ fontSize: 12, margin: '16px 0 0' }}>
                                    默认角色：比赛评审 / 教学演示用户
                                </Paragraph>
                            </div>
                        </div>
                    </Card>
                </div>
            </Layout>
        );
    }

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
                    height: 72,
                    lineHeight: 'normal',
                }}
            >
                <Space align="center">
                    <SafetyCertificateOutlined style={{ color: '#4fc3f7', fontSize: 24 }} />
                    <div>
                        <Title level={4} style={{ color: 'white', margin: 0 }}>
                            VeriMind-MedAudit 审计工作站
                        </Title>
                        <Text style={{ color: 'rgba(255,255,255,0.72)', fontSize: 12 }}>
                            面向儿科用药场景的证据链审计与风险提醒系统
                        </Text>
                    </div>
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
                    <Alert
                        showIcon
                        type="info"
                        message="医疗安全边界"
                        description="本系统仅用于科研、教学和比赛展示；不提供真实诊断或处方建议。所有医学结论必须追溯到证据，证据不足时应拒答或提示人工复核。"
                        style={{ marginBottom: 12, borderRadius: 12 }}
                    />
                    <Card
                        style={{ flex: 1, overflowY: 'auto', marginBottom: 12, borderRadius: 12 }}
                        styles={{ body: { padding: '12px 20px' } }}
                    >
                        {messages.length === 0 ? (
                            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Empty
                                    description={
                                        <Space direction="vertical" size={4}>
                                            <Text strong>选择一个演示题，或输入儿科用药审核问题</Text>
                                            <Text type="secondary">系统会展示证据来源、页码、TrustScore 和安全边界判断。</Text>
                                        </Space>
                                    }
                                />
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
                                                        {msg.content || (msg.isStreaming ? `正在审计中：${NODE_LABELS[msg.currentNode || ''] || '启动审计链路'}...` : '')}
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
                    <Title level={5} style={{ marginBottom: 4 }}>⚖️ 审计详情面板</Title>
                    <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 16 }}>
                        展示模型回答是否有证据、是否忠实于证据，以及是否触发医疗安全边界。
                    </Paragraph>

                    {selectedMsg?.role === 'assistant' ? (
                        <Space direction="vertical" style={{ width: '100%' }} size="middle">
                            {renderWorkflowStatus(selectedMsg)}

                            <Card
                                size="small"
                                title="审计结论"
                                style={{
                                    borderRadius: 10,
                                    borderLeft: `4px solid ${getDecisionCopy(selectedMsg).color}`,
                                }}
                            >
                                <Text strong style={{ color: getDecisionCopy(selectedMsg).color }}>
                                    {getDecisionCopy(selectedMsg).title}
                                </Text>
                                <Paragraph style={{ margin: '8px 0 0', fontSize: 13 }}>
                                    {getDecisionCopy(selectedMsg).detail}
                                </Paragraph>
                            </Card>

                            {/* Trust-Score 仪表盘 */}
                            {selectedMsg.trustScore ? (
                                renderTrustGauge(selectedMsg.trustScore)
                            ) : (
                                <Card size="small" style={{ borderRadius: 8 }}>
                                    <Space>
                                        <Spin size="small" />
                                        <Text>审计进行中，正在更新证据与评分...</Text>
                                    </Space>
                                    {selectedMsg.currentNode && (
                                        <div style={{ marginTop: 8 }}>
                                            <Tag color="processing">
                                                {NODE_LABELS[selectedMsg.currentNode] || selectedMsg.currentNode}
                                            </Tag>
                                        </div>
                                    )}
                                </Card>
                            )}

                            {/* 分项指标 */}
                            {selectedMsg.trustScore && (
                                <Descriptions column={1} size="small" bordered>
                                <Descriptions.Item label="检索相关性 S_ret">
                                    <Progress
                                        percent={Math.round(selectedMsg.trustScore.s_ret * 10)}
                                        size="small"
                                        strokeColor="#1890ff"
                                    />
                                    <Text type="secondary" style={{ fontSize: 12 }}>证据是否真正命中问题</Text>
                                </Descriptions.Item>
                                <Descriptions.Item label="证据忠实度 S_faith">
                                    <Progress
                                        percent={Math.round(selectedMsg.trustScore.s_faith * 10)}
                                        size="small"
                                        strokeColor="#722ed1"
                                    />
                                    <Text type="secondary" style={{ fontSize: 12 }}>回答是否被证据支持</Text>
                                </Descriptions.Item>
                                <Descriptions.Item label="资料权威度 W_authority">
                                    <Text strong>{selectedMsg.trustScore.w_authority.toFixed(2)}</Text>
                                    <br />
                                    <Text type="secondary" style={{ fontSize: 12 }}>指南 / 共识 / 目录等来源权重</Text>
                                </Descriptions.Item>
                                </Descriptions>
                            )}

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
                                <Card size="small" title={`📑 可追溯证据链 (${selectedMsg.evidenceCount || selectedMsg.evidence.length})`} style={{ borderRadius: 8 }}>
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
