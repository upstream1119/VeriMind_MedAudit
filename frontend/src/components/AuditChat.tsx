import { useState, useEffect } from 'react';
import {
    Layout,
    Typography,
    Input,
    Button,
    List,
    Card,
    Tag,
    Space,
    Badge,
    Spin,
    message,
} from 'antd';
import {
    SendOutlined,
    SafetyCertificateOutlined,
    WarningOutlined,
    StopOutlined,
} from '@ant-design/icons';
import { useAppStore } from '../stores/useAppStore';
import { checkHealth, auditQueryStream } from '../services/api';

const { Header, Content, Sider } = Layout;
const { Title, Text, Paragraph } = Typography;

export default function AuditChat() {
    const [inputText, setInputText] = useState('');
    const [backendStatus, setBackendStatus] = useState<string>('checking...');

    const { messages, isLoading, addMessage, updateLastMessage, setLoading } =
        useAppStore();

    // 挂载时检查后端健康状态
    useEffect(() => {
        checkHealth()
            .then((res) => {
                setBackendStatus(`已连接 (${res.data.llm_provider})`);
            })
            .catch(() => {
                setBackendStatus('断开连接');
                message.error('无法连接到后端服务，请检查 FastAPI 是否启动');
            });
    }, []);

    const handleSend = () => {
        if (!inputText.trim() || isLoading) return;

        const userMsgId = Date.now().toString();
        const query = inputText.trim();

        // 1. 添加用户消息
        addMessage({
            id: userMsgId,
            role: 'user',
            content: query,
            timestamp: new Date(),
        });

        setInputText('');
        setLoading(true);

        // 2. 添加 AI 占位消息 (流式加载中)
        const assistantMsgId = (Date.now() + 1).toString();
        addMessage({
            id: assistantMsgId,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isStreaming: true,
        });

        // 3. 发起 SSE 请求
        let accumulatedContent = '';
        auditQueryStream(
            query,
            // onToken
            (token) => {
                accumulatedContent += token;
                updateLastMessage(accumulatedContent);
            },
            // onDone
            (finalAnswer) => {
                setLoading(false);
                useAppStore.setState((state) => {
                    const newMessages = [...state.messages];
                    const lastIndex = newMessages.length - 1;
                    newMessages[lastIndex] = {
                        ...newMessages[lastIndex],
                        content: finalAnswer,
                        isStreaming: false,
                        // TODO: 解析完毕后挂载真实的 trustLevel 和证据链
                        trustLevel: 'TRUSTED',
                    };
                    return { messages: newMessages };
                });
            },
            // onError
            (err) => {
                setLoading(false);
                message.error(`审计出错: ${err}`);
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

    return (
        <Layout style={{ height: '100vh' }}>
            <Header
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: '#001529',
                    padding: '0 24px',
                }}
            >
                <Space>
                    <SafetyCertificateOutlined style={{ color: '#1890ff', fontSize: 24 }} />
                    <Title level={4} style={{ color: 'white', margin: 0 }}>
                        VeriMind-Med 审计工作站
                    </Title>
                </Space>
                <Badge
                    status={backendStatus.includes('已连接') ? 'success' : 'error'}
                    text={<span style={{ color: 'white' }}>后端端点: {backendStatus}</span>}
                />
            </Header>

            <Layout>
                {/* 左侧/中间：对话区 */}
                <Content style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
                    <Card
                        style={{ flex: 1, overflowY: 'auto', marginBottom: 16 }}
                        bodyStyle={{ padding: '16px 24px' }}
                    >
                        <List
                            itemLayout="horizontal"
                            dataSource={messages}
                            renderItem={(msg) => (
                                <List.Item style={{ borderBottom: 'none', padding: '12px 0' }}>
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
                                                background: msg.role === 'user' ? '#1890ff' : '#f0f2f5',
                                                color: msg.role === 'user' ? 'white' : 'black',
                                                padding: '12px 16px',
                                                borderRadius: 8,
                                            }}
                                        >
                                            <Space direction="vertical" style={{ width: '100%' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                    <Text strong style={{ color: msg.role === 'user' ? 'white' : 'inherit' }}>
                                                        {msg.role === 'user' ? '医生' : '审计系统'}
                                                    </Text>
                                                    {renderTrustBadge(msg.trustLevel)}
                                                </div>
                                                <Paragraph style={{ margin: 0, color: 'inherit', whiteSpace: 'pre-wrap' }}>
                                                    {msg.content}
                                                    {msg.isStreaming && <Spin size="small" style={{ marginLeft: 8 }} />}
                                                </Paragraph>
                                            </Space>
                                        </div>
                                    </div>
                                </List.Item>
                            )}
                        />
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
                            placeholder="输入医疗咨询、处方审核问题... (Shift+Enter 换行)"
                            autoSize={{ minRows: 2, maxRows: 6 }}
                        />
                        <Button
                            type="primary"
                            icon={<SendOutlined />}
                            onClick={handleSend}
                            loading={isLoading}
                            style={{ height: 'auto' }}
                        >
                            发送审计
                        </Button>
                    </div>
                </Content>

                {/* 右侧：Trust-Score 与证据面板 */}
                <Sider width={400} theme="light" style={{ padding: '24px', borderLeft: '1px solid #f0f0f0' }}>
                    <Title level={5}>🔍 证据链溯源</Title>
                    <Text type="secondary">点击左侧对话查看详细判定依据</Text>
                    {/* TODO: 阶段四集成 ECharts 和 PDF Viewer */}
                    <Card style={{ marginTop: 16 }}>
                        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fafafa' }}>
                            仪表盘占位
                        </div>
                    </Card>
                </Sider>
            </Layout>
        </Layout>
    );
}
