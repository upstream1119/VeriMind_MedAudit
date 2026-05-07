"""
VeriMind-Med 智能体工作流 (LangGraph DAG)
=========================================
编排 Router -> Retriever -> Generator -> Auditor 的流转图
目前为单向管道 (Pipeline), 但通过使用 StateGraph, 我们为后续
添加"打回重做"的双向循环边预留了框架能力。
"""

import logging
from langgraph.graph import StateGraph, START, END
from app.agents.state import AuditState
from app.agents.nodes.router import router_node
from app.agents.nodes.retriever_node import retriever_node
from app.agents.nodes.generator import generator_node
from app.agents.nodes.auditor import auditor_node

logger = logging.getLogger(__name__)

def build_audit_graph():
    """
    构建并编译审计大模型图
    
    返回: 
        CompiledGraph: 可以直接 `.invoke({"original_query": "..."})` 调用的引擎实例
    """
    # 按照 TypedDict 定义的结构初始化图
    workflow = StateGraph(AuditState)
    
    # 注册所有的行动节点
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("auditor", auditor_node)
    
    # 编排物理连线 (有向边)
    # START -> Router 
    workflow.add_edge(START, "router")
    
    # Router -> Retriever
    workflow.add_edge("router", "retriever")
    
    # Retriever -> Generator
    # TODO: 之后这里可以加条件边: 如果 retriever 没有查到任何证据, 甚至可以直接短路跳到 END
    workflow.add_edge("retriever", "generator")
    
    # Generator -> Auditor
    workflow.add_edge("generator", "auditor")
    
    # Auditor -> END
    # TODO: 之后这里可以加条件边: 如果审计判定为 REJECTED, 且重试次数<2, 走回 generator 要求按更高安全标准重写
    workflow.add_edge("auditor", END)
    
    # 编译整个 DAG 图
    compiled_graph = workflow.compile()
    logger.info("[LangGraph] 引擎 DAG 结构编译完成")
    
    return compiled_graph

# 提供一个全局可用的编译就绪实例
audit_engine = build_audit_graph()
