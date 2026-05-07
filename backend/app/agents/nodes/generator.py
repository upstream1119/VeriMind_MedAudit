"""
VeriMind-Med 智能体节点: Generator (严谨医学解答生成)
======================================================
职责:
  1. 读取 State 中的 `normalized_query` 和 `evidence`
  2. 严格基于证据（不脱离上下文瞎编）起草正式解答
  3. 将解答存在 `draft_answer`
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import AuditState
from app.services.llm_client import get_langchain_client

logger = logging.getLogger(__name__)

GENERATOR_PROMPT = """你是一个严谨的儿科临床药学辅助大模型。
请你**严格只基于**下方[系统提供的专业文献片段]，来回答用户的医学查询。

规则要求：
1. **不准产生幻觉**：如果文献中没提到、无法支撑用户的疑问，必须明确回答“现有文献依据不足”。
2. **注明引用**：在你的回答中，必须引述你的结论来源于哪个文件指南。
3. **安全底线**：涉及儿科明确禁忌的，必须在开头用极具警示性的语言指出。
4. **极致精简（加速推理）**：请用最简练直接的语言回答核心问题，严禁无意义的重复和罗列，将总字数严格控制在 150 字以内。

[专业文献片段]:
{context}

[用户查询]:
{query}

请产出你的严谨解答：
"""

def generator_node(state: AuditState) -> AuditState:
    """
    LangGraph Node: 医学推理生成
    """
    query = state.get("normalized_query") or state["original_query"]
    evidence_chunks = state.get("evidence", [])
    
    logger.info(f"[Agent::Generator] 准备生成回答, 拥有证据数量: {len(evidence_chunks)}")
    
    # 拼装上下文
    if not evidence_chunks:
        context_str = "本次检索没有找到任何相关的专业医学依据。"
    else:
        context_parts = []
        for i, chunk in enumerate(evidence_chunks):
            # 将块的来源、内容、权威度一起交由 LLM 参考
            part = f"--- 证据 {i+1} ---\n来源: {chunk.source_file} (页码 {chunk.page_number})\n权威权重: {chunk.authority_weight}\n内容: {chunk.content}"
            context_parts.append(part)
        context_str = "\n\n".join(context_parts)
    
    # 构建 Prompt 并调用推理大模型
    prompt_template = ChatPromptTemplate.from_template(GENERATOR_PROMPT)
    llm = get_langchain_client(role="generator")
    
    chain = prompt_template | llm
    
    try:
        response_msg = chain.invoke({"context": context_str, "query": query})
        draft_answer = response_msg.content
        
        state["draft_answer"] = draft_answer
        state["current_node"] = "generator"
        
        logger.info(f"[Agent::Generator] 生成完毕, 字数: {len(draft_answer)}")
    except Exception as e:
        logger.error(f"[Agent::Generator] 推理生成崩溃: {e}")
        state["error_message"] = f"Generator 生成失败: {str(e)}"
        
    return state
