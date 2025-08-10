from typing import List, Dict, Any, Tuple
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from settings import settings
from tools.doc_search_tool import DocSearchTool, _search_docs
from chains.answer_chain import answer_with_citations, document_to_passage

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an onboarding assistant for OpenModelica.\n"
            "Always follow this workflow:\n"
            "1) Call the 'doc_search' tool with the user question to get passages.\n"
            "2) Read ONLY those passages. Do NOT use outside knowledge.\n"
            "3) Answer concisely. If information is missing, say so.\n"
            "Citations:\n"
            "- Each passage has:\n"
            "    page_label_start / page_label_end (human page labels from the PDF viewer)\n"
            "    pdf_page_start / pdf_page_end (physical PDF page numbers for #page= links)\n"
            "- Include both in citations, like:\n"
            "  'References: p. 23 (PDF 31), p. 24–25 (PDF 32–33)'.\n"
            "- List each citation at most once, ascending by the **human page label**.\n"
            "- If a range covers multiple pages, format as 'p. X–Y (PDF A–B)'.\n\n"
            "The 'doc_search' tool returns an array of objects with keys: "
            "page (int, 1-based), source (str, file path), content (str, text snippet).\n\n"
            "Tools available: {tool_names}\n\n"
            "To use a tool, follow this exact format:\n"
            "Thought: I need to use a tool\n"
            "Action: <tool name>\n"
            "Action Input: <tool input>\n"
            "Observation: <tool output>\n"
            "... (repeat as needed)\n"
            "Thought: I have the answer\n"
            "Final Answer: <your response with citations>\n\n"
            "{tools}\n"
            "{agent_scratchpad}\n",
        ),
        ("human", "{input}"),
    ]
)


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    prompt = AGENT_PROMPT.partial(agent_scratchpad="")
    agent = create_react_agent(llm=llm, tools=[DocSearchTool], prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=[DocSearchTool],
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )


def _extract_passages_from_steps(steps: List[Tuple[Any, Any]]) -> List[Dict]:
    """Find the latest doc_search observation and turn it into passage dicts."""
    for action, observation in reversed(steps or []):
        tool = getattr(action, "tool", None)
        if tool == "doc_search":
            if isinstance(observation, str):
                try:
                    data = json.loads(observation)
                    if isinstance(data, list) and all(
                        isinstance(x, dict) for x in data
                    ):
                        return data
                except Exception:
                    pass
            try:
                return [document_to_passage(d) for d in observation]
            except Exception:
                return []
    return []


def ask_agent(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"input": question})
    passages = _extract_passages_from_steps(result.get("intermediate_steps"))
    if not passages:
        try:
            docs = _search_docs(question)
            passages = [document_to_passage(d) for d in docs]
        except Exception:
            passages = []
    return answer_with_citations(question, passages)
