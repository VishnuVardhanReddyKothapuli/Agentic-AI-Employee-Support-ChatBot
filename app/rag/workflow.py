import logging
from pathlib import Path
from typing import Literal
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings 
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None 


def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state: AgentState, message: str) -> list[str]:
    return [*state.get("trace", []), message]


def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision)
    decision = router.invoke(
        "You route messages for an enterprise HR policy and employee support assistant.\n"
        "Classify the question into one of three routes:\n"
        "- 'kb': Questions about company HR policies, leave, holidays, benefits, payroll, remote work, attendance, onboarding, conduct, or workplace guidance.\n"
        "- 'direct': Greetings, thanks, polite pleasantries, or questions asking about your identity and role as the HR assistant.\n"
        "- 'off_topic': Questions completely unrelated to HR or workplace topics (e.g., questions about external AI models like ChatGPT/Claude, coding, cooking, math, entertainment, or general non-workplace trivia).\n\n"
        f"Question: {state['question']}\n"
    )
    return {"source_used": decision.route, "trace": add_trace(state, f"Router -> {decision.route.upper()}")}


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer", "off_topic"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    elif state["source_used"] == "off_topic":
        return "off_topic"
    return "direct_answer"


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval -> {len(docs)} chunks")}


def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade)
    context = "\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade = grader.invoke(
        "You grade evidence for an enterprise HR policy and employee support assistant.\n"
        f"Question: {state['question']}\n"
        f"Private company HR KB evidence:\n{context}\n\n"
        "Return grade 'good' if the evidence is sufficient to answer confidently and specifically.\n"
        "Otherwise return grade 'weak'."
    )
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade -> {grade.grade.upper()}")}


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback -> Tavily search"),
    }


def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade)
    grade = grader.invoke(
        f"Question: {state['question']}\n"
        f"Web evidence:\n{state['web_results']}\n\n"
        "Return grade 'good' if the evidence is sufficient and directly relevant; otherwise return 'weak'."
    )
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade -> {grade.grade.upper()}")}


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    rewritten = llm().invoke(
        "Rewrite this HR/employee-support question for better private knowledge retrieval and public web search.\n"
        "Preserve intent, add useful HR/policy keywords, do not answer, return only the query.\n"
        f"Question: {state['question']}\n"
    ).content
    rewritten_text = str(rewritten).strip()
    return {
        "current_query": rewritten_text,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite -> {rewritten_text}"),
    }


def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(
        "You are an enterprise HR policy and employee support copilot. Answer ONLY from the private company HR KB below.\n"
        "Be concise, direct, respectful, and policy-grounded. Present key facts in 2-4 bullet points without unnecessary filler.\n"
        "Do not invent policy details. Mention that the answer is based on the company's private knowledge base.\n\n"
        f"Question: {state['question']}\n\nPrivate KB:\n{context}\n"
    ).content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            title = Path(src).name if src else "Private KB"
            citations.append({"title": title, "url": "", "type": "private_kb"})
    return {"answer": str(answer), "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation -> PRIVATE KB")}


def generate_from_web(state: AgentState):
    answer = llm().invoke(
        "You are an enterprise HR policy and employee support copilot. The private company HR KB was insufficient.\n"
        "Answer ONLY from the web evidence below. Clearly say this is external public information and may require HR validation before being treated as company policy or employment guidance.\n\n"
        f"Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}\n"
    ).content
    return {"answer": str(answer), "source_used": "web_search", "trace": add_trace(state, "Answer generation -> WEB SEARCH")}


def direct_answer(state: AgentState):
    prompt = (
        f"You are the {settings.app_name}, a dedicated enterprise HR policy and employee support assistant.\n"
        "Guardrails:\n"
        "- Never identify yourself as ChatGPT, OpenAI, GPT-4, Claude, or any external third-party model.\n"
        "- If asked about your identity, name, or model, state clearly that you are the organization's HR & Employee Support Assistant, built to help employees with workplace policies, leave, and benefits.\n"
        "- Respond briefly, warmly, and politely to greetings, pleasantries, or thank-you messages, and offer to help with HR matters.\n"
        "- Keep responses professional, grounded, and concise.\n\n"
        f"User message: {state['question']}"
    )
    answer = llm().invoke(prompt).content
    return {"answer": str(answer), "source_used": "direct", "trace": add_trace(state, "Direct response -> no retrieval")}


def off_topic(state: AgentState):
    return {
        "answer": (
            f"I am the {settings.app_name}, specialized in company HR policies, employee benefits, "
            "attendance, and workplace procedures. I cannot assist with non-workplace questions or external AI discussions. "
            "Please feel free to ask any HR or policy-related question!"
        ),
        "source_used": "guardrail_off_topic",
        "trace": add_trace(state, "Guardrail -> OFF_TOPIC rejection"),
    }


def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company HR knowledge base or external search to answer confidently. Please contact the HR team or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped -> insufficient reliable evidence"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "off_topic": off_topic,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer",
        "off_topic": "off_topic",
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("off_topic", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)


def ask_stream(question: str):
    state: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    # 1. Routing
    route_result = route_question(state)
    state["source_used"] = route_result["source_used"]
    state["trace"] = route_result["trace"]
    yield {"type": "trace", "message": state["trace"][-1]}

    # Off-topic branch
    if state["source_used"] == "off_topic":
        off_res = off_topic(state)
        state["answer"] = off_res["answer"]
        state["trace"] = off_res["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}
        yield {"type": "token", "content": state["answer"]}
        yield {
            "type": "done",
            "source_used": state["source_used"],
            "citations": [],
            "trace": state["trace"],
            "answer": state["answer"],
        }
        return

    # Direct branch
    if state["source_used"] == "direct":
        prompt = (
            f"You are the {settings.app_name}, a dedicated enterprise HR policy and employee support assistant.\n"
            "Guardrails:\n"
            "- Never identify yourself as ChatGPT, OpenAI, GPT-4, Claude, or any external third-party model.\n"
            "- If asked about your identity, name, or model, state clearly that you are the organization's HR & Employee Support Assistant, built to help employees with workplace policies, leave, and benefits.\n"
            "- Respond briefly, warmly, and politely to greetings, pleasantries, or thank-you messages, and offer to help with HR matters.\n"
            "- Keep responses professional, grounded, and concise.\n\n"
            f"User message: {state['question']}"
        )
        collected = []
        for chunk in llm().stream(prompt):
            content = str(chunk.content or "")
            if content:
                collected.append(content)
                yield {"type": "token", "content": content}
        full_answer = "".join(collected)
        state["answer"] = full_answer
        state["trace"] = add_trace(state, "Direct response -> no retrieval")
        yield {"type": "trace", "message": state["trace"][-1]}
        yield {
            "type": "done",
            "source_used": "direct",
            "citations": [],
            "trace": state["trace"],
            "answer": full_answer,
        }
        return

    # KB Branch (with retry fallback)
    while True:
        ret_result = retrieve_kb(state)
        state["kb_docs"] = ret_result["kb_docs"]
        state["trace"] = ret_result["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}

        grade_result = grade_kb(state)
        state["kb_grade"] = grade_result["kb_grade"]
        state["trace"] = grade_result["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}

        if state["kb_grade"] == "good":
            context = "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in state["kb_docs"])
            prompt = (
                "You are an enterprise HR policy and employee support copilot. Answer ONLY from the private company HR KB below.\n"
                "Be concise, direct, respectful, and policy-grounded. Present key facts in 2-4 bullet points without unnecessary filler.\n"
                "Do not invent policy details. Mention that the answer is based on the company's private knowledge base.\n\n"
                f"Question: {state['question']}\n\nPrivate KB:\n{context}\n"
            )
            collected = []
            for chunk in llm().stream(prompt):
                content = str(chunk.content or "")
                if content:
                    collected.append(content)
                    yield {"type": "token", "content": content}
            full_answer = "".join(collected)
            citations = []
            seen = set()
            for d in state["kb_docs"]:
                src = d.metadata.get("source", "Private KB")
                if src not in seen:
                    seen.add(src)
                    title = Path(src).name if src else "Private KB"
                    citations.append({"title": title, "url": "", "type": "private_kb"})
            state["answer"] = full_answer
            state["source_used"] = "private_kb"
            state["citations"] = citations
            state["trace"] = add_trace(state, "Answer generation -> PRIVATE KB")
            yield {"type": "trace", "message": state["trace"][-1]}
            yield {
                "type": "done",
                "source_used": "private_kb",
                "citations": citations,
                "trace": state["trace"],
                "answer": full_answer,
            }
            return

        web_res = search_web(state)
        state["web_results"] = web_res["web_results"]
        state["citations"] = web_res["citations"]
        state["trace"] = web_res["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}

        gw_res = grade_web(state)
        state["web_grade"] = gw_res["web_grade"]
        state["trace"] = gw_res["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}

        if state["web_grade"] == "good":
            prompt = (
                "You are an enterprise HR policy and employee support copilot. The private company HR KB was insufficient.\n"
                "Answer ONLY from the web evidence below. Clearly say this is external public information and may require HR validation before being treated as company policy or employment guidance.\n\n"
                f"Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}\n"
            )
            collected = []
            for chunk in llm().stream(prompt):
                content = str(chunk.content or "")
                if content:
                    collected.append(content)
                    yield {"type": "token", "content": content}
            full_answer = "".join(collected)
            state["answer"] = full_answer
            state["source_used"] = "web_search"
            state["trace"] = add_trace(state, "Answer generation -> WEB SEARCH")
            yield {"type": "trace", "message": state["trace"][-1]}
            yield {
                "type": "done",
                "source_used": "web_search",
                "citations": state.get("citations", []),
                "trace": state["trace"],
                "answer": full_answer,
            }
            return

        if state["retry_count"] < settings.max_retries:
            rw_res = rewrite_query(state)
            state["current_query"] = rw_res["current_query"]
            state["retry_count"] = rw_res["retry_count"]
            state["trace"] = rw_res["trace"]
            yield {"type": "trace", "message": state["trace"][-1]}
            continue

        ins_res = insufficient(state)
        state["answer"] = ins_res["answer"]
        state["source_used"] = ins_res["source_used"]
        state["trace"] = ins_res["trace"]
        yield {"type": "trace", "message": state["trace"][-1]}
        yield {"type": "token", "content": state["answer"]}
        yield {
            "type": "done",
            "source_used": state["source_used"],
            "citations": [],
            "trace": state["trace"],
            "answer": state["answer"],
        }
        return
