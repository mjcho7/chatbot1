"""DATA 폴더의 PDF를 대상으로 답하는 Streamlit RAG 챗봇입니다."""

import os
import re
from pathlib import Path

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 이 파일을 기준으로 DATA 폴더를 찾으므로, 어느 위치에서 실행해도 동작합니다.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
RETRIEVER_K = 4
MAX_CONTEXT_PAGES = 8
MAX_DISPLAY_SOURCES = 3


def get_openai_api_key() -> str | None:
    """Cloud Secrets를 우선 사용하고, 로컬에서는 .env 파일을 사용합니다."""
    # 로컬 개발 환경에서는 기존 .env 파일의 값을 환경 변수로 불러옵니다.
    load_dotenv(PROJECT_DIR / ".env")
    try:
        # Streamlit Cloud의 Secrets 입력란에 저장한 값은 이 방식으로 읽습니다.
        cloud_api_key = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        # 로컬에 secrets.toml이 없는 경우는 정상이며 .env 값을 사용합니다.
        cloud_api_key = None

    return cloud_api_key or os.getenv("OPENAI_API_KEY")


def get_pdf_paths() -> list[Path]:
    """DATA 폴더 안의 모든 PDF 파일 경로를 이름순으로 반환합니다."""
    return sorted(DATA_DIR.glob("*.pdf"))


def get_data_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """파일이 바뀌면 Streamlit 캐시도 새로 만들기 위한 식별값입니다."""
    return tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def load_pdf_documents(paths: list[Path]) -> list[Document]:
    """PDF의 각 페이지를 LangChain Document로 변환해 파일명과 쪽수를 보존합니다."""
    documents: list[Document] = []
    for path in paths:
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_number},
                    )
                )
    return documents


@st.cache_resource(show_spinner=False)
def load_source_pages(
    data_signature: tuple[tuple[str, int, int], ...]
) -> list[Document]:
    """검색 결과 주변 페이지를 보강할 수 있도록 원문 페이지를 한 번만 읽습니다."""
    _ = data_signature
    return load_pdf_documents(get_pdf_paths())


@st.cache_resource(show_spinner=False)
def build_vector_store(
    api_key: str, data_signature: tuple[tuple[str, int, int], ...]
) -> InMemoryVectorStore:
    """문서를 청크로 나누고 OpenAI 임베딩으로 InMemoryVectorStore를 만듭니다."""
    # data_signature는 캐시 무효화에 사용합니다. 함수 안에서는 PDF 목록을 다시 읽습니다.
    _ = data_signature
    documents = load_pdf_documents(get_pdf_paths())
    if not documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 PDF 텍스트를 찾지 못했습니다.")

    # 한 청크가 지나치게 길지 않으면서 문맥이 이어지도록 일부를 겹칩니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def format_context(documents: list[Document]) -> str:
    """검색된 문서 조각을 모델에 전달할 문맥 문자열로 바꿉니다."""
    return "\n\n".join(
        f"[파일: {doc.metadata['source']} | 페이지: {doc.metadata['page']}]\n{doc.page_content}"
        for doc in documents
    )


def expand_with_neighbor_pages(
    retrieved_documents: list[Document], source_pages: list[Document]
) -> list[Document]:
    """Q&A가 다음 쪽에 이어지는 PDF 특성을 고려해 앞뒤 페이지를 함께 제공합니다."""
    page_lookup = {
        (str(document.metadata["source"]), int(document.metadata["page"])): document
        for document in source_pages
    }
    selected_keys: list[tuple[str, int]] = []

    # 우선순위가 높은 검색 결과 자체를 먼저 넣습니다.
    for document in retrieved_documents:
        key = (str(document.metadata["source"]), int(document.metadata["page"]))
        if key not in selected_keys:
            selected_keys.append(key)

    # 상위 결과의 전후 페이지에는 질문의 답변, 예외, 표가 이어질 수 있습니다.
    for document in retrieved_documents[:3]:
        source = str(document.metadata["source"])
        page = int(document.metadata["page"])
        for neighbor_page in (page - 1, page + 1):
            neighbor_key = (source, neighbor_page)
            if neighbor_key in page_lookup and neighbor_key not in selected_keys:
                selected_keys.append(neighbor_key)

    return [page_lookup[key] for key in selected_keys[:MAX_CONTEXT_PAGES] if key in page_lookup]


def format_conversation_history(messages: list[dict[str, object]], limit: int = 6) -> str:
    """최근 대화를 검색어 보정과 질문 해석에 쓸 수 있는 짧은 문자열로 만듭니다."""
    recent_messages = messages[-limit:]
    if not recent_messages:
        return "(이전 대화 없음)"

    history_lines: list[str] = []
    for message in recent_messages:
        role = "사용자" if message.get("role") == "user" else "챗봇"
        # 지나치게 긴 답변이 다음 요청의 문맥을 모두 차지하지 않도록 길이를 제한합니다.
        content = str(message.get("content", "")).strip()[:600]
        if content:
            history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines) or "(이전 대화 없음)"


def rewrite_search_query(
    question: str, conversation_history: str, api_key: str
) -> str:
    """이전 질문·답변과 후속 질문을 하나의 독립 검색 질문으로 만듭니다."""
    if conversation_history == "(이전 대화 없음)":
        return question

    # 답변을 만들지 않고 검색할 질문 한 문장만 반환하도록 별도 Runnable을 구성합니다.
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 문서 검색을 위한 질문 보정 도우미입니다.
이전 질문, 이전 답변, 현재 후속 질문을 함께 읽고 문서 검색에 쓸 독립적인 질문 한 문장으로 다시 작성하세요.
현재 질문이 금액·횟수·날짜의 정정 또는 확인이면, 이전 질문의 대상과 이전 답변의 비교 대상 수치를 모두 포함하세요.
예를 들어 '3만원 아니야?'는 무엇의 금액인지와 2만원·3만원 중 무엇을 확인할지 분명히 적어야 합니다.
이전 답변은 검색 주제를 복원하는 단서일 뿐 사실로 확정하지 마세요. 답하거나 새로운 사실을 추가하지 마세요.
검색 질문 한 문장만 한국어로 반환하세요.""",
            ),
            (
                "human",
                "이전 대화:\n{conversation_history}\n\n현재 질문: {question}",
            ),
        ]
    )
    rewrite_chain = (
        rewrite_prompt
        | ChatOpenAI(model=CHAT_MODEL, api_key=api_key, temperature=0)
        | StrOutputParser()
    )
    rewritten_query = rewrite_chain.invoke(
        {"conversation_history": conversation_history, "question": question}
    ).strip()
    return rewritten_query or question


def evidence_sentence(text: str, limit: int = 300) -> str:
    """출처 아래에 표시할 읽기 쉬운 근거 문장(원문 발췌)을 만듭니다."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+|\s*•\s*", clean_text)
        if sentence.strip()
    ]
    # 질문 제목(Q&A, 물음표)보다 실제 답변 문장을 우선합니다.
    answer_sentences = [
        sentence
        for sentence in sentences
        if "Q&A" not in sentence and not sentence.rstrip().endswith("?")
    ]
    # 금액이 있는 문장을 먼저 보여 주면 수치 확인 후속 질문의 근거가 명확합니다.
    money_excerpt = next(
        (sentence for sentence in answer_sentences if "만원" in sentence),
        None,
    )
    evidence_words = ("정액", "지급", "가능", "불가", "제외")
    excerpt = money_excerpt or next(
        (
            sentence
            for sentence in answer_sentences
            if any(word in sentence for word in evidence_words)
        ),
        answer_sentences[0] if answer_sentences else (sentences[0] if sentences else clean_text),
    )
    return excerpt[:limit] + ("…" if len(excerpt) > limit else "")


@st.cache_data(show_spinner=False)
def render_pdf_page(source: str, page_number: int, modified_time_ns: int) -> bytes:
    """원본 PDF의 한 페이지를 화면에 표시할 PNG 이미지로 변환합니다."""
    # modified_time_ns는 PDF가 갱신되었을 때 이전 이미지 캐시를 쓰지 않게 합니다.
    _ = modified_time_ns
    pdf_path = DATA_DIR / source
    pdf_document = pymupdf.open(pdf_path)
    try:
        # 확대 배율을 적용해 작은 글자도 읽기 쉽게 렌더링합니다.
        page = pdf_document.load_page(page_number - 1)
        image = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        return image.tobytes("png")
    finally:
        pdf_document.close()


@st.dialog("출처 페이지")
def show_source_page(source: str, page_number: int) -> None:
    """버튼을 눌렀을 때 원문 PDF의 해당 페이지를 팝업으로 보여 줍니다."""
    pdf_path = DATA_DIR / source
    if not pdf_path.is_file():
        st.error("출처 PDF 파일을 찾지 못했습니다.")
        return

    st.caption(f"{source} · {page_number}쪽")
    try:
        page_image = render_pdf_page(source, page_number, pdf_path.stat().st_mtime_ns)
        st.image(page_image, use_container_width=True)
    except Exception as error:
        st.error(f"출처 페이지를 열지 못했습니다: {error}")


def render_sources(sources: list[dict[str, str | int]], key_prefix: str) -> None:
    """답변에 연결된 출처와 원문 페이지 열기 버튼을 화면에 표시합니다."""
    st.markdown("#### 출처와 근거 문장")
    for index, source_info in enumerate(sources[:MAX_DISPLAY_SOURCES], start=1):
        source = str(source_info["source"])
        page = int(source_info["page"])
        st.markdown(f"**{index}. {source} (p. {page})**")
        st.caption(f"근거 문장: {source_info['evidence']}")
        # 이전 대화의 출처 버튼과 현재 답변의 버튼이 겹치지 않도록 고유 key를 사용합니다.
        button_key = f"{key_prefix}-source-page-{index}-{source}-{page}"
        if st.button("출처 페이지 열기", key=button_key):
            show_source_page(source, page)


def main() -> None:
    """Streamlit 화면을 그리고 질문-검색-답변 흐름을 실행합니다."""
    api_key = get_openai_api_key()

    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더의 문서 내용만 근거로 답변합니다.")

    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        st.error("DATA 폴더에 PDF 파일이 없습니다.")
        return
    if not api_key:
        st.error("OPENAI_API_KEY를 .env 또는 Streamlit Cloud Secrets에 설정해 주세요.")
        return

    with st.sidebar:
        st.subheader("색인 문서")
        for path in pdf_paths:
            st.write(f"- {path.name}")
        if st.button("문서 색인 다시 만들기"):
            st.cache_resource.clear()
            st.rerun()

    try:
        with st.spinner("문서를 읽고 검색 인덱스를 준비하고 있습니다..."):
            vector_store = build_vector_store(api_key, get_data_signature(pdf_paths))
    except Exception as error:
        st.error(f"문서 색인 생성에 실패했습니다: {error}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 버튼을 클릭해 화면이 다시 실행되어도, 저장해 둔 출처를 다시 표시합니다.
            if message["role"] == "assistant" and message.get("sources"):
                if message.get("search_query"):
                    with st.expander("검색에 사용한 질문"):
                        st.write(message["search_query"])
                render_sources(message["sources"], key_prefix=f"history-{message_index}")

    question = st.chat_input("문서에 관해 질문해 보세요")
    if not question:
        return

    # 현재 질문을 저장하기 전의 기록만 사용해야 같은 질문이 두 번 들어가지 않습니다.
    conversation_history = format_conversation_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 이전 질문·답변과 후속 질문을 합쳐, 벡터 검색용 독립 질문 하나를 만듭니다.
    try:
        rewritten_query = rewrite_search_query(
            question, conversation_history, api_key
        )
    except Exception as error:
        # 검색어 보정에 실패해도 원래 질문으로 검색해 챗봇 사용이 중단되지 않게 합니다.
        st.warning(f"검색어 보정에 실패해 원래 질문으로 검색합니다: {error}")
        rewritten_query = question

    # 여러 대화 문장을 그대로 넣는 대신, 보정된 독립 질문만으로 검색해 유사도 혼선을 줄입니다.
    search_query = rewritten_query

    retrieved_documents = vector_store.similarity_search(search_query, k=RETRIEVER_K)
    # 검색된 페이지뿐 아니라 이어지는 페이지도 포함해 Q&A의 답변 문장이 누락되지 않게 합니다.
    context_documents = expand_with_neighbor_pages(
        retrieved_documents, load_source_pages(get_data_signature(pdf_paths))
    )
    context = format_context(context_documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 제공된 문서만 근거로 답하는 도우미입니다.
문맥에 답이 없거나 근거가 부족하면 반드시 '제공된 문서에서 확인할 수 없습니다.'라고 답하세요.
문맥에 없는 사실을 추측하거나 일반 지식으로 보완하지 마세요.
이전 대화는 현재 질문의 지시 대상과 생략된 표현을 해석하는 데만 사용하세요.
답변의 사실 근거는 반드시 아래 문서 문맥에서만 찾고, 이전 대화를 근거로 삼지 마세요.
현재 질문이 이전 답변의 금액·횟수·날짜를 정정하거나 확인하는 내용이면, 문서 문맥의 수치와 직접 비교해 답하세요.
답변은 한국어로 간결하고 명확하게 작성하세요.""",
            ),
            (
                "human",
                "이전 대화:\n{history}\n\n문서 문맥:\n{context}\n\n현재 질문: {question}",
            ),
        ]
    )
    chain = prompt | ChatOpenAI(model=CHAT_MODEL, api_key=api_key, temperature=0) | StrOutputParser()

    with st.chat_message("assistant"):
        with st.spinner("문서 근거를 바탕으로 답변을 작성하고 있습니다..."):
            try:
                answer = chain.invoke(
                    {
                        "history": conversation_history,
                        "context": context,
                        "question": question,
                    }
                )
            except Exception as error:
                st.error(f"답변 생성에 실패했습니다: {error}")
                return
        st.markdown(answer)
        with st.expander("검색에 사용한 질문"):
            st.write(search_query)
        sources = [
            {
                "source": str(document.metadata["source"]),
                "page": int(document.metadata["page"]),
                "evidence": evidence_sentence(document.page_content),
            }
            for document in context_documents
        ]
        render_sources(sources, key_prefix=f"current-{len(st.session_state.messages)}")

    # 출처까지 대화 기록에 저장해야 다음 화면 실행에서도 버튼이 동작합니다.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "search_query": search_query,
        }
    )


if __name__ == "__main__":
    main()
