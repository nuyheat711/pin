import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile
import os

# 페이지 설정
st.set_page_config(
    page_title="PDF RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API 키 설정
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# 프롬프트 템플릿
CUSTOM_PROMPT = PromptTemplate(
    template="""당신은 PDF 문서 내용을 기반으로 질문에 답변하는 전문 어시스턴트입니다.

주어진 컨텍스트를 사용하여 질문에 답변하세요.
답변할 때 다음 규칙을 따르세요:
1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
2. 컨텍스트에서 답을 찾을 수 없다면, "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 정직하게 답변하세요.
3. 추측하거나 문서에 없는 정보를 만들어내지 마세요.
4. 답변은 명확하고 간결하게 작성하세요.
5. 가능하다면 문서의 어느 부분을 참조했는지 언급하세요.

컨텍스트:
{context}

대화 기록:
{chat_history}

질문: {question}

답변:""",
    input_variables=["context", "chat_history", "question"]
)


@st.cache_resource
def initialize_llm():
    """LLM 초기화"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )


@st.cache_resource
def initialize_embeddings():
    """임베딩 모델 초기화"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


def load_and_process_pdf(pdf_file):
    """PDF 파일 로드 및 처리"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # PDF 로드
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        return splits
    finally:
        os.unlink(tmp_path)


def create_vectorstore(documents):
    """벡터 스토어 생성"""
    embeddings = initialize_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def create_conversation_chain(vectorstore):
    """대화 체인 생성"""
    llm = initialize_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True,
        verbose=False
    )

    return chain


def main():
    # 헤더
    st.markdown('''<div class="main-header"><h1>📚 PDF RAG 챗봇</h1><p>PDF 문서를 업로드하고 질문하세요!</p></div>''', unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        st.header("📄 문서 업로드")

        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=["pdf"],
            help="PDF 파일을 업로드하면 내용을 분석하여 질문에 답변합니다."
        )

        # 기본 test.pdf 사용 옵션
        use_default = st.checkbox("기본 test.pdf 사용", value=False)

        if use_default and os.path.exists("test.pdf"):
            with open("test.pdf", "rb") as f:
                uploaded_file = f
                st.success("✅ test.pdf 로드됨")

        st.markdown('''<div class="sidebar-info">''', unsafe_allow_html=True)
        st.markdown("""
        ### 사용 방법
        1. PDF 파일을 업로드합니다
        2. 문서 처리가 완료될 때까지 기다립니다
        3. 채팅창에 질문을 입력합니다
        4. AI가 문서 내용을 기반으로 답변합니다
        """)
        st.markdown('''</div>''', unsafe_allow_html=True)

        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain = None
            st.rerun()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # PDF 처리
    if uploaded_file is not None:
        if st.session_state.vectorstore is None:
            with st.spinner("📖 PDF 문서를 분석 중입니다..."):
                try:
                    # PDF 로드 및 처리
                    if hasattr(uploaded_file, 'getvalue'):
                        documents = load_and_process_pdf(uploaded_file)
                    else:
                        loader = PyPDFLoader("test.pdf")
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        documents = text_splitter.split_documents(documents)

                    # 벡터 스토어 생성
                    st.session_state.vectorstore = create_vectorstore(documents)

                    # 대화 체인 생성
                    st.session_state.chain = create_conversation_chain(
                        st.session_state.vectorstore
                    )

                    st.success(f"✅ 문서 처리 완료! ({len(documents)}개의 청크로 분할됨)")

                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {str(e)}")
                    return

    # 채팅 인터페이스
    chat_container = st.container()

    with chat_container:
        # 이전 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📎 참조 문서"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**출처 {i}** (페이지 {source.get('page', 'N/A')})")
                            st.caption(source.get('content', '')[:300] + "...")

    # 사용자 입력
    if prompt := st.chat_input("PDF 문서에 대해 질문하세요...", disabled=st.session_state.chain is None):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("생각 중..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response["answer"]

                    # 소스 문서 정보 추출
                    sources = []
                    if "source_documents" in response:
                        for doc in response["source_documents"]:
                            sources.append({
                                "page": doc.metadata.get("page", "N/A"),
                                "content": doc.page_content
                            })

                    st.markdown(answer)

                    # 참조 문서 표시
                    if sources:
                        with st.expander("📎 참조 문서"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**출처 {i}** (페이지 {source['page']})")
                                st.caption(source['content'][:300] + "...")

                    # 어시스턴트 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # PDF 미업로드 시 안내 메시지
    if st.session_state.chain is None:
        st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")


if __name__ == "__main__":
    main()
