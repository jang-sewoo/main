import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")

    st.title("Private Data QA Chat :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    with st.sidebar:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            st.stop()

        process = st.button("Process")

    if process:
        # 문서 로드 및 처리
        documents = load_pdfs()
        text_chunks = split_text(documents)
        vectorstore = create_vectorstore(text_chunks)

        st.session_state.conversation = create_conversation_chain(vectorstore, openai_api_key)
        st.success("문서 처리가 완료되었습니다.")

    if st.session_state.conversation:
        query = st.text_input("질문을 입력하세요:")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                chain = st.session_state.conversation
                with st.spinner("답변 생성 중..."):
                    result = chain({"question": query})
                    response = result['answer']
                    source_documents = result['source_documents']
                    st.markdown(response)

                    # 참고 문서 표시
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(f"- {doc.metadata['source']}")

            st.session_state.messages.append({"role": "assistant", "content": response})

def load_pdfs():
    pdf_files = [
        "환보설비 운전보수 매뉴얼 (22E1002 지석) 240219.pdf",
        "한글 순환 아스콘 플랜트 매뉴얼 R2(120423).pdf",
        "저소음버너(한글)매뉴얼 - 체크본.pdf"
    ]

    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load_and_split()
        documents.extend(docs)

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def create_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()