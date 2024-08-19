import streamlit as st
import openai
import os
import pickle
from loguru import logger
import gdown

CACHE_FILE = "cached_vectorstore.pkl"

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_SPECO Data :red[QA Chat]_ :books:")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", 
                                      "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key is None:
        st.info("OpenAI API 키를 환경 변수로 설정해주세요.")
        st.stop()

    # 세션 상태에서 대화 기록을 불러오기
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 로직
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = get_assistant_response(query, openai_api_key)
                st.markdown(response)

        # 비서 응답을 대화 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_assistant_response(query, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 도움이 되는 비서입니다."},
            {"role": "user", "content": query}
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    main()
