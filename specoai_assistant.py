import streamlit as st
import openai
import os

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
    response = openai.Completion.create(
        model="text-davinci-003",  # 또는 사용 가능한 최신 모델로 변경
        prompt=f"User: {query}\nAssistant:",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    main()
