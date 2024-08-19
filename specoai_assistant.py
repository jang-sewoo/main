import requests
import streamlit as st
import os

# OpenAI API 키
openai_api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}",
    "OpenAI-Beta": "assistants=v2"
}

# 비서를 통한 질문 응답 함수
def ask_question(assistant_id, query):
    url = f"https://api.openai.com/v1/assistants/{assistant_id}/messages"
    data = {
        "role": "user",
        "content": query
    }
    response = requests.post(url, headers=headers, json=data)
    
    # 응답 확인
    if response.status_code == 200:
        response_data = response.json()
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            st.error("응답을 처리하는 중 오류가 발생했습니다. 응답 형식이 예상과 다릅니다.")
            st.json(response_data)  # 전체 응답 데이터를 보여줍니다.
            return "응답을 처리하는 중 오류가 발생했습니다."
    else:
        st.error(f"API 요청 실패: {response.status_code}")
        st.json(response.json())  # 오류 응답 내용을 보여줍니다.
        return "API 요청 실패"

def main():
    st.set_page_config(page_title="Speco API", page_icon=":books:")

    st.title("Speco API Assistant")

    if "assistant_id" not in st.session_state:
        # 비서 ID를 세션 상태에 저장
        st.session_state.assistant_id = "asst_t3toIekxpUOO5BYzVniTy0Uc"
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

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
                assistant_response = ask_question(st.session_state.assistant_id, query)
                st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == '__main__':
    main()
