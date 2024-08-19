import streamlit as st

# Streamlit 웹앱 제목
st.title("Simple Chat Application")

# 채팅 메시지를 저장할 리스트
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사용자가 입력한 채팅 메시지
user_input = st.text_input("Enter your message:")

# 'Send' 버튼을 누르면 메시지를 리스트에 추가
if st.button("Send"):
    if user_input:
        st.session_state.messages.append(user_input)
        user_input = ""  # 입력 필드를 초기화

# 채팅 메시지를 출력
if st.session_state.messages:
    st.write("### Chat History")
    for msg in st.session_state.messages:
        st.write(msg)
