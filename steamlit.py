import streamlit as st
import openai

st.title("LLM 프롬프트 데모")

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("OpenAI API Key", type="password")

prompt = st.text_area("프롬프트를 입력하세요:")

if st.button("응답 받기") and prompt and openai.api_key:
    with st.spinner("LLM에 요청 중..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 사용 가능한 다른 모델
            messages=[{"role": "user", "content": prompt}]
        )
        st.write("응답:")
        st.success(response.choices[0].message["content"])