import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 데이터셋 로드
dataset = load_dataset("jinzzang23/test_data_4")

# 모델과 토크나이저 로드
model_name = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 응답 생성 함수
def generate_response(prompt):
    # 입력 프롬프트 인코딩
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")

    # 응답 생성
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    # 응답 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit 앱 설정
st.title("챗봇")
st.write("안녕하세요! 무엇을 도와드릴까요?")

# 사용자 입력 받기
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input)
    st.write("Chatbot:", response)
