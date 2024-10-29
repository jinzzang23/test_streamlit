import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("Test Chatbot by HuggingFace")

# 허깅페이스 모델 로드
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        device_map = "auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

tokenizer, model = load_model()

# 파일 업로드
uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")
query = st.text_area("Enter your query in Korean")

# 처리
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head(3))

    if query:
        try:
            df_into = df.describe().to_string()
            prompt = f"다음 데이터를 기반으로 질문에 답하세요 : {df_into}\n질문: {query}\n답변:"

            # 입력 텍스트 토크나이즈
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # 모델 통해서 답변 생성
            outputs = model.generate(inputs.input_ids, max_length=300,  do_sample=True, top_p=0.95, top_k=60)

            # 생성 답변 디코딩
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답 출력
            st.write("Chatbot Response:")
            st.write(response)

        except Exception as e:
            st.error(f"An error Occurred: {str(e)}")
