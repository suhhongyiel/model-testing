import streamlit as st
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import matplotlib.pyplot as plt
# 위험도 색상 함수 정의
def get_risk_color(probability):
    """위험도를 색상으로 구분하는 함수"""
    if probability <= 30:
        return 'green'
    elif 31 <= probability <= 70:
        return 'yellow'
    else:
        return 'red'


# 모델 정의
class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # 마지막 시퀀스의 출력을 사용
        return output

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 마지막 시퀀스의 출력을 사용
        return output

# Attention-LSTM 모델 정의
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])  # 마지막 시퀀스의 출력을 사용
        return output

# 모델 로드 함수
def load_model(model_name, model_path, input_dim, hidden_dim, output_dim, num_layers):
    if model_name == "GRU":
        model = SimpleGRU(input_dim, hidden_dim, output_dim, num_layers)
    elif model_name == "LSTM":
        model = SimpleLSTM(input_dim, hidden_dim, output_dim, num_layers)
    elif model_name == "Attention-LSTM":
        hidden_dim = 32  # 이전 설정에서 사용한 hidden_dim
        num_layers = 1  # 학습 시 사용한 num_layers
        model = AttentionLSTM(input_dim, hidden_dim, output_dim, num_layers)
    elif model_name == "RNN":
        model = SimpleRNN(input_dim, hidden_dim, output_dim, num_layers)
        
    model_path = model_name + '_model.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 모델을 평가 모드로 전환
    return model

# Streamlit 웹 애플리케이션 시작
st.title("User-Based Emergency Visit Prediction")

# 모델 선택
model_name = st.selectbox("Choose a model", ["GRU", "LSTM", "Attention-LSTM", "RNN"])

# 학습된 모델 로드
model_path = "model.pt"  # 각 모델에 맞는 가중치 파일 경로를 지정해야 함
input_dim = 15  # 선택한 입력 차원 수 (선택한 열의 개수)
hidden_dim = 8  # 이전 설정에서 사용한 hidden_dim
output_dim = 1  # 출력 차원 (이진 분류)
num_layers = 2  # 모델 레이어 수

model = load_model(model_name, model_path, input_dim, hidden_dim, output_dim, num_layers)

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # CSV 파일 읽기
    data = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(data.head())  # 데이터를 표로 표시

    # user_id 별로 데이터 그룹화
    if 'user_id' in data.columns:
        user_groups = data.groupby('user_id')

        # 추론 실행
        if st.button("Run Inference"):
            results = []

            for user_id, group in user_groups:
                # 필요한 열만 추출
                X = group[['activeZoneMinutes', 'fatBurnActiveZoneMinutes', 'cardioActiveZoneMinutes',
                           'totalTimeInBed', 'stages_deep', 'stages_light', 'stages_rem', 'stages_wake',
                           'distance', 'steps', 'avg_hr', 'max_hr', 'min_hr', 'std_hr', 'daily_compliance_rate']]
                X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가

                with torch.no_grad():
                    output = model(X_tensor)
                    probability = torch.sigmoid(output).item() * 100  # 확률로 변환

                # 위험도 색상 결정
                risk_color = get_risk_color(probability)

                # 결과 저장
                results.append({
                    'user_id': user_id,
                    'Emergency Visit Probability (%)': f"{probability:.2f}",
                    'Risk Level': risk_color
                })

            # 결과를 DataFrame으로 변환
            result_df = pd.DataFrame(results)

            # 결과를 표로 표시
            st.write("Inference Results by User:")
            st.dataframe(result_df)

            # 각 사용자별로 위험도 표시
            st.write("Risk Level by Color (Green: Low, Yellow: Medium, Red: High)")
            for _, row in result_df.iterrows():
                user_id = row['user_id']
                probability = row['Emergency Visit Probability (%)']
                risk_level = row['Risk Level']
                st.markdown(f"**User {user_id}:** Emergency Visit Probability: **{probability}%**", unsafe_allow_html=True)
                st.markdown(f"<span style='color:{risk_level}'>Risk Level: {risk_level}</span>", unsafe_allow_html=True)