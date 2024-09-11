import streamlit as st
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
st.title("Model Inference Web App")

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
    # data.fillna(data.mean(), inplace=True)
    # 데이터 전처리
    st.write("Original Data:")
    st.write(data.head())

    # 'Output' 값이 2인 경우 0으로 치환 (필요한 경우)
    if 'Output' in data.columns:
        data['Output'] = data['Output'].replace(2, 0)




    # Output 라벨 분포 확인
    st.write("Output Label Distribution:")
    label_counts = data['Output'].value_counts()

    # Output 라벨 분포를 표로 출력
    st.write(label_counts)

    # Output 라벨 분포를 시각화 (막대 그래프)
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Output Labels')
    ax.set_xlabel('Output Label')
    ax.set_ylabel('Count')
    st.pyplot(fig)






    # 'Sequence' 열 생성
    data['Sequence'] = None
    sequence_id = 0

    # 시퀀스 생성 로직
    for idx in range(len(data) - 1, -1, -1):
        if pd.notna(data.loc[idx, 'Output']):
            sequence_id += 1
            data.loc[idx, 'Sequence'] = sequence_id
            j = idx - 1
            while j >= 0 and pd.isna(data.loc[j, 'Output']):
                data.loc[j, 'Sequence'] = sequence_id
                j -= 1

    # 시퀀스 그룹핑
    sequences = data.dropna(subset=['Sequence'])
    sequence_groups = sequences.groupby('Sequence')

    # 선택할 열만 필터링
    selected_columns = [
        'activeZoneMinutes', 'fatBurnActiveZoneMinutes', 'cardioActiveZoneMinutes',
        'totalTimeInBed', 'stages_deep', 'stages_light', 'stages_rem', 'stages_wake',
        'distance', 'steps', 'avg_hr', 'max_hr', 'min_hr', 'std_hr', 'daily_compliance_rate'
    ]

    # 추론을 위한 데이터 입력
    X = []
    for name, group in sequence_groups:
        group = group[selected_columns]
        X.append(torch.tensor(group.values, dtype=torch.float32))

        # 각 시퀀스의 raw 데이터를 Streamlit에 표시
        st.write(f"Raw data for Sequence {name}:")
        st.dataframe(group)  # 시퀀스별 데이터 테이블로 표시

    # 시퀀스 데이터를 패딩 (길이가 다를 경우 대비)
    X_padded = pad_sequence(X, batch_first=True)

    # 추론 실행
    if st.button("Run Inference"):
        with torch.no_grad():
            output = model(X_padded)
            predictions = torch.sigmoid(output).numpy()

        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame({
            'Sample': range(1, len(predictions) + 1),
            'Class 1 Probability': [f"{pred[0]:.2f}" for pred in predictions],
            'Predicted Class': ['Class 1' if pred >= 0.5 else 'Class 0' for pred in predictions]
        })

        # 결과를 표로 표시
        st.write("Inference Results:")
        st.dataframe(result_df)

        # 요약된 결과 표시
        num_class_1 = result_df['Predicted Class'].value_counts().get('Class 1', 0)
        num_class_0 = result_df['Predicted Class'].value_counts().get('Class 0', 0)
        st.write(f"Number of samples classified as **Class 1**: {num_class_1}")
        st.write(f"Number of samples classified as **Class 0**: {num_class_0}")
