import streamlit as st
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# 모델 로드 함수
def load_model_pt(model_path, input_dim, hidden_dim, output_dim, num_layers):
    class SimpleGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super(SimpleGRU, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            gru_out, _ = self.gru(x)
            output = self.fc(gru_out[:, -1, :])  # 마지막 시퀀스의 출력을 사용
            return output

    model = SimpleGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 모델을 평가 모드로 전환
    return model

# Streamlit 웹 애플리케이션 시작
st.title("GRU Model Inference Web App")

# 학습된 모델 로드
model_path = "gru_model.pt"  # 저장된 모델 경로
input_dim = 14  # 선택한 입력 차원 수 (선택한 열의 개수)
hidden_dim = 8  # 이전 설정에서 사용한 hidden_dim
output_dim = 1  # 출력 차원 (이진 분류)
num_layers = 2  # GRU 레이어 수

model = load_model_pt(model_path, input_dim, hidden_dim, output_dim, num_layers)

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # CSV 파일 읽기
    data = pd.read_csv(uploaded_file)

    # 데이터 전처리
    st.write("Original Data:")
    st.write(data.head())

    # 'Output' 값이 2인 경우 0으로 치환 (필요한 경우)
    if 'Output' in data.columns:
        data['Output'] = data['Output'].replace(2, 0)

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

    # 시퀀스 데이터를 패딩 (길이가 다를 경우 대비)
    X_padded = pad_sequence(X, batch_first=True)

    # 추론 실행
    if st.button("Run Inference"):
        with torch.no_grad():
            output = model(X_padded)
            predictions = torch.sigmoid(output).numpy()

        # 결과 출력
        st.write("Predictions:")
        st.write(predictions)
