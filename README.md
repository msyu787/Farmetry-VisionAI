# Farmetry


스마트팜 수경재배 데이터를 활용해 **상추·바질의 생육 분석, 생장량 예측, 병해 진단**을 수행하는 프로젝트입니다.

본 저장소에는 다음 항목이 포함되어 있습니다.

- 분석 및 학습 코드
- 처리된 CSV 데이터
- 클래스별 샘플 이미지
- 학습된 모델 가중치
- 결과 이미지 및 CSV

> 전체 재현을 위해서는 아래의 **데이터 준비** 항목을 참고하세요.

---

## 프로젝트 개요

Farmetry는 스마트팜 환경에서 수집한 이미지 및 환경 데이터를 기반으로 다음 작업을 수행합니다.

1. 상추 캐노피 이미지 분석
2. 상추 생장량 예측 및 환경 제어
3. 상추 병해 진단
4. 바질 병해 진단

---

## 노트북 구성

| 노트북 | 주제 | 핵심 기법 |
|---|---|---|
| `notebooks/01_lettuce_canopy_analysis.ipynb` | 상추 캐노피 분석 | HSV + ExG 잎 분할, 식생지수, 생장 변화율 |
| `notebooks/02_lettuce_growth_prediction.ipynb` | 상추 생장 예측 · 환경 제어 | 전날 환경 → 생장량 회귀, Optuna 최적화 |
| `notebooks/03_lettuce_disease_diagnosis.ipynb` | 상추 병해 진단 | ResNet18 분류 + YOLOv8 병반 탐지 |
| `notebooks/04_basil_disease_diagnosis.ipynb` | 바질 병해 진단 | ResNet18 2단계 진단 |

각 노트북은 저장소 루트를 자동으로 인식하므로, 별도의 경로 수정 없이 실행할 수 있습니다.

---

## 폴더 구조

```bash
Farmetry/
├── notebooks/                 # 노트북 4개
├── scripts/                   # notebook 02용 파이썬 모듈
├── data/
│   ├── lettuce_processed/     # 전처리 CSV
│   ├── lettuce_canopy/        # 캐노피 이미지 30장 (notebook 01)
│   ├── private_healthy_crops/ # 자체 데이터 잎 crop (notebook 04)
│   └── samples/               # 병해 샘플 (notebook 03/04)
├── models/                    # 학습된 가중치
├── results/                   # 결과 이미지 및 CSV
├── runs/                      # YOLOv8 산출물
└── yolov8n.pt                 # YOLOv8 사전학습 가중치
