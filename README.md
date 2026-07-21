# Farmetry

> **AI-powered Smart Farming Startup Project**

스마트팜(수경재배) 데이터로 상추·바질의 **생육 분석 · 생장량 예측 · 병해 진단**을 다루는 프로젝트.

<img width="984" height="789" alt="farmetr_git" src="https://github.com/user-attachments/assets/b24a70a3-c32a-4771-a556-c639ddf0e198" />



- 코드 + 처리된 CSV + 클래스별 샘플 이미지 + 학습된 가중치 포함
- 데이터셋은 [데이터 준비](#데이터-준비) 참고

## 노트북

| 노트북 | 주제 | 핵심 기법 |
| --- | --- | --- |
| `notebooks/01_lettuce_canopy_analysis.ipynb` | 상추 캐노피 분석 | 잎 분할, 식생지수, 생장 변화율 파악 모델 개발 |
| `notebooks/02_lettuce_growth_prediction.ipynb` | 상추 생장 예측·환경 제어 | 전날 환경 → 생장량 회귀, Optuna 최적화 |
| `notebooks/03_lettuce_disease_diagnosis.ipynb` | 상추 병해 진단 | ResNet18 분류 + YOLOv8 병반 탐지 |
| `notebooks/04_basil_disease_diagnosis.ipynb` | 바질 병해 진단 | ResNet18 2 class 분류|
| `notebooks/05_our_data.ipynb` | Farmetry 데이터 적용 | notebook2 모델 적용 - 생장량 최대화 방향 설정 |

- 저장소 루트를 자동 인식하므로 경로 수정 없이 실행 가능

## 폴더 구조

```
Farmetry/
├── notebooks/            # 노트북 5개
├── scripts/              # notebook 02용 파이썬 모듈
├── data/
│   ├── lettuce_processed/    # 전처리 CSV
│   ├── lettuce_canopy/       # 캐노피 이미지 30장 (notebook 01)
│   ├── private_healthy_crops/ # 자체 데이터 잎 crop (notebook 04)
│   └── samples/              # 병해 샘플 (notebook 03/04)
├── models/              # 학습된 가중치
├── results/             # 결과 이미지·CSV
├── runs/                # YOLOv8 산출물
└── yolov8n.pt           # YOLOv8 사전학습 가중치
```


## 데이터 준비

- **상추 생육 (notebook 02)**: https://zenodo.org/records/17041810
- **상추 병해 (notebook 03)**: AI hub 시설 작물 질병 진단 이미지 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=153)
- **바질 병해 (notebook 04)**: https://data.mendeley.com/datasets/7hmt25zc56/1
- **private data**: data/Farmetry
