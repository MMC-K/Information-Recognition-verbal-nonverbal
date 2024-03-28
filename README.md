# Information-Recognition-verbal-nonverbal


[VL-KE-T5](https://github.com/AIRC-KETI/VL-KE-T5) 모델을 이용한 multi-modal emotion classification

## 필요 패키지 설치

```bash
    pip install torch
    pip install -r requirements.txt
```


## 모델 사용 방법
```bash
    CUDA_VISIBLE_DEVICES='0' python training_emotion.py --batch_size 8 --epochs 30
```

## Download

### Models


[VisionT5MeanBiEncoder](https://drive.google.com/file/d/1Cq9GldmJz7qpHnJQoXrnA8YD317_3lOz/view?usp=sharing)

--------------------

## Data 구축

AIHub의 감성 대화 말뭉치와 한국인 감정인식을 위한 복합 영상 데이터를 이용하여 json 데이터 파일을 구축했습니다.

### 1) 감성 대화 말뭉치
- 사용자와 시스템 간의 3턴 대화 문장과 정보를 가진 xlsx 및 json 파일로 구성되어 있습니다.
- 나이는 청소년, 청년, 중년, 노년으로 분류되어 있습니다.
- 중립 레이블에 해당하는 데이터가 존재하지 않습니다.

### 2) 한국인 감정인식을 위한 복합 영상 데이터
- 지정된 감정을 연기한 사람의 얼굴 이미지 파일과 해당 정보를 담은 json 파일로 구성되어 있습니다.
- 나이는 10대, 20대, 30대, 40대, 50대, 60대로 분류되어 있습니다.


### 데이터 전처리
#### 1) 감성 대화 말뭉치
- 사람문장1+사람문장2은 기쁨, 슬픔, 분노, 당황, 불안, 상처로 분류되고, 시스템문장1은 중립으로 분류했습니다.

#### 2) 한국인 감정인식을 위한 복합 영상 데이터
- 지정된 감정("faceExp_uploader")과 annotator 3인("annot_A/B/C")의 감정("faceExp")이 모두 일치하는 데이터만 추출했습니다.
- 감성 대화 말뭉치에 맞추어 10대는 청소년, 20대와 30대는 청년, 40대와 50대는 중년, 60대는 노년으로 분류했습니다.

### 최종 데이터
- 성별은 남, 여로, 나이는 청소년, 청년, 중년, 노년으로 분류했습니다.
- 데이터 수는 두 데이터 중 작은 데이터의 수에 맞추어 잘랐습니다.
- train:valid:test=8:1:1 (데이터셋 간 겹치는 인물이 존재하지 않도록 처리했습니다.)
