# 뇌동맥류 진단 모델 - ResNet50, Spatial attention, LSTM을 기반으로 
고강빈, 김동우, 김민서
<br>
{sweetrainforyou, kimdw0611, minskim010203}@gmail.com

해당 보고서는 2023 K-ium 의료인공지능경진대회 참가팀 ‘코딩베셀즈’의 결과 레포트입니다.

## Abstract
본 모델은 실제 뇌동맥류를 진단하는 과정에 근거하여 디자인되었다. 
해상도 손실을 최소화하여 resizing하였고, 배경이 아닌 혈관에 집중할 수 있도록 하기 위해 이미지의 색상을 반전(invert)하여 혈관 부분에 높은 pixel value를 주었으며, 혈관 이미지를 보강하기 위해 이미지의 선명도를 높이는 전처리 과정을 거쳤다.
모델은 조영제를 투입하여 얻은 서로 다른 각도의 Angiogram 이미지 8장을 입력 받는다. 입력 받은 8장의 이미지로부터 ResNet50을 통해 feature를 추출하고, 추출된 feature는 Spatial-attention layer를 거치며 뇌동맥류와 관련하여 주목해야할 부분에 대해서 mapping한다. 이후 이미지 시퀀스 형태로 LSTM layer를 거치며 모델은 뇌동맥류 진단에 대한 정보를 모델링하게 된다.

## Introduction
동맥류(Aneurysm)는 혈관 정상 직경의 50%를 초과하는 비정상적인 확장 또는 부풀어 오른 약화된 영역을 의미한다. 동맥류에는 다양한 종류가 있지만, 그중 뇌혈관에 발생하는 동맥류를 뇌동맥류(Cerebral Aneurysm)라고 한다. 뇌동맥류는 갑작스러운 두통, 메스꺼움, 구토, 시각 장애, 의식 상실을 유발할 수 있으며, 동맥류의 파열은 치명적인 출혈을 야기하고 환자를 사망에 이르게 할 수도 있기 때문에 증상이 발생하기 전 사전에 탐지되고 예방되는 것이 중요하다. 
이에 해당 보고서에서는 딥러닝(Deep Learning) 기술을 이용하여 뇌혈관조영술 (Cerebral Angiography) 영상을 분석하고, 이를 통해 빠르고 정확한 탐지를 돕는 알고리즘을 제시하였다.

## Dataset
해당 대회에서 제공하는 데이터는 1,127명의 환자로부터 수집된 익명화된 뇌혈관조영술 이미지 데이터셋으로, 각 환자별로 서로 다른 위치에서 촬영한 8장의 이미지를 가지므로 총 9,016장의 이미지로 이루어져있다. 1,127명의 환자 중 Aneurysm 레이블이 0인 정상군은 604명, Aneurysm 레이블이 1인 환자군은 523명이다.

## Methods
### Model Architecture
해당 과제를 수행하기 위해 모델 조사를 수행하였다. 최근 대부분의 Vision Task에서는 Transformer와 같은 Self-attention based 모델이 우세하지만 이는 학습에 있어 대량의 데이터를 필요로 한다는 단점이 존재한다. 추가적인 Unlabeled data를 이용하여 pretrain을 진행할 경우 learning curve가 steep해져 적은 데이터로도 Transformer를 학습시킬 수 있다는 연구 결과가 있으나 해당 대회의 경우 외부 데이터 사용을 허가하지 않기에 사용할 수 없다고 판단하였다. 이와 같은 이유로, 기본 베이스라인 모델을 Self-attention based가 아닌 CNN-based 모델로 선정하였다.
다만, 기본적인 CNN 모델은 하나의 이미지만을 받을 수 있는데 해당 태스크에서는 한번에 8장의 이미지를 동시에 입력 받아 Aneurysm의 위치 및 여부를 판단해야 한다. 각 1장의 이미지를 입력받는 CNN 모델을 8개 Ensemble 해 줌으로써 간단하게 접근 가능하지만, 이 경우 전체 모델 파이프라인의 크기가 커지므로 임상적 사용에 있어 제약이 있을 수 있다고 판단하였다. 따라서 여러 개의 CNN을 사용하는 대신 RNN 계열의 모델을 추가하여 8장의 이미지를 연속으로 입력 받아 처리할 수 있도록 하였다.

사용 모델의 전체적인 구성은 실제 Aneurysm의 진단 과정에 근거하여 디자인 되었다. 
모델은 딥러닝 신경망으로 구성되며, CNN 계열 모델인 ResNet50, RNN 계열 모델인 LSTM을 활용하였고, 8장 이미지 각각에 대하여 잡음 회피 및 동맥 탐지에 초점을 둘 수 있도록 Spatial-attention Mechanism을 활용하였다.
딥러닝 신경망 모델 구조에서 ResNet50을 사용하여 이미지의 feature를 추출하고, Spatial-attention layer로 Attention map을 생성하는 과정은 개별 혈관 조영 이미지를 관찰하고 혈관의 주요 특징 및 이상 징후를 포착하는 과정과 유사하다. 이후 LSTM을 거쳐 여러 장의 이미지를 차례로 처리하는 것은 포착된 이상 징후를 바탕으로 다각도에서 촬영한 조영술 이미지를 검토함으로써 혈관의 3차원적 특성을 고려하여 진단하는 과정과 유사하다. 모델을 통해 입력된 뇌혈관 조영술 이미지를 처리하고 분류를 수행하는 과정에 대한 자세한 설명은 다음과 같다.

먼저 ResNet50 layer에서 8장의 720*720 크기의 RGB 이미지를 입력으로 받는다. ResNet50은 사전 훈련된 가중치를 사용하여 입력 이미지의 특징을 추출하고 3D Tensor 형태의 특징 벡터를 출력하게 된다. 이렇게 출력된 특징 벡터는 이어지는 Spatial-attention layer에 입력되어 Attention map을 생성하게 되고, Multiply layer를 통해 입력에 곱해짐으로써 이미지 내에서 중요한 부분(i.e. 뇌동맥류)에 집중하는 중요도 반영 출력을 생성한다. 
이후 Flatten layer에 의해 1D 벡터로 변환된 각각의 이미지 시퀀스는 다시 시간 축 방향으로 연결되어 2D 텐서를 형성하게 되고, 이는 LSTM layer에 입력되어 각 조영술 이미지 시퀀스 간의 상관 관계를 학습하고 뇌동맥류 진단에 대한 정보를 모델링하게 된다. 즉, 부위별 이미지로 학습을 진행하는 것이 아닌, 환자 한 명의 다각도 CT 사진을 분석하고 혈관에 초점을 두어 정확도를 높이게 된다. 
최종적으로 LSTM layer의 출력은 Dense layer를 거쳐 22개 레이블에 대한 확률 값을 출력하게 되고, 이를 통해 모델은 입력 이미지 시퀀스로부터 Aneurysm의 위치 및 여부에 대한 예측을 수행하게 된다.
모델의 구조는 다음과 같다.

![image](https://github.com/willystumblr/k-ium-coding-vessels/assets/76294398/179dbd54-4fbf-47c2-b3bf-e458f9b87b6e)
<br>
[Figure 1] 모델 구조. ResNet50, Spatial Attention, LSTM으로 구성되어 있다.

### Experiments
보다 높은 AUC score와 Accuracy를 위해, 다음과 같은 데이터셋 전처리 과정을 모델 전반부에 추가하여 모델을 학습하였다.
주어진 이미지를 모두 균일한 크기로 resize하되, 해상도 손실을 최소화하기 위해 주어진 이미지가 가지고 있는 화소 값 중 최소 화소인 720*720을 채택하였다.
혈관 이미지를 보강하기 위해 이미지의 선명도를 높인다. 이 과정은 조영제가 혈관에 균일하게 퍼지지 않은 문제를 해결한다.
배경이 아닌 혈관에 집중할 수 있도록 하기 위해 이미지의 색상을 반전(invert)하여 혈관 부분에 높은 pixel value를 주었다.

![image](https://github.com/willystumblr/k-ium-coding-vessels/assets/76294398/27ff5e84-de14-4bbe-a096-5bb0d7bb7b40)
<br>
[Figure 2] Data Augmentation. 이미지 선명도를 높이고 색상 반전을 줌으로써 다각도로 훈련을 시도하였다.

모델의 training 및 evaluation은 Google Colaboratory 환경에서 A100 GPU 1장을 이용하여 진행되었다. 사용된 주요 라이브러리로는 tensorflow, keras, sklearn, numpy, pandas를 활용하였다. 자세한 버전 및 라이브러리 정보는 requirements.txt에 기록되어 있다.

Hyperparameter, 활성화함수, 최적화함수는 다음과 같다.
learning rate: 초기 1e-5로 시작, time-based decay scheduler를 이용하여 epoch가 진행됨에 따라 감소하도록 설정
활성화함수: ReLU(1), Sigmoid(2)
최적화함수: Adam optimizer

## Results
제공된 /test_set/test.csv 파일의 환자 번호와 그에 상응하는 이미지 파일들로 evaluation set을 구성, 이를 바탕으로 예측(predition)을 진행한 결과 아래와 같은 AUROC score와 뇌동맥류 위치에 대한 accuracy를 얻을 수 있었다.
<br>
> AUROC of the provided model
> 
> 0.5958132045088567
>
> Accuracy for locations
>
> 0.9619047619047619

## Future Works
이번 대회를 통해 뇌동맥류의 위치 및 여부를 진단하는 인공지능 모델을 개발하였다. 이또한 물론 유의미한 작업이지만, 비파열 뇌동맥류의 경우 같은 위치에 발생하더라도 그 크기 및 여러가지 risk factor에 따라 응급도가 천차만별일 수 있기에 해당 내용에 대한 고려 또한 필요할 것이다.
의료 시장에서 결국 원하고 필요로 하는 기술은 뇌동맥류의 크기를 빠르고 정확하게 탐지하고, 환자의 과거정보 및 위험인자 등을 함께 입력받음으로써 응급도를 알려주는 기술이라고 생각된다. 따라서 추후 진행할 연구에서는 1) 뇌동맥류의 크기 2) 뇌동맥류의 위치를 빠르게 탐지하고, 3) 환자의 과거력을 포함한 위험인자 데이터를 추가로 입력받음으로써 종합적인 응급위험도를 알려주는 모델을 개발해보고자 한다.
뇌동맥류의 크기 및 위치는 U-Net 등의 Segmentation 모델을 활용할 수 있을 것이라고 생각되며, 테이블 형태 혹은 그래프 형태의 환자 과거력을 이미지와 함께 입력받을 수 있는 Multi-modal 모델을 사용하여 응급도 분류를 수행할 수 있도록 할 경우 임상적으로 유의미한 연구를 진행할 수 있을 것이라 생각된다.

## Reference
[1] "Aneurysm." Johns Hopkins Medicine.
https://www.hopkinsmedicine.org/health/conditions-and-diseases/aneurysm.
[2] "CNN Long Short-Term Memory Networks." Machine Learning Mastery. https://machinelearningmastery.com/cnn-long-short-term-memory-networks/.

## Appendix
Files and program execution
제출물은 다음과 같은 파일과 디렉토리로 구성된다.
kium-output/: 모델의 checkpoint가 저장된 디렉토리.
utils/SpatialAttention.py: Spatial Attention Mechanism이 구현된 class가 정의된 파일.
utils/ImagePreprocess.py: 이미지를 모델의 입력값으로 넘겨주기 전, 모델의 입력값 shape에 맞게 crop/resize 해주는 method가 정의된 파일
main.py: 모델의 인스턴스를 생성하고 저장된 체크포인트를 불러와 예측을 진행하여 output.csv를 생성하는 파일.
requirements.txt: 모델 검증 전, 프로그램 작동을 위해 필요한 오픈 소스 라이브러리와 패키지를 설치하는 파일.

파일 실행 순서는 다음과 같다.
터미널에서 위에 나열된 파일이 위치한 디렉토리로 이동한다.
터미널에서 아래 명령어를 실행한다.
pip install -r requirements.txt
필요한 모든 라이브러리가 설치되면, evaluate.py 파일에서 정의된 CSV_PATH, CSV_FILENAME, IMG_PATH, IMG_FILE_EXTENSION을 각각 재정의한다.
현재 디렉토리에서 python main.py를 실행한다.

코드 및 자료는 다음에서 확인할 수 있다.
https://github.com/willystumblr/k-ium-coding-vessels
