# -*- coding: utf-8 -*-
"""
플라스크를 활용하여 REST API로 파이토치 배포하기 
========================================================
**저자**: `Avinash Sajjanshetty <https://avi.im>`_
**번역**: `송진영 <https://github.com/diligejy>`_

이번 튜토리얼에서는, 플라스크를 활용해서 파이토치 모델을 배포하고 REST API방식을 사용해서 모델 추론을 해보겠습니다. 
특히, 이번엔 이미지를 탐지하도록 사전학습된 Dense 121 모델을 배포해보겠습니다.

.. 공지:: 여기 있는 모든 코드는 MIT 라이센스하에 배포되며 `깃허브 <https://github.com/avinassh/pytorch-flask-api>`_.에서 사용 가능합니다.

이 글은 실상황(in production) 파이토치 모델 배포 튜토리얼 시리즈의 첫 번째 글입니다. 
플라스크를 이렇게 사용하는 건 여러분의 파이토치 모델을 서빙하는 가장 쉬운 방법일 것입니다. 
하지만 고성능 요구사항에는 적합하지 않을 수 있습니다. 

그럴 경우에 아래의 문서를 참조하시길 바랍니다.

    - 만약 TorchScript에 익숙하시다면, `C++에서 TorchScript 모델 로딩하기<https://pytorch.org/tutorials/advanced/cpp_export.html>`_ 튜토리얼을 참고하실 수 있습니다.

    - 만약 TorchScript를 까먹으셨다면, `TorchScript 입문<https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_ 튜토리얼을 참고하실 수 있습니다.
"""


######################################################################
# API 정의
# --------------
#
# 먼저 API 엔드포인트를 요청(request)과 응답(response) 타입으로 정의하겠습니다.
# API 엔드포인트의 형태는 ``/predict``가 될 것입니다. 
# 이 엔드포인트는 HTTP POST 방식으로 이미지를 포함하는 ``file`` 매개변수를 요청합니다.
# 응답은 JSON 형식이 될 것입니다.
# 응답에는 다음과 같은 예측치가 포함됩니다 
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
#

######################################################################
# 의존사항(Dependencies)
# ------------------------
#
# 아래의 명령어를 실행시켜서 실행에 필요한 의존사항을 설치하세요:
#
# ::
#
#     $ pip install Flask==2.0.1 torchvision==0.10.0


######################################################################
# 간단한 웹 서버
# -----------------
#
# 아래는 플라스크 공식문서에서 가져온 간단한 웹 서버 예시입니다.


from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

###############################################################################
# 위에 있는 코드를 ``app.py`` 파일로 저장하신 다음 아래와 같이 입력하시면 플라스크 개발 서버를 실행하실 수 있습니다:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

###############################################################################
# 웹 브라우저를 여신 다음 ``http://localhost:5000/``을 띄워보세요. ``Hello World!``라는 텍스트를 보실 수 있을 겁니다.

###############################################################################
# 위의 코드를 약간 바꿔서 우리 API 정의에 맞게 만들어보겠습니다.  
# 첫째, 메서드 이름을 ``predict``로 바꿔보겠습니다. 
# 엔드포인트 주소는 ``/predict``로 바꾸겠습니다. 
# 이미지 파일은 HTTP POST 요청을 따라 전송될 것이기 때문에, POST 방식만 받아서 업데이트할 것입니다.
# Since the image files will be sent via HTTP POST requests, we will update it so that it also accepts only POST
# 요청:

@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'

###############################################################################
# 응답 방식을 바꿔서 ImageNet 클래스 id와 name을 포함하는 JSON으로 반환하도록 만들어보겠습니다.
# 수정된 ``app.py`` 파일은 이렇게 바뀔 것입니다: 

from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


######################################################################
# 추론
# -----------------
# 다음 섹션에서는 추론 코드 작성을 중점으로 해보겠습니다.
# 여기에는 두 부분이 포함됩니다. 
# 하나는 DenseNet에 공급할 수 있도록 이미지를 준비하는 부분이고 다음은 모델에서 실제 예측을 가져오는 코드를 작성하는 부분입니다.
#
# 이미지 준비하기
# ~~~~~~~~~~~~~~~~~~~
#
# DenseNet 모델에서는 이미지가 224 x 224 크기의 3채널 RGB 이미지여야 합니다.
# 또한 필요한 평균 및 표준 편차 값으로 이미지 텐서를 정규화합니다.
# 더 알아보고 싶은 분은 `여기 <https://pytorch.org/vision/stable/models.html>`_ 를 참조하세요. 
# 
# ``torchvision``라이브러리에서 ``transforms``을 사용해서 이미지를 요구사항에 맞춰 변환해주는 transform pipeline을 만들어보겠습니다.
# transforms에 대해 더 알아보고 싶으시다면 `여기 <https://pytorch.org/vision/stable/transforms.html>`_ 를 참조하세요.

import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


######################################################################
# 위에 있는 메소드는 이미지 데이터를 바이트로 바꿔주고, 여러 변환(transforms) 단계를 거쳐 텐서(tensor)를 반환해 줍니다.
# 메소드를 테스트 하기 위해서, 이미지 파일을 바이트 모드로 읽어주세요. 
# (먼저 `../_static/img/sample_file.jpeg`를 실제 여러분의 컴퓨터 파일 경로로 바꿔주세요)
# 그러면 텐서를 보실 수 있을 겁니다:

with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)

######################################################################
# 예측
# ~~~~~~~~~~~~~~~~~~~
#
# 지금부터 사전학습된 DenseNet 121 모델을 이미지 클래스를 예측하는데 사용해보겠습니다.
# ``torchvision`` 라이브러리에서 하나를 사용해서 모델을 불러오고 추론을 해보겠습니다. 
# 이 예시에서는 사전학습된 모델을 사용하겠지만, 여러분도 여러분의 모델에 같은 방식으로 적용해보실 수 있습니다.
# 모델 불러오기에 대해 더 알아보고 싶으신 분은 이 :doc:`tutorial </beginner/saving_loading_models>` 를 참조하세요.
 
from torchvision import models

# `weights`는 사전학습된 가중치를 사용하기 위해 `IMAGENET1K_V1`으로 설정합니다: 
model = models.densenet121(weights='IMAGENET1K_V1')
# 우리는 모델을 오직 추론을 위해 사용하기 떄문에, `eval` 모드로 바꿔줍니다:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


######################################################################
# ``y_hat`` 텐서는 예측된 클래스 id의 인덱스를 포함합니다.
# 하지만, 우리는 인간이 읽을 수 있는 클래스 name을 필요로 합니다. 
# 클래스 id를 name에 매핑하기 위해, `이 파일 <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>`_ 을 ``imagenet_class_index.json``로 다운로드 하십시오.
# 그리고 어디에 저장해두었는지 기억해두십시오. (만약 여러분이 이 튜토리얼을 정확하게 따라오셨다면, `tutorials/_static`에 저장하십시오)
# 이 파일을 ImageNet 클래스의 id와 name 매핑 데이터가 포함되어있습니다. 
# 이 JSON 파일을 사용해서 예측된 인덱스의 클래스 name을 도출해보겠습니다.

import json

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


######################################################################
# ``imagenet_class_index`` 딕셔너리를 사용하기 전에, 먼저 텐서 값을 문자열 값으로 바꿔주겠습니다. 
# 왜냐하면 ``imagenet_class_index``의 키값은 문자열이기 때문입니다. 
# 위에 있는 메소드를 테스트해보겠습니다:

with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

######################################################################
# 실행하면 아래처럼 결과가 나와야 합니다:

['n02124075', 'Egyptian_cat']

######################################################################
# 
# 배열의 첫번째 요소는 ImageNet 클래스의 id이고, 두 번째 요소는 사람이 읽을 수 있는 name입니다.
#
# .. 주목 ::
#    ``model`` 변수가 ``get_prediction`` 메서드의 일부가 아님을 알고 계신가요? 또는 모델이 전역 변수인 이유는 무엇일까요?
#    모델 읽어들이기는 메모리 및 컴퓨팅 측면에서 비용이 많이 드는 작업일 수 있습니다.
#    ``get_prediction`` 메서드에서 모델을 로드하면 메서드가 호출될 때마다 불필요하게 로드됩니다.
#    웹 서버를 구축하고 있으므로 초당 수천 건의 요청이 있을 수 있으므로 모든 추론에 대해 모델을 중복 로드하는 데 시간을 낭비해서는 안 됩니다.
#    따라서 메모리에 로드된 모델을 한 번만 유지합니다.
#    프로덕션 시스템에서는 규모에 맞게 요청을 처리할 수 있도록 컴퓨팅을 효율적으로 사용해야 하므로 일반적으로 요청을 처리하기 전에 모델을 로드해야 합니다.

######################################################################
# API 서버에 모델을 통합하기
# ---------------------------------------
#
# 이 마지막 파트에서는 플라스크 API에 모델을 추가해보겠습니다. 
# API 서버는 이미지 파일을 가져와야 하므로 ``predict`` 메서드를 업데이트하여 요청에서 파일을 읽습니다:
#
# .. code-block:: python
#
#    from flask import request
#
#    @app.route('/predict', methods=['POST'])
#    def predict():
#        if request.method == 'POST':
#            # we will get the file from the request
#            file = request.files['file']
#            # convert that to bytes
#            img_bytes = file.read()
#            class_id, class_name = get_prediction(image_bytes=img_bytes)
#            return jsonify({'class_id': class_id, 'class_name': class_name})

######################################################################
# 이제 ``app.py`` 파일이 완성되었습니다. 다음은 정식 버전입니다. 경로를 파일을 저장한 경로로 바꾸면 다음이 실행됩니다.
# .. code-block:: python
#
#   import io
#   import json
#
#   from torchvision import models
#   import torchvision.transforms as transforms
#   from PIL import Image
#   from flask import Flask, jsonify, request
#
#
#   app = Flask(__name__)
#   imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
#   model = models.densenet121(weights='IMAGENET1K_V1')
#   model.eval()
#
#
#   def transform_image(image_bytes):
#       my_transforms = transforms.Compose([transforms.Resize(255),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(
#                                               [0.485, 0.456, 0.406],
#                                               [0.229, 0.224, 0.225])])
#       image = Image.open(io.BytesIO(image_bytes))
#       return my_transforms(image).unsqueeze(0)
#
#
#   def get_prediction(image_bytes):
#       tensor = transform_image(image_bytes=image_bytes)
#       outputs = model.forward(tensor)
#       _, y_hat = outputs.max(1)
#       predicted_idx = str(y_hat.item())
#       return imagenet_class_index[predicted_idx]
#
#
#   @app.route('/predict', methods=['POST'])
#   def predict():
#       if request.method == 'POST':
#           file = request.files['file']
#           img_bytes = file.read()
#           class_id, class_name = get_prediction(image_bytes=img_bytes)
#           return jsonify({'class_id': class_id, 'class_name': class_name})
#
#
#   if __name__ == '__main__':
#       app.run()

######################################################################
# 웹 서버를 테스트해 봅시다! 실행해보세요:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

#######################################################################
# `requests <https://pypi.org/project/requests/>`_ 라이브러리를 사용해서 POST 요청을 보낼 수 있습니다.
#
# .. code-block:: python
#
#    import requests
#
#    resp = requests.post("http://localhost:5000/predict",
#                         files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})

#######################################################################
# `resp.json()`은 다음과 같이 표시됩니다.
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#

######################################################################
# 다음 단계
# --------------
#
# 
# 우리가 작성한 서버는 매우 초보적인 단계이기 때문에 프로덕션 애플리케이션에 필요한 모든 작업을 수행하지 못할 수 있습니다.
# 그래서 더 좋게 만들기 위해 할 수 있는 몇 가지 일들이 있습니다:
#
#
# - 엔드포인트 ``/predict``는 요청에 이미지 파일이 항상 있다고 가정합니다.
#   이는 모든 요청에 적용되지 않을 수 있습니다.
#   사용자는 다른 매개변수로 이미지를 보내거나 이미지를 전혀 보내지 않을 수 있습니다.
# 
# - 이미지 형식이 아닌 파일도 보낼 수 있습니다. 하지만 오류를 처리하지 않기 때문에 서버가 중단됩니다.
#   예외를 발생시키는 명시적인 오류 처리 경로를 추가하면 잘못된 입력을 더 잘 처리할 수 있습니다.
#
# - 모델이 많은 수의 이미지 클래스를 인식할 수 있지만 모든 이미지를 인식하지 못할 수 있습니다.
#   모델이 이미지에서 아무것도 인식하지 못하는 경우를 처리하도록 구현을 개선합니다.
#
# - 프로덕션 환경에 배포하기에 적합하지 않은 개발 모드에서 플라스크 서버를 실행합니다.
#   프로덕션에서 플라스크 서버를 배포하는 방법은 `이 튜토리얼 <https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/>`_을 확인하세요.
#
# - 이미지를 가져와 예측을 표시하는 양식으로 페이지를 만들어 UI를 추가할 수도 있습니다.
#   유사한 프로젝트의 `데모 <https://pytorch-imagenet.herokuapp.com/>`_와 `소스 코드 <https://github.com/avinassh/pytorch-flask-api-heroku>`_를 확인하세요.
# 
# - 이 튜토리얼에서는 한 번에 하나의 이미지에 대한 예측을 반환할 수 있는 서비스를 빌드하는 방법만 보여주었습니다.
#   한 번에 여러 이미지에 대한 예측을 반환할 수 있도록 서비스를 수정할 수 있습니다. 
#   또한 `service-streamer <https://github.com/ShannonAI/service-streamer>`_ 라이브러리는 서비스에 대한 요청을 자동으로 대기열에 넣고 모델에 공급할 수 있는 미니 배치로 샘플링합니다.
#   `이 튜토리얼 <https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer>`_을 확인할 수 있습니다.
# 
# - 마지막으로 페이지 상단에 연결된 PyTorch 모델 배포에 대한 다른 자습서를 확인하는 것이 좋습니다.
