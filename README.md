# python_practice

1. 파이썬 설치 
- https://www.python.org/downloads/release/python-353/
- 3.6.4 설치
- Add PATH 체크하고 Install Now
- `python -V` : python 버전 확인, 3.6.4 면 됨

2. 아나콘다 설치
- Python Data Science Platform : package를 이미 포함하고 있는 플랫폼
- https://www.anaconda.com/download/#windows
- 파이썬 3.6 버전,윈도우 64비트
- 셋업.exe 관리자 권한으로 실행 - default 로 설치
- Anaconda prompt를 관리자 권한으로 실행

3. Tensorflow 설치
- Anaconda prompt
  - pip upgrade : `python -m pip install --upgrade pip`
  - Conda 환경 만들기 : `conda create -n tensorflow python=3.5`
  - Coda 환경 확인 하기 : `conda info --envs`
  - 활성화 : `activate tensorflow` (안되면 앞에 `source` 붙이기)
    - 비활성화 : `deactivate tensorflow`
  - @Prompt가 `tensorflow`로 바뀜
  - `pip install tensorflow`
- 설치 확인 코드
```python
import tensorflow as th
hello = tf.constant('hello')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
```

4. Jupyter 설치하기
   - `pip install jupyter` -> `jupyter notebook` -> localhost:8888 접속하면 됨

