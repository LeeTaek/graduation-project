# graduation-project ( 2017.09 - 2017.12 )
2017 graduation project : Tensorflow 라이브러리를 이용하여 구현한 작곡하는 인공지능 프로그램 


개발 환경 및 기술
언어 : Python3
사용 라이브러리 : Tensorflow



- 프로젝트 목적 
1인 미디어 시대에서 음악 사용의 저작권 문제를 해결하기 위해 저작권으로 부터 자유로운 음악을 작곡하는 인공지능 프로그램을 만든다. 


- 프로젝트 제안 동기 
사회에서 한번씩 터지는 표절 이슈를 접하다가 ‘AI가 만든 창작물의 저 권은 어떻게 될까?’ 라는 질문에서 시작되었다. 1인 미디어 시대에 누구나 쉽게 창작에만 집중할 수 있도록, 인공지능을 통해 저작권으로부터 자유로운 음악을 만들어 본다.
 

- 구현 : 
문자열로 악보를 표현하는 표기법인 ABC Notaion을 장르별로 DB로 만들어 RNN(LSTM) Cell에 학습시킨다. 음악은 시간의 흐름에 따른 코드와 음의 진행이기 때문에 시계열 처리에 적합한 RNN Cell 구조를 채택했다. 
해당 프로그램에서는 16세기 아일랜드 지역의 민요로부터 유래된 jigs 장르와 walts 장르의 데이터를 2000회 학습한 모델을 저장하여 실행한다. 초기 값으로 박자와 첫 코드를 설정해 준 후 학습된 데이터를 통해 장르별로 원하는 ABC Notation을 출력한다. 


- 한계 및 개선방향 :
1. 화성학과 같은 음악적 요소에 대한 것이 전혀 고려되지 않았다.
2. 학습 속도가 매우 느리다. 클라우드 서비스를 사용해 학습 속도 개선하는 방법을 생각해볼 것. 


- 배운 내용 : 
1. Python 언어
2. 기초적인 인공신경망 알고리즘 종류와 강점
