## 팀
1. 메카닉스응용실험 끝나고 시간 할애
## 팀 규칙
1. 맡은 스케줄 밀리지 않게 하기 
2. 시간약속 준수하기
## 정보
<details>
  <summary>소프트웨어</summary>
  1. 맵핑 - 이동훈, 최성현 </br>
  2. TTS & STT - 김지호 </br>
  3. 라인트레이싱 - 이동훈, 최성현 </br>
  4. 상황인지 </br>
  5. hw 설계 및 제작 - 윤석현(설계), 류정현(전자회로) </br>
  -------------------------------------------------- </br>
  a) 로봇이 물건 찾는 작동 방식 </br>
    1. 사진을 서버에 올림 </br>
    2. 서버가 이미지를 뿌림 </br>
    3. 이미지에 맞는 물건을 찾는게 목표 - SLAM </br>
    4. 로봇이 지나가는 구간에 찾고자 하는 물건이 없으면, 지도에 표시 - SLAM </br>
    5. 의심되는 물건은 사진을 찍어서 서버에 보내고 위치를 표시 - 카메라 인식 </br>
    6. 로봇이 찾아볼수 없는 부분은 지도에 알려주기 - 상자 같은 서랍 구분이 필요(딥러닝) </br>
  b) 주차/월별 계획 </br>
    3월 5주차 ~ 4월 2주차 </br>
      이미지 처리/딥러닝•머신러닝 </br>
    4월 3주차 ~ 5월 1주차 </br>
      OpenCV: 이미지 처리, SLAM에 이용 </br>
      SLAM: 센서 신호 처리(움직임 추정, 장애물 회피), 그래프 최적화 </br>
      *특이 사항: 4월 말~5월 초에 대회 공지가 올라올 예정이므로 공지 체크할 것. </br>
    5월 2주차 ~ 5월 5주차 </br>
  --------------------------------------------------- </br>
  <img width="863" alt="image" src="https://github.com/DH10032/Teams/assets/155617166/7b859d7a-5345-4ada-b6a8-235f7b1e94e1">
 처리 순서는 (frondend - > backend) - > Map representation(맵 작성) </br>
 frondend를 먼저 끝낸 후 backend를 진행하면 어떨까 생각. (협의 후 결정 예정) </br>
 1) Data Acquistion, Visual odometry, Loop closure detection - > frondend </br>
  Data Acquisiton(데이터 획득): 카메라/라이다 같은 센서로부터 정보 획득(+ 데이터로부터 노이즈 제거 필요) </br>
  Visual odometry(시각적 주행 거리 측정): 데이터 특정 추출 -> 상대적 움직임 예측  </br>
  Loop closure detection(루프 폐쇄 검출): 방문한 위치인지 판단. </br>
2) Backend optimization(최적화) - > backend </br>
 
</details>

<details>
  <summary>하드웨어</summary>
  3/28일까지</br>
  - 윤석현: 소형 로봇 몸체 구현(전체적으로) </br>
  - 류정현: 구동 회로 구현 + 배터리 </br>
  4/4까지</br>
  - 윤석현: 제작시작</br>
  - 류정현: 제작시작 </br>
  4/11까지</br>
  - 윤석현: 기본형완성 </br>
  - 류정현: 기본형완성 </br>
  시험기간</br>
  5월초</br>
  - 윤석현: 추가적인 제작? </br>
  - 류정현: 추가적인 제작? </br>
  5월 말 </br>
  - 윤석현: 모든 제작품의 제작 완  </br>
  - 류정현: 모든 제작품의 제작 완 </br>

  
  https://www.youtube.com/@GDSB/playlists
</details>
  
<details>
  <summary>회로 및 라즈베리</summary>
  우분투 20.04로 설치 완
  
  모터는 dc모터+드라이버 (배달중) </br>
  배터리는 9v짜리 건전지 </br>
  카메라는 미정 </br>

  
</details>

<details>
  <summary>STT</summary>
  API 사용 시
  - Whisper API 사용해야 할 가능성 높음</br>
  - TTS 방식에는 gTTS, 네이버 클로바 등이 있음.</br>
  - 클로바-요금이 매우 비쌈</br>
  - Whisper API는 가격이 무료라는 정보와 토큰별로 받는 요금이라는 정보가 둘 다 있어 추후 찾아볼 예정.</br>
  - 단점
    - 매우 높은 확률로 네트워크 연결 필요</br>
  API 미사용 시
  - STT를 직접 구현하기에는 몇 백 시간의 데이터를 학습해야 하며, 부정확한 결과가 나올 수 있음.
  - 푸리에 변환, 분포 등 매우 어려운 이론적 지식이 필요
  - 관련 정보 찾기가 어려움
  
</details>

<details>
  <summary>공간인식</summary>
  
  1. 아두이노 실내 위치추적 모듈(오차 10cm내외) - DWM1000 모듈
  
  2. [SLAM 방식](https://hjdevelop.tistory.com/15/)
     
  SLAM 방식
    <details>
      <summary>프런트 엔드</summary>
    </details>
  
  <details>
      <summary>백 엔드</summary>
    </details>
  
</details>

## 초등학교 안전/방범 지킴이
<details>
  1. 이미지 분석 </br>
  2. 

  [최성현]
  1. 학교 관계자(학생, 선생님 등) 얼굴 인식 기능
      - 활용 분야 : 
</details>
