## 팀
1. 메카닉스응용실험 끝나고 시간 할애
## 팀 규칙
1. 맡은 스케줄 밀리지 않게 하기 
2. 시간약속 준수하기
## 계획

## 정보
<details>
  <summary>소프트웨어</summary>
  1. 맵핑 - 이동훈, 최성현 </br>
  2. TTS & STT - 김지호 </br>
  3. 라인트레이싱 - 이동훈, 최성현 </br>
  4. 상황인지 </br>
  5. hw 설계 및 제작 - 윤석현(설계), 류정현(전자회로) </br>
  ------------------------------------------------------------------------------ </br>
  a) 로봇이 물건 찾는 작동 방식 </br>
   1. 사진을 서버에 올림 </br>
   2. 서버가 이미지를 뿌림 </br>
   3. 이미지에 맞는 물건을 찾는게 목표 - SLAM </br>
   4. 로봇이 지나가는 구간에 찾고자 하는 물건이 없으면, 지도에 표시 - SLAM </br>
   5. 의심되는 물건은 사진을 찍어서 서버에 보내고 위치를 표시 - 카메라 인식 </br>
   6. 로봇이 찾아볼수 없는 부분은 지도에 알려주기 - 상자 같은 서랍 구분이 필요(딥러닝) </br>
  b) 주차/월별 계획 </br>
   3월 5주차 ~ 4월 2주차 </br>
    SLAM 실습 세팅 & 수학적 이론(2명): 김지호, 노태윤 </br>
    딥러닝•머신러닝(1명): 이동 </br>
   4월 3주차 ~ 5월 1주차 </br>
    OpenCV: 이미지 처리(카메라) -> Visual SLAM에 이용 </br>
    SLAM: 센서 신호 처리(움직임 추정, 장애물 회피), 그래프 최적화 </br>
    *특이 사항: 4월 말~5월 초에 대회 공지가 올라올 예정이므로 공지 체크할 것. </br>
   5월 2주차 ~ 5월 5주차 </br>
  ------------------------------------------------------------------------------ </br>
  <img width="863" alt="image" src="https://github.com/DH10032/Teams/assets/155617166/7b859d7a-5345-4ada-b6a8-235f7b1e94e1"> </br>
  처리 순서는 (frontend - > backend) - > Map representation (맵 작성) </br>
  frontend를 먼저 끝낸 후 backend를 진행하면 어떨까 생각. (협의 후 결정 예정) </br>

 1) Data Acquistion, Visual odometry, Loop closure detection - > frontend </br>
 Data Acquisiton (데이터 획득): 카메라/라이다 같은 센서로부터 정보 획득 (+데이터로부터 노이즈 제거 필요) </br>
 Visual odometry (시각적 주행 거리 측정): 데이터 특정 추출 -> 상대적 움직임 예측 </br>
 Loop closure detection (루프 폐쇄 검출): 방문한 위치인지 판단. </br>
2) Backend optimization(최적화) - > backend </br>
------------------------------------------------------------------------------------- </br>
<필요한 tool 및 도서> </br>
1. 프로그래밍 언어 </br>
Python: Opencv 통해 C++ 보다 구현이 쉬움. But 쓰레드 관리, 최적화, HW 호환성 문제가 있음. 딥러닝 slam training 시에는 유용 </br>
C++: 빠르고 라이브러리가 많음.  </br>
Ros: tcp/ip 통신 보안상 문제로 현업에서는 안 씀. 그러나 쉬워서 학생들에게 추천. </br>
Ros2: Ros 보안 문제 개선, 현재는 개발 중. </br>
2. 필요한 수학적 이론 </br>
a) 선형대수학+베이즈 확률론(slam 기초☆) </br> 
 선형대수: 공간 이해 </br>
 베이즈 확률론: 상태 추정+ sensor fusion(센서 데이터 병합(merging)) </br>
b) 최소자승법 문제+최적화(최신 slam) </br>
3. 관련 라이브러리  </br>
Opencv, Eigen, Ceres, g2o, DBoW </br>
4. 최신 Slam </br>
Deep slam(Slam + deep learning) </br>
5. 추천 도서 </br>
OpenCV로 배우는 컴퓨터 비전과 머신 러닝, Computer Vision(고양이 표지, 이론 중심),Computer vision Algorithm and Application(저자:  Richard), An invitation to 3-D vision, Multiple View in geometry in
computer vision </br>


 
</details>

<details>
  <summary>하드웨어</summary>
  3/24일부터 구매할 파워보드/카메라 조사 및 구매, 각 수치 측정 후</br>
  3/25일부터 -> 모터드라이버 도착예정 -> 바로 회로연결(모터작동 시험) -> 후에 전체적인 배치 구상후  몸체설계 담당에게 정보 전달 </br>
  3/26일부터 -> 몸체 제작 시작</br>
  3/27 - 카메라 및 파워보드 연결 및 시험</br>
  3/30~3/31 몸체 제작 밑 결합 -> 이론상 여기까지 기본형 제작</br>
  4/1일까지</br>
  
  4/1부터</br>
  - 윤석현: 기본형 점검 / 관리</br>
  - 류정현: 라즈베리 구동 관련 코딩 </br>
  4/11까지</br>
  
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
  <summary>SLAM</summary>
  
  1. [SLAM 방식](https://hjdevelop.tistory.com/15/)
     
  SLAM 방식
    <details>
      <summary>프런트 엔드</summary>
    </details>
  
  <details>
      <summary>백 엔드</summary>
    </details>
  
</details>

## 문제점
<details>
  1. DC모터 회전수 측정(DC모터 대신 스텝모터로 변경) </br>
  2. 
  
</details>

## 부품
  
  1. [라즈베리 카메라](https://www.eleparts.co.kr/goods/view?no=12391455) </br>
  2. [스텝모터 드라이버](https://parts-parts.co.kr/product/pp-a710-nema17-3d-%ED%94%84%EB%A6%B0%ED%84%B0%EB%AA%A8%ED%84%B0-17hs4023-%EC%8A%A4%ED%85%9D%ED%95%91%EB%AA%A8%ED%84%B0/1143/category/155/display/1/) </br>
  3. [스텝모터](https://parts-parts.co.kr/product/pp-a710-nema17-3d-%ED%94%84%EB%A6%B0%ED%84%B0%EB%AA%A8%ED%84%B0-17hs4023-%EC%8A%A4%ED%85%9D%ED%95%91%EB%AA%A8%ED%84%B0/1143/category/155/display/1/) </br>
  4. [CM4108000](https://kr.element14.com/raspberry-pi/cm4108000/rpi-compute-module-4-lite-8gb/dp/3678911) </br>
  5. [라즈베리파이 PCIe 보드](https://ko.aliexpress.com/i/1005002923796998.html) </br>
  6. 그래픽카드 GTX 1660또는 RTX 3080 </br>
  7. LED
  8. Power
  9. 납땜
