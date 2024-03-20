## 팀
1. 수요일 4시~6시30분 (병합 및 방향성 논의)
2. 필요하다면 메카닉스응용실험 끝나고 시간 할애 (필요하다면)
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
</details>

<details>
  <summary>하드웨어</summary>
  1. 무한궤도 조사 (구할 수 있는 곳, 판매가격이나 수치등등, 괜찮은 여러 종류를 표로 만들기?)  </br>
  2. 디자인 구상 (대략적인 큰 디자인, 방향성의 시각화느낌)  </br>
  3. 대략적인 물리적 수치(중량, 크기등등) 디자인 구상에 기초하여 예측하기 (이건 목요일에 만날때 회의해보기) </br>
  https://www.youtube.com/@GDSB/playlists
</details>
  
<details>
  <summary>회로 및 라즈베리</summary>

    
  라즈베리에서 ros 설치가 되야하는 데, 잘 안됨,apt update 할때와 직접 다운받을때 릴리즈 페키지를 받아야하는 데 릴리즈가 되지않았다고 나옴, 아마 키 사이트의 문제같음, 제대로 입력하면 접근이 안되고, 돌려서 입력하면 접근은 되는 데 릴리즈 된게 없다고 나옴
    우선은 우분투 설치 + 그 곳에 ros 설치 느낌으로 진행할듯?  근데 그럴려면 sd카드가 필요한 거 같은데 못 찾겠어서 일단은 목요일전까지 ros 직접 설치해보고 안되면 목욜에 남은 sd카드 빌려가서 우분투 깔아볼깨용 -> 만약 성공하면 우분투 걸칠 필요가 없어서 중간과정이 쉬워질것 같음, 그래도 안되면 되는거 해야지
    </br>

  https://foni.tistory.com/85 <- 여기와 동일한 문제 발생
   1. ROS Repository에서 gpg 설정문제 -> apt-key가 아닌 gpg를 다운받겠다고 하면 받아짐 ( 이것이 문제 해결이 된 것인지는 확신이 들지 않음)
   2. apt-get에서 6개중 무작위로 한게씩 ign, err이 발생하는 문제 (할 때마다 무작위로 발생, 1개는 발생함)
   3. 문제가 되는 부분, 프로그램 다운시 설치가 안됨, 패키지의 위치를 찾을 수 없다고 함

  여기 밑은 라즈베리에서 ros로 바로 설치할 때 필요한 사이트들(작동 안됐슴) </br>
    https://kyubot.tistory.com/90 

http://wiki.ros.org/ROSberryPi/Installing%20ROS%20Kinetic%20on%20the%20Raspberry%20Pi#Setup_ROS_Repositories </br>

위에서 말한 gpg 해결법
https://unix.stackexchange.com/questions/399027/gpg-keyserver-receive-failed-server-indicated-a-failure <br>

  sudo gpg --keyserver keyserver.ubuntu.com --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 </br>
    


  
</details>

<details>
  <summary>공간인식</summary>
  1. 아두이노 실내 위치추적 모듈(오차 10cm내외) </br>
  2. 
</detais>

## 초등학교 안전/방범 지킴이
<details>
  1. 이미지 분석 </br>
  2. 

  [최성현]
  1. 학교 관계자(학생, 선생님 등) 얼굴 인식 기능
      - 활용 분야 : 
</details>
