"""적외선 센서 감지 확인
	if(적외선 센서 감지)

		모터11(DC모터) 0.8바퀴 역방향 회전
		모터10(DC모터) 역방향(반시계) 회전 / 1초
		고정장치(왼쪽) 풀기
		모터10(DC모터) 역방향(반시계) 회전 / 스위치 센서5이 접촉 시 정지
		Pump 키기

		모터10(DC모터) 순방향(시계) 회전 / 0.5초
		고정장치(왼쪽) 잡기
		모터10(DC모터) 순방향(시계) 회전 / 스위치 센서6이 접촉 시 정지
		카메라로 사진 찍기

		고정장치(오른쪽) 풀기
		Pump 끄기
		모터11(DC모터) 0.8바퀴 순방향 회전
		delay 0.2초
		고정장치(오른쪽) 잡기
	if문 끝

	적외선 센서 감지 안됨
		모터12(DC모터) 순방향 회전 / 스위치 센서11이 접촉시 정지 (동작 시간 측정)
		모터12(DC모터) 역방향 회전 / 바로 윗줄에서 측정한 시간만큼 동작
"""




import time

import RPi.GPIO as GPIO
from gpiozero import Motor

import busio                            # I2C 통신을 설정하기 위한 라이브러리

from board import SCL, SDA              # I2C 통신에서 사용되는 SCL(클럭)과 SDA(데이터) 핀.
from adafruit_pca9685 import PCA9685    # PWM 드라이버로, 모터의 속도를 제어하는 데 사용
from adafruit_motor import motor, servo # DC 모터 제어, 서보 모터 제어를 위한 라이브러리

#==============================================================================================

i2c = busio.I2C(SCL, SDA)           # I2C 통신을 설정. SCL과 SDA: I2C 통신을 위한 핀

pca = PCA9685(i2c, address=0x40)    # I2C를 통해 PCA9685 장치를 초기화. 0x40: PCA9685의 기본 I2C 주소
pca.frequency = 100                 # PCA9685의 PWM 주파수를 100Hz로 설정

                                    # 주의: 서보모터는 50Hz에서 동작하므로 해결 필요
                                    Warning # 위 주의를 보게 하기 위한 인위적인 에러 코드(추후 삭제)

#==============================================================================================

# <채널 수정 필요>

motor1 = motor.DCMotor(pca.channels[1], pca.channels[2])    # DC / 책 투입용 1
motor2 = motor.DCMotor(pca.channels[3], pca.channels[4])    # DC / 책 투입용 2

motor3 = motor.DCMotor(pca.channels[8], pca.channels[7])    # DC / 책 고정장치 리드스크류

motor4 = motor.DCMotor(pca.channels[10], pca.channels[9])   # 서보 / 고정장치1(아래)
motor5 = motor.DCMotor(pca.channels[1], pca.channels[2])    # 서보 / 고정장치1(위)

motor6 = motor.DCMotor(pca.channels[3], pca.channels[4])    # 서보 / 고정장치2(아래)
motor7 = motor.DCMotor(pca.channels[8], pca.channels[7])    # 서보 / 고정장치2(위)

motor8 = motor.DCMotor(pca.channels[10], pca.channels[9])   # 서보 / 고정장치3(아래)
motor9 = motor.DCMotor(pca.channels[1], pca.channels[2])    # 서보 / 고정장치3(위)

motor10 = motor.DCMotor(pca.channels[3], pca.channels[4])   # DC / 유체축 회전
motor11 = motor.DCMotor(pca.channels[8], pca.channels[7])   # DC / 유체축 길이조절 리드스크류
motor12 = motor.DCMotor(pca.channels[10], pca.channels[9])  # DC / 책 배출 리드 스크류

#==============================================================================================

GPIO.setmode(GPIO.BCM)  # GPIO 설정

ultrasonic_pins = {
    "sensor1": {"trig": 17, "echo": 18},  # 초음파 센서1
    "sensor2": {"trig": 22, "echo": 23},  # 초음파 센서2
    "sensor8": {"trig": 24, "echo": 25}   # 초음파 센서8
}

ir_sensor_pin = 27  # 적외선 센서 (sensor3)

switch_sensors = {
    "sensor5": 5,       # 유체축 회전 1(왼쪽)
    "sensor6": 6,       # 유체축 회전 2(오른쪽)
    "sensor7": 13,      # 유체축 길이
    "sensor8": 5,       # 고정장치 1
    "sensor9": 6,       # 고정장치 2
    "sensor10": 13,     # 고정장치 3
    "sensor11": 5,      # 책 배출 리드스크류
}

GPIO.setup(ir_sensor_pin, GPIO.IN)

for key, pin in switch_sensors.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull-up 저항 사용
    
#==============================================================================================

def get_distance(trig_pin, echo_pin):   # 초음파 센서 거리 계산 함수
    
    GPIO.output(trig_pin, True)
    
    time.sleep(0.00001)                 # 아주 짧은 시간 동안 초음파 신호 보냄(10 마이크로초: 초음파 펄스를 보내기 위한 최소 펄스 길이)
    
    GPIO.output(trig_pin, False)

    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start    
    distance = pulse_duration * 17150   # 거리 계산 (cm) 17150: 초음파가 공기 중에서 이동하는 속도(34300 cm/s)를 2로 나눈 값
    return round(distance, 2)

#==============================================================================================

def is_ir_detected(sensor_pin):                 # 적외선 센서 감지 함수
    return GPIO.input(sensor_pin) == 1

#==============================================================================================

def is_switch_pressed(sensor_pin):              # 스위치 센서 감지 함수
    return GPIO.input(sensor_pin) == GPIO.LOW   # 눌렸을 때 LOW 신호 발생

#==============================================================================================

def calculate_rotation_time(degrees, rpm):      # 모터가 돌아갈 시간을 계산
    return degrees / (rpm * 6)

# =============================================================================================

# 모터 제어 함수  
# motor : 모터 객체 (예: motor1)
# degrees : 모터가 돌아갈 각도 (모터2, 3에서는 360*바퀴수로 입력)
# rotation : 회전 방향 (True: 시계방향, False: 반시계방향)
# rpm : 모터의 RPM (분당 회전수)
# sensor_pin : 센서 핀 (None이면 센서 무시)

def motor_dc_control(motor, degrees, rotation, rpm, sensor_pin):
    rotation_time = calculate_rotation_time(degrees, rpm)
    
    # 회전 방향 설정
    if rotation:        
        motor.throttle = 0.8    # 모터 회전속도: 최대치의 0.8배(임의로 정함)
    else:
        motor.throttle = -0.8
    
    start_time = time.time()
    
    while time.time() - start_time < rotation_time:  # 예정된 이동 시간 동안
        if sensor_pin is not None and GPIO.input(sensor_pin):  # 센서가 있고, 입력값이 참인 경우
            motor.stop()                                      # 모터 정지 후 함수 종료
            return
        time.sleep(0.01)  # CPU 자원을 절약하기 위해 루프 주기 설정
    
    motor.stop()  # 회전 시간이 끝나면 모터를 정지
    return

# =============================================================================================

def lead_screw_motor(motor, switch_pin):
    motor.throttle = 1  # 모터를 정방향으로 회전 시작
    start_time = time.time()  # 회전 시작 시점 시간 기록

    # 스위치 센서가 접촉될 때까지 모터 회전
    while True:
        if GPIO.input(switch_pin) == GPIO.LOW:  # 스위치가 눌렸을 때 (접촉된 상태)
            motor.throttle = 0  # 모터 정지
            break
        time.sleep(0.01)  # 루프가 너무 빠르게 실행되지 않도록 대기

    # 모터가 정지한 시점에서의 시간을 기록
    elapsed_time = time.time() - start_time

    # 역방향으로 회전 시작
    motor.throttle = -1
    time.sleep(elapsed_time)  # 정방향으로 회전한 시간만큼 역방향으로 회전

    motor.throttle = None  # 모터 정지
    return

# =============================================================================================

def initialize_servo(pca, channel):
    """
    서보 모터를 초기화하는 함수
    :param pca: PCA9685 객체
    :param channel: 서보 모터가 연결된 PCA9685 채널
    :return: 초기화된 서보 모터 객체
    """
    servo_motor = servo.Servo(pca.channels[channel], min_pulse=500, max_pulse=2500)
    return servo_motor

# =============================================================================================

def set_servo_angle(servo_motor, angle):
    """
    서보 모터의 각도를 설정하는 함수
    :param servo_motor: 제어할 서보 모터 객체
    :param angle: 설정할 각도 (0도 ~ 180도)
    """
    if 0 <= angle <= 180:
        servo_motor.angle = angle

# =============================================================================================

def left_servo_grab(switch_pin1, switch_pin2):
    """
    두 개의 서보 모터를 제어하고, 스위치가 눌리면 동작을 멈추는 함수
    :param servo1: 첫 번째 서보 모터 객체
    :param servo2: 두 번째 서보 모터 객체
    :param switch_pin: 스위치 센서의 GPIO 핀 번호
    """
    
    servo1_1 = motor4
    servo1_2 = motor5
    
    servo2_1 = motor6
    servo2_2 = motor7
    
    # 각 단계별 서보 모터 각도 설정 (0) ~ (5)
    servo_positions = [
        (11.83, 21.57),
        (25.37, 31.49),
        (31.44, 41.67),
        (33.58, 52.22),
        (33.37, 63.28),
        (31.52, 75.05)  # 아때 무조건 닿는다고 함.
    ]
    
    # GPIO 핀 설정
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(switch_pin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(switch_pin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    try:
        for pos1, pos2 in servo_positions:
            # 서보 모터 각도 설정
            set_servo_angle(servo1_1, pos1)
            set_servo_angle(servo2_1, pos1)
            
            set_servo_angle(servo1_2, pos2)
            set_servo_angle(servo2_2, pos2)

            # 스위치가 눌렸는지 확인
            if (GPIO.input(switch_pin1) == GPIO.LOW)or(GPIO.input(switch_pin2) == GPIO.LOW):  # 스위치가 눌린 경우
                break  # 동작을 멈추고 함수 종료

            time.sleep(0.05)  # 다음 단계로 넘어가기 전에 잠시 대기 (0.05초, 변경 가능)
            
    finally:
        GPIO.cleanup()  # GPIO 설정 정리

# =============================================================================================

def right_servo_grab(switch_pin3):
    """
    두 개의 서보 모터를 제어하고, 스위치가 눌리면 동작을 멈추는 함수
    :param servo1: 첫 번째 서보 모터 객체
    :param servo2: 두 번째 서보 모터 객체
    :param switch_pin: 스위치 센서의 GPIO 핀 번호
    """
    
    servo3_1 = motor8
    servo3_2 = motor9
    
    # 각 단계별 서보 모터 각도 설정 (0) ~ (5)
    servo_positions = [
        (11.83, 21.57),
        (25.37, 31.49),
        (31.44, 41.67),
        (33.58, 52.22),
        (33.37, 63.28),
        (31.52, 75.05)  # 아때 무조건 닿는다고 함.
    ]
    
    # GPIO 핀 설정
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(switch_pin3, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    try:
        for pos1, pos2 in servo_positions:
            # 서보 모터 각도 설정
            set_servo_angle(servo3_1, pos1)
            set_servo_angle(servo3_2, pos2)

            # 스위치가 눌렸는지 확인
            if GPIO.input(switch_pin3) == GPIO.LOW:  # 스위치가 눌린 경우
                break  # 동작을 멈추고 함수 종료

            time.sleep(0.05)  # 다음 단계로 넘어가기 전에 잠시 대기 (0.05초, 변경 가능)
            
    finally:
        GPIO.cleanup()  # GPIO 설정 정리

# =============================================================================================

def left_servo_release():
    
    servo1_1 = motor4
    servo1_2 = motor5
    
    servo2_1 = motor6
    servo2_2 = motor7

    try:
        # 서보 모터 각도 설정
        set_servo_angle(servo1_1, 90)
        set_servo_angle(servo2_1, 180)
        
        set_servo_angle(servo1_2, 180)
        set_servo_angle(servo2_2, 180)
            
    finally:
        GPIO.cleanup()  # GPIO 설정 정리

# =============================================================================================

def right_servo_grab(switch_pin3):

    servo3_1 = motor8
    servo3_2 = motor9
    
    try:
        # 서보 모터 각도 설정
        set_servo_angle(servo3_1, 180)
        set_servo_angle(servo3_2, 180)
            
    finally:
        GPIO.cleanup()  # GPIO 설정 정리

# =============================================================================================


try:
    # 메인 제어 로직
    
    
finally:
    GPIO.cleanup()
    pca.deinit()
