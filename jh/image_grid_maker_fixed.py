import os
import cv2
import numpy as np

# 기본 경로 설정
base_path = '/home/jijijiho/문서/openbook/'
input_folder = os.path.join(base_path, 'grid_before')
output_folder = os.path.join(base_path, 'grid_after') 

# 입력 폴더가 존재하는지 확인
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder does not exist: {input_folder}") # output_folder가 없으면 폴더 생성, 있으면 넘어감

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 가로, 세로 분할 개수 설정
num_horizontal_splits = 30  # 예시 숫자로, 추후 수정
num_vertical_splits = 50    # 예시 숫자로, 추후 수정

# 숫자 값과 문자열 값을 매핑하는 사전
label_mapping = {
    '0': 'background',
    '1': 'text',
    '2': 'highlighter',
    '3': 'ripped',
    '4': 'stain',
    '5': 'crumpled'
}

# 이미지 분할 및 텍스트 파일 저장 함수
def split_image_and_save(image_path, output_folder):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")    # 이미지가 없거나 읽기 실패시 에러
        return
    
    image_name = os.path.basename(image_path).split('.')[0]  # 이미지 파일명에서 확장자 제거

    # 이미지 크기 얻기
    height, width = image.shape[:2]

    # 각 분할된 이미지의 크기 계산
    tile_width = width // num_horizontal_splits
    tile_height = height // num_vertical_splits

    total_tiles = num_horizontal_splits * num_vertical_splits   # 추후 processing tile n/total_tiles를 위함
    tile_count = 0

    for i in range(num_vertical_splits):
        for j in range(num_horizontal_splits):
            tile_count += 1
            
            # 분할된 이미지의 좌표 계산
            x_start = j * tile_width
            y_start = i * tile_height
            x_end = (j + 1) * tile_width
            y_end = (i + 1) * tile_height
            # 오른쪽 위 꼭짓점부터 왼쪽 아래 꼭짓점까지
            
            # 이미지 분할
            tile = image[y_start:y_end, x_start:x_end]
            
            # 분할된 이미지 저장 경로
            tile_filename = f"{image_name}_tile_{i}_{j}.jpg"
            tile_path = os.path.join(output_folder, tile_filename)
            cv2.imwrite(tile_path, tile)

            # 분할된 이미지 화면에 출력
            cv2.imshow('Tile', tile)
            print(f"Processing tile {tile_count}/{total_tiles}")

            # 키보드 입력 받기
            while True:
                # 0:배경, 1:텍스트, 2:형광펜, 3:찢어짐. 4:얼룩, 5:구겨짐
                print("Enter a value (0: background, 1: text, 2: highlighter, 3: ripped, 4: stain, 5: crumpled) for the displayed image:")
                key = cv2.waitKey(0)
                if key == 27:  # ESC 키를 누르면 종료
                    print("Process terminated by user.")
                    cv2.destroyAllWindows()
                    return
                
                # 사용자가 누른 값이 '0', '1', '2', '3', '4', '5' 중 하나인지 확인
                elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:   # ord(): 주어진 문자의 ASCII 코드 값 반환
                    value = chr(key)
                    label = label_mapping[value]
                    break
                
                else:
                    print("Invalid input. Please enter a value between 0 and 5.")
            
            # txt 파일로 값 저장
            txt_filename = f"{image_name}_tile_{i}_{j}.txt"
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, 'w') as f:
                f.write(label)
            
            # 현재 이미지 창을 닫기
            cv2.destroyWindow('Tile')

# 폴더 내의 모든 이미지 파일에 대해 처리
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_folder, filename)
        split_image_and_save(image_path, output_folder)

# 모든 작업 완료 후 모든 창 닫기
cv2.destroyAllWindows()
