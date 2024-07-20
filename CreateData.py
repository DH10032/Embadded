'''
데이터 생성 코드


output 형식      label list

코드 실행전 주의
사진 경로 설정하시고

터미널에서 아래 명령어를 통해 모듈 설치
pip3 install tk

설치 후에도 에러가 있다면
pip install --upgrade pillow
해주세요

'''
# ========================================================= # import

from pathlib import Path
from tkinter import *
from PIL import Image, ImageTk                               # Image.open()을 위해 부름
from glob import *
import os
import json

# ========================================================= # class
'''
tkinter 라는 gui 라이브러리를 이용했습니다.
오늘 처음 써봐서 좀 익숙하지 않네요
'''
class display:
    def __init__(self, root, ori_img_paths, grid_img_paths):

        self.root = root
        self.ori_img_paths = ori_img_paths
        self.grid_img_paths = grid_img_paths
        self.status_list = ['낙서', '얼룩', '찢어짐', '꾸겨짐', '변색']

        self.canvas = Canvas(New_window, width = Canvas_width, height = Canvas_height, bg='white')  # canvas 설정
        self.canvas.pack()

        self.ori_index = 0
        self.grid_index = 0

        self.i = 0
        self.x = 0
        self.y = 0
        
        # 사각형 객체 생성
        self.rectangle = self.canvas.create_rectangle(0, 0, 0, 0, outline = "black", width=2)

        # 버튼 객체 리스트 생성
        self.button_list = [Button(self.root, text = status, command = lambda idx=i: self.status_update(idx)) for i, status in enumerate(self.status_list)]

        # 버튼 화면에 넣기
        for button in self.button_list:
            button.pack(side = "right")
        
        self.result = []    # 여기에 라벨을 리스트로 저장

        self.print_image()

    def print_image(self):  # 이미지를 print하는 함수

        if self.grid_index < len(self.grid_img_paths) and self.ori_index < len(self.ori_img_paths):

            # 원본 이미지 canvas에 출력
            pil_img = Image.open(self.ori_img_paths[self.ori_index])
            self.ori_photo = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.ori_photo)

            self.canvas.tag_raise(self.rectangle)                       # 레이어 우선 순위를 조절해 사각형이 이미지 위에 오도록 조절
            self.canvas.coords(self.rectangle, self.x, self.y, self.x + grid, self.y + grid)

            # rextangle 출력 후 다음 사각형 x, y 계산
            if self.x + grid < img.size[0]:
                self.x += grid
            else:
                self.x = 0
                self.y += grid

            # grid 이미지 canvas에 출력
            pil_img = Image.open(self.grid_img_paths[self.grid_index])
            self.grid_photo = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(Canvas_width/2, 0, anchor="nw", image=self.grid_photo)
            self.grid_index += 1

            # 원본 이미지 인덱스 조절
            if (img.size[0]/grid + 1) * (img.size[1]/grid + 1) < self.grid_index:
                self.ori_index += 1
                self.y = 0

        else:
            print('완료')

    def status_update(self, idx):
        self.result.append(idx)
        self.print_image()

# ========================================================= # 초기 설정

path = '/home/ldh/Desktop'
grid = 30   # 단위 픽셀
output_folder_name = '/grid_img'
Canvas_width = 1500
Canvas_height = 1000

# ========================================================= # 이미지 자르기 code

Early_path = Path(path+'/img')                         # 초기 경로
load_paths = sorted(list(Early_path.glob('*')))         # 이미지들 경로 list
imgs = [Image.open(img_path) for img_path in load_paths]

# output_folder_name의 폴더 생성 (이미 같은 이름이 존재하면 건너뜀 -> exist_ok=True)
os.makedirs(path + output_folder_name, exist_ok=True)


'''
grid 단위로 자른 이미지 save하는 코드
grid_img 디렉토리에 저장됨
'''

for num, img in enumerate(imgs):
    grid_num = 0
    for y in range(0, img.size[1], grid):           # img.size[1] 이미지 세로
        for x in range(0, img.size[0], grid):       # img.size[0] 이미지 가로
            img_cropped = img.crop((x, y, x+grid, y+grid))
            img_cropped.save(path + '/grid_img/' + str(num+grid_num) + '.png', 'png')
            grid_num += 1

# ========================================================= # 이미지 출력 코드

# gui에서 불러올 grid 이미지 경로 설정
grid_img_paths = Path(path+'/grid_img')                     # 초기 경로
grid_img_paths = sorted(list(grid_img_paths.glob('*')))         # 이미지들 경로 list



New_window = Tk()                           # 창 생성

Dis = display(New_window, load_paths, grid_img_paths)

New_window.mainloop()

with open('data.json', 'w') as f:
    json.dump(Dis.result, f)

print("데이터가 JSON 파일로 저장되었습니다.")
