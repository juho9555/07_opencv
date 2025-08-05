# OpenCV

<details>
<summary>실습 1</summary>  
    
## 실습1

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 군집 개수 설정
k = 8

# 이미지 읽기
img = cv2.imread('../img/load_line.jpg')  # <- 이미지 파일 변경
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

# BGR → RGB (matplotlib 용)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# K-means 입력 데이터 준비
data = img.reshape((-1, 3)).astype(np.float32)

# K-means 파라미터 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
result = centers[labels.flatten()]
result_image = result.reshape(img.shape)

# 원본 + 결과 이미지 좌우로 연결
merged = np.hstack((img, result_image))

# === 고정 너비로 리사이즈 ===
target_width = 1000  # 고정된 너비
h, w = merged.shape[:2]
scale = target_width / w
resized = cv2.resize(merged, (target_width, int(h * scale)))

# OpenCV 창에 출력
cv2.imshow(f'Original (left) vs K-means (k={k}, right)', resized)

# 색상 비율 계산
_, counts = np.unique(labels, return_counts=True)
ratios = counts / counts.sum()
colors_rgb = [color[::-1]/255 for color in centers]

# matplotlib 시각화
plt.figure(figsize=(10, 4))

# 색상 팔레트
plt.subplot(1, 2, 1)
plt.title("Color Palette")
for i, color in enumerate(colors_rgb):
    plt.bar(i, 1, color=color)
plt.xticks([])
plt.yticks([])

# 색상 비율 파이차트
plt.subplot(1, 2, 2)
wedges, texts, autotexts = plt.pie(
    ratios,
    colors=colors_rgb,
    startangle=90,
    autopct='%1.1f%%',
    textprops={'color': 'white'}
)

plt.legend(
    wedges,
    [f'{color[::-1].tolist()}' for color in centers],
    title="BGR Colors",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

plt.title("Color Distribution")
plt.axis('equal')

plt.tight_layout()
plt.show()

# 콘솔 출력
for i, (color, count, ratio) in enumerate(zip(centers, counts, ratios)):
    print(f"Cluster {i+1}: BGR={color.tolist()}, Pixels={count}, Ratio={ratio:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()

```
**<결과>**  
<img width="1002" height="362" alt="image" src="https://github.com/user-attachments/assets/10620e52-2692-4f7f-93d5-2305f61bf071" />

**<색상 표와 파이차트>**  
<img width="995" height="468" alt="image" src="https://github.com/user-attachments/assets/44852b33-f971-461d-92a7-36027d54043c" />
  
**<색상 중심값 좌표>**  
<img width="452" height="138" alt="image" src="https://github.com/user-attachments/assets/dc3e66bc-8e8f-476a-99ae-e9f71577717d" />
  

</details> <details> <summary>실습 2</summary>
    
## kNN을 이용해 색상 학습 후 옷 색상 인식하기  

```
import cv2
import numpy as np
import csv
import os

# 색상 정의
color_labels = {
    1:'Red',
    2:'Blue',
    3:'Green',
    4:'Yellow',
    5:'Black',
    6:'White',
    7:'Gray'
}

# 데이터 저장 리스트
collected_data = [] 
current_label = None
mode = "collect"
knn = None

# CSV 파일명
csv_filename = "color_dataset.csv"

# CSV에서 데이터 불러오기
def load_csv_data(filename):
    data = []
    if not os.path.exists(filename):
        return data
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            if len(row) == 4:
                r, g, b, label = row
                data.append([int(r), int(g), int(b), int(label)])
    print(f"CSV에서 {len(data)}개 샘플 로드됨")
    return data

#마우스 클릭
def mouse_callback(event, x, y, flags, param):
    global current_label
    if event == cv2.EVENT_LBUTTONDOWN and current_label is not None and mode == "collect":
        pixel = frame[y, x]
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        print(f'선택된 RGB: ({r}, {g}, {b}), HSV: ({h}, {s}, {v}), 라벨: {color_labels[current_label]}')
        collected_data.append([r, g, b, h, s, v, current_label])

def train_from_memory(data_list):
    x = []
    y = []
    for r, g, b, h, s, v, label in data_list:
        x.append([r / 255.0, g / 255.0, b / 255.0, h / 180.0, s / 255.0, v / 255.0])
        y.append(label)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)

# 프로그램 시작 시 CSV 데이터 로드
collected_data = load_csv_data(csv_filename)

# 웹캠 연결
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cv2.namedWindow('Color Capture')
cv2.setMouseCallback('Color Capture', mouse_callback)

print('1~7 숫자 키로 색상 라벨 지정')
print('s 키: 저장, r 키: 리셋, p 키: 예측 모드 전환, q 키: 종료')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if current_label:
        label_text = f"Current Label: {color_labels[current_label]}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

    # 예측 모드일 때는 중앙 100x100 영역 평균 색상으로 예측
    if mode == "predict" and knn is not None:
        h, w = frame.shape[:2]
        x1, y1 = w//2 - 50, h//2 - 50
        x2, y2 = w//2 + 50, h//2 + 50
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
        avg_color_bgr = np.mean(roi.reshape(-1, 3), axis=0)
        b, g, r = avg_color_bgr
        avg_color_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = avg_color_hsv
        sample = np.array([[r/255.0, g/255.0, b/255.0, h_val/180.0, s_val/255.0, v_val/255.0]], dtype=np.float32)
        ret, results, neighbors, dist = knn.findNearest(sample, k=3)
        pred_label = int(results[0][0])
        pred_name = color_labels.get(pred_label, "Unknown")
        cv2.putText(frame, f"Predict: {pred_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Color Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): # q키를 누르면 종료
        break

    # 숫자 키로 라벨 선택
    elif key in [ord(str(i)) for i in range(1, 8)]:
        current_label = int(chr(key))
        print(f"선택된 라벨: {color_labels[current_label]}")

    # s 키로 CSV 저장
    elif key == ord('s'):
        with open(csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['R', 'G', 'B', 'H', 'S', 'V', 'label'])  # 헤더
            for row in collected_data:
                *features, label_num = row
                label_name = color_labels.get(label_num, "Unknown")
                writer.writerow(features + [label_name])

        print(f"CSV 저장 완료: {csv_filename}")

    # r 키로 데이터 리셋
    elif key == ord('r'):
        collected_data.clear()
        print("데이터 리셋 완료")

    # 데이터 학습 및 예측 모드 전환
    elif key == ord('p'):
        if collected_data:
            X_train, y_train = train_from_memory(collected_data)
            knn = cv2.ml.KNearest_create()
            knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
            mode = "predict"
            print("메모리에서 학습 완료. 예측 모드로 전환.")
        else:
            print("학습할 데이터가 없습니다.")

cap.release()
cv2.destroyAllWindows()
```  


  
**<학습 후 옷 인식>**  
<img width="642" height="512" alt="image" src="https://github.com/user-attachments/assets/374a76b7-0b75-4518-99b7-f56bbee727a6" />  
  
**<색상 데이터가 저장된 csv파일>**  
<img width="489" height="754" alt="image" src="https://github.com/user-attachments/assets/fd92928b-a9c6-4c12-b5b9-5b1a579515c1" />  
