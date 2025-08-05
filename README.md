# OpenCV

## 실습

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

