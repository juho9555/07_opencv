import numpy as np
import cv2

k = 16  # 군집화 개수
img = cv2.imread('../img/image.png')

if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

# 데이터 평균을 구할 때 실수형 변환
data = img.reshape((-1, 3)).astype(np.float32)

# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환
center = np.uint8(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]

# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Kmeans color', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
