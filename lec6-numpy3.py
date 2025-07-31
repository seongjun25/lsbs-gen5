import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
img1

# 행렬 이미지화
plt.figure(figsize=(10, 5))  # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();


img_mat = np.loadtxt('./data/img_mat.csv', delimiter=',', skiprows=1)
img_mat

np.min(img_mat)
np.max(img_mat)


# 행렬 값을 0과 1 사이로 변환
img_mat = img_mat / 255.0
img_mat.shape

# 1단계: 전체에 0.2를 더함
# 2단계: 1이상 값을 가지는 애들 -> 1 변환
# 0.2 더하기
img_mat = img_mat + 0.5
img_mat = img_mat / 1.5
img_mat

# 필터링: 1보다 큰 값은 1로 설정
# img_mat[img_mat > 1.0] = 1.0

# 행렬을 이미지로 변환하여 출력
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();


x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)


x.transpose()

# 행렬 x, y 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
y = np.arange(1, 7).reshape((2, 3))
print("행렬 x:\n", x)
print("행렬 y:\n", y)

x.shape
y.shape

x.dot(y)
np.matmul(x, y)

# mat_A
# (1, 2,
#  4, 3)

# mat_B
# (2, 1,
#  3, 1)

# 두 행렬의 곱을 구해보세요.
mat_A = np.array([[1, 2],
                  [4, 3]])

mat_B = np.array([[2, 1],
                  [3, 1]])

mat_A.dot(mat_B)
np.matmul(mat_A, mat_B)
mat_A @ mat_B

mat_A * mat_B

# 역행렬
mat_A
inv_A=np.linalg.inv(mat_A)
mat_A @ inv_A

2 * 3
3 * (1/3)
# 행렬의 세계: 1 == 단위행렬
# 행렬의 세계: 역수 == 역행렬
mat_A @ np.eye(2)

mat_C = np.array([[3, 1],
                  [6, 2]])
inv_C=np.linalg.inv(mat_C)

# 행렬
# 역행렬이 존재하는 행렬 vs. 존재x 행렬
# non-singular vs. singular
# 칼럼이 선형 독립인 경우 -> 역행렬 존재
# 칼럼이 선형 종속인 경우 -> 역행렬 존재 x


# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array
my_array.shape

my_array[0, :, :]
my_array[1, :, :]
my_array[:, 1, :]

my_array.reshape(2, 3, 2)
my_array.reshape(-1, 3, 2)


import imageio
# 이미지 읽기
jelly = imageio.imread("./data/stat.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)

import matplotlib.pyplot as plt
plt.imshow(jelly);
plt.axis('off');
plt.show();

# 흑백으로 변환
bw_jelly = np.mean(jelly[:, :, :3], axis=2)
jelly[:, :, 3].max()
jelly[:, :, 3].min()
bw_jelly.shape
plt.imshow(bw_jelly, cmap='gray');
plt.axis('off');
plt.show();


# 연습문제 7
import numpy as np

# 행렬 정의
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = np.array([[9, 10],
              [11, 12]])

# 3D 행렬 T 생성: A와 B를 세로로 쌓음
T = np.stack([A, B], axis=0)  # shape: (2, 2, 2)
T.shape

# T의 각 면(slice)와 C 곱하기
result = np.matmul(T, C)
C.shape
result
A @ C
B @ C


# 멱등(idempotent) 행렬
mat_S = np.array([[2, -1],
                  [-1, 2]])
mat_S.transpose() @ mat_S

np.linalg.inv(mat_S)
mat_S.trace()