def add(a, b):
    result = a + b
    return result

add(3, 4)
add(a=3, b=4)
add(b=4, a=3)

# 매개변수: parameters
# 인수: arguments

def say():
    return "hi"

say()

# p.121
# if 구문

money = 5000
card = True

if ((money > 4500) or (card == True)):
    print("택시를 타세요.")
else:
    print("걸어가세요.")

# 시험 점수가 60점 이상이면 합격,
#  그렇지 않으면 불합격을
# 출력하는 if문 작성
score = 50

if score >= 60:
    print("합격")
else:
    print("불합격")

# x의 수가 홀수이면서 7의 배수이면
# "7의배수이면서 홀수"를 출력,
# 그렇지 않으면, "조건 불만족" 출력
x = 14
if ((x % 2 == 1) and (x % 7 == 0)):
    print("7의배수이면서 홀수")
else:
    print("조건 불만족")

# M이면 "남자입니다"
# F이면 "여자입니다"
# nan이면 "비어있습니다"
gender = "M"

if (gender == "M"):
    print("남자입니다.")
elif (gender == "F"):
    print("여자입니다.")
else:
    print("비어있습니다.")

# 다음은 공원 입장료 정보입니다.
# 나이에 따른 입장료를 계산하는 if문 작성
# 유아(7세 이하): 무료
# 어린이: 3000원
# 성인(20세 이상): 7000원
# 노인(60세 이상): 5000원
# cal_price() 함수를 만들어보세요!
# 매개변수 age
# 예: cal_price(10) -> 3000
age = 10
price = 0 # 입장료 값을 받을 변수

if (age <= 7):
    price = 0
elif (age < 20):
    price = 3000
elif (age < 60):
    price = 7000
else:
    price = 5000

price

def cal_price(age):
    # 조건 체크
    if (age <= 7):
        price = 0
    elif (age < 20):
        price = 3000
    elif (age < 60):
        price = 7000
    else:
        price = 5000
    # 결과 반환    
    return price
    
cal_price(10)


# while 문 p.133
# 조건을 만족하는 (True) 동안 코드를 반복실행
treeHit = 0
while treeHit < 10:
    treeHit += 1
    print(f"나무를 {treeHit}번 찍었습니다.")
    if treeHit == 6:
        print("일 그만!")
        break

# break와 continue
a = 0
while a < 10:
    a += 1
    if (a % 2 == 0): # a가 짝수인경우
        continue     # while 루프 처음으로 넘어가
    print(a)


# for 루프
# for 변수 in 순서가있는객체:
#     반복할 내용1
#     반복할 내용2
#     ...

test_list = ["one", "two", "three"]
for i in test_list:
    print(i)

a = [(1, 2), (3, 4), (5, 6), (7, 8)]

for (fir, snd) in a:
    print(fir + snd)

# Q. 1에서 100까지 넘파이 벡터 만들기
import numpy as np
a=np.arange(1, 101)

for i in a:
    if (i % 7 == 0):
        continue
    print(i)

[x**2 + 3 for x in range(1, 6)]


a = [1, 2, 4, 3, 5]
# a의 각 원소에 3을 곱한 값을 다시 리스트로 만들기
a + [10, 10]
a * 2
[num*3 for num in a]

[x**2 for x in range(1, 11)]
# for x in range(1, 11):
#     print(x**2)

for x in range(1, 21):
    if (x % 2 == 0): print(x)

[x for x in range(1, 21) if (x % 2 == 0)]

nums = [-3, 5, -1, 0, 8]
# 리스트에 있는 숫자 중에서 음수는 0으로 바꾸고, 
# 양수는 그대로 유지
[0 if (x < 0) else x for x in nums]

[x if x >= 0 else 0 for x in nums]

words = ['apple', 'banana', 'cherry', 'avocado']
for i in words:
    if (i.startswith("a")):
        print(i)

[i for i in words if (i.startswith("a"))]

# "apple".startswith("a")
# "banana".startswith("a")
# "cherry".startswith("a")


# def 함수(*par):
#     수행할 문장

def add_many(*nums):
    result=0
    for i in nums:
        result = result + i
    return result

add_many(3, 4, 2)
nums=[3, 4, 2]

def cal_how(method, *nums):
    if (method == "add"):
        result = 0
        for i in nums:
            result += i
    elif (method == "mul"):
        result = 1
        for i in nums:
            result *= i
    else:
        print("해당연산 수행할 수 없음")
        result = None
    return result

cal_how("add", 3, 2, 5, 4)
cal_how("mul", 3, 2, 5, 4)
cal_how("squared", 3, 2, 5, 4)


def add_and_mul(a=5, b=4):
    return (a+b, a * b)

add_and_mul(b=3)
result = add_and_mul(3)
result[0]
result[1]

import seaborn as sns
import pandas as pd
df = sns.load_dataset('titanic')
df = df.dropna()
df.info()
