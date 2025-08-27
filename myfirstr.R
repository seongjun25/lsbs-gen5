a <- 1
a

a = c(1:10)
a = a * 2
a[3:5]
a[1]

a = matrix(c(1:20), ncol=4)
a[2:4, 2:3]
a[,2]

cars$speed
cars$dist

head(cars)
tail(cars)

# Ctrl + Shift + m
# 파이프 연산자
cars |> head()




# 데이터 분석, 연구소, 바이오 분야, 정부기관(한국은행)
# R, SAS (유료), SPSS

# R
# 통계, 데이터 분석 전용
# .exe 프로그램 만들기는 부적합
# 시각화, 통계분석
# 여러언어를 잘 다루는 전담 팀들 존재하는 회사는
# R을 사용해도 무방, 훨씬 편함
# 데이터분석가는 완전 해피

# Python
# 범용 언어 - 다재다능
# 사용하는 사람들이 여러 백그라운드를 가지고 있음
# 백엔드 엔지니어들, 데이터 분석가들 ..
# 분석가를 지원하는 팀이 사이즈가 작다면?
# 데이터 분석가가 백엔드 혹은 다른 팀들에게 맞춰줘야
# 함 - 을의 입장


# 필요한 패키지 로드
# install.packages("tidyverse")
# install.packages("palmerpenguins")
# install.packages(c("tidyverse", "palmerpenguins"))
library(tidyverse)
library(palmerpenguins)

# 결측치 제거
penguins_clean <- penguins |>
  filter(!is.na(bill_length_mm), 
         !is.na(bill_depth_mm),
         !is.na(species))
penguins_clean

# 종별 평균 부리 길이 계산
mean_bill_length <- penguins_clean |>
  group_by(species) |>
  summarise(평균_부리_길이_mm = mean(bill_length_mm))

# 결과 출력
print(mean_bill_length)

# 종별 산점도 (부리 길이 vs 부리 깊이)
ggplot(penguins_clean, 
  aes(x = bill_length_mm, y = bill_depth_mm, color = species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(
    title = "펭귄 종별 부리 길이 vs 부리 깊이",
    x = "부리 길이 (mm)",
    y = "부리 깊이 (mm)",
    color = "종"
  ) +
  theme_minimal()
