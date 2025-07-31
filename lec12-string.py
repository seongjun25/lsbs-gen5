import pandas as pd
data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)
df

df.info()

df['제품명_길이'] = df['가전제품'].str.len()
df['브랜드_길이'] = df['브랜드'].str.len()
df

df['브랜드'] = df['브랜드'].str.lower()
df

df['브랜드'].str.contains("i")
df['브랜드'].str.startswith("s")
df['브랜드'].str.endswith("g")

df['가전제품'].str.replace("에어컨", "선풍기")
df['가전제품'].str.replace("기", "")

df['브랜드'].str.split("a", expand=True)

df['가전제품']
df['브랜드']
df['제품_브랜드']  = df['가전제품'].str.cat(df['브랜드'], sep=', ')
df



df['가전제품'] = df['가전제품'].str.replace('전자레인지', ' 전자 레인지  ')
df['가전제품'].str.replace(" ", "")


data = {
    '주소': ['서울특별시 강남구 테헤란로 123', '부산광역시 해운대구 센텀중앙로 45', '대구광역시 수성구 동대구로 77-9@@##', '인천광역시 남동구 예술로 501&amp;&amp;, 아트센터', '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)
print(df.head(2))


df['주소'].str.extract(r'(\d)', expand=False)

df = pd.DataFrame({
    'text': [
        'apple',        # [aeiou], (a..e), ^a
        'banana',       # [aeiou], (ana), ^b
        'Hello world',  # ^Hello, world$
        'abc',          # (abc), a.c
        'a1c',          # a.c
        'xyz!',         # [^aeiou], [^0-9]
        '123',          # [^a-z], [0-9]
        'the end',      # d$, e.
        'space bar',    # [aeiou], . (space)
        'hi!',           # [^0-9], [aeiou]
        'blue',
        'lue'
    ]
})
df["text"].str.extract(r'([aeiou])')
df["text"].str.extractall(r'([aeiou])')

df["text"].str.extract(r'([^0-9])')
# df["text"].str.extractall(r'([^0-9])')

df["text"].str.extract(r'(a.c)')
df["text"].str.extract(r'(^Hello)')
df["text"].str.extract(r'(b?lue)')
df["주소"].str.extract(r'([가-힣]+(?:특별시|광역시))')

# 실습 데이터
import pandas as pd
data = {
    '주소': ['서울특별시 강남구 테헤란로 123', '부산광역시 해운대구 센텀중앙로 45', '대구광역시 수성구 동대구로 77-9@@##', '인천광역시 남동구 예술로 501&amp;&amp;, 아트센터', '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)
print(df.head(2))

df["도시"] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
df
 
df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')

df['주소_특수문자제거'] = df['주소'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)
df['주소_특수문자제거'] = df['주소_특수문자제거'].str.replace("ampamp", "")
df['주소_특수문자제거']

# 정규표현식 연습
df=pd.read_csv("./data/regex_practice_data.csv")
df
  
# 이메일 잡아오기
df['전체_문자열'].str.extract(r'([\w\.]+@[\w\.]+)')

# 핸드폰 번호 잡아오기 + 핸드폰 번호 입력한 사람들 정보
df['전체_문자열'].str.extract(r'(010-[0-9\-]+)').dropna()

# 일반 번호 잡아오기
phone_num=df['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
phone_num.iloc[:,0]
~phone_num.iloc[:,0].str.startswith("01")
phone_num.loc[~phone_num.iloc[:,0].str.startswith("01"),:]

# 주소에서 '구' 단위만 추출하기
df['전체_문자열'].str.extract(r'(\b\w+구\b)')
df['전체_문자열'].str.extract(r'([가-힣]+구)')

# 날짜(YYYY-MM-DD) 형식 찾기
df['전체_문자열'].str.extract(r'(\d{4}-\d{2}-\d{2})')

# 날짜 형식 가져오기
df['전체_문자열'].str.extract(r'(\d{4}\W\d{2}\W\d{2})')
df['전체_문자열'].str.extract(r'(\d{4}[-/.]\d{2}[-/.]\d{2})')

# 가격 정보(₩ 포함) 찾기
df['전체_문자열'].str.extract(r'(₩[\d,]+)')

# 가격에서 숫자만 추출하기 (₩ 제거)
df['전체_문자열'].str.extract(r'₩([\d,]+)')
# df['전체_문자열'].str.extract(r'₩(\d+\,?[\d,]+)')

# 이메일의 도메인 추출하기
# @
df['전체_문자열'].str.extract(r'@([\w.]+)')

# 데이터에서 한글 이름만 추출하세요.
df['전체_문자열'].str.extract(r'([가-힣]+)')


df=pd.read_csv("./data/소상공인시장진흥공단_상가(상권)정보_세종_202503.csv")
df.head()
df.info()

pd.set_option('display.max_columns', None)  # 모든 열 보기
pd.set_option('display.max_rows', None)     # 모든 행 보기
pd.set_option('display.width', None)        # 한 줄로 넓게 보기
pd.set_option('display.max_colwidth', None) # 열 내용 길이 제한 해제
df.loc[:, df.columns.str.contains("상권업종")].head()

