# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:50:53 2024

@author: KIMHYESOO
"""
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


#[1]
# 데이터 불러오기
plt.rcParams['font.family'] = 'NanumGothic'
industry_data = pd.read_csv("Industry - Destination.csv", encoding='cp949')
delivery_data = pd.read_csv("delivery_seoul.csv",encoding='cp949') 
weather_data = pd.read_csv("weather.csv",encoding='cp949')
# 데이터 확인
print(delivery_data.head(10).T)
print(industry_data.head(20))
print(weather_data.head().T)



#[0] 데이터 가공
industry_seoul = industry_data[industry_data['광역시도'] == '서울특별시']
industry_data = industry_data.rename(columns={'자치구': 'ana_ccd_nm'})
industry_seoul=industry_data
industry_seoul= industry_data.rename(columns={'ana_ccd_nm': '시군구명'})
# 'DE' 열을 datetime 형식으로 변환
weather_data['DE'] = pd.to_datetime(weather_data['DE'])


#[0] 시각화
#시간대별 주문량 집계
hourly_delivery = weather_data.groupby(['DE', 'HRLY_TIME'], as_index=False).sum()

# 시각화: 시간대별 배달 주문량의 패턴 확인
plt.figure(figsize=(12, 6))
sns.lineplot(x='HRLY_TIME', y='KORFD_DLVR_CASCNT', data=hourly_delivery, hue='DE', marker='o')
plt.title('시간대별 배달 주문량의 패턴')
plt.xlabel('시간대')
plt.ylabel('배달 주문량')
plt.xticks(range(0, 24))
plt.legend(title='날짜', loc='upper left')
plt.grid(True)
plt.show()





#[1] 데이터 가공
# 열 이름 확인
print(weather_data.columns)
print(delivery_data.columns)

# 날짜 열 이름 변경
weather_data.rename(columns={'DE': '날짜'}, inplace=True)
delivery_data.rename(columns={'crym': '날짜'}, inplace=True)

# 날짜 열을 datetime 형식으로 변환
weather_data['날짜'] = pd.to_datetime(weather_data['날짜'])
delivery_data['날짜'] = pd.to_datetime(delivery_data['날짜'])


# 서울특별시 데이터 필터링
weather_data_seoul = weather_data[weather_data['RTC_NM'].str.contains("서울특별시")]
weather_data_seoul
delivery_data_seoul = delivery_data[delivery_data['ana_mgpo_nm'].str.contains("서울특별시")]
delivery_data_seoul.T

# 데이터 병합
merged_data = pd.merge(delivery_data_seoul, weather_data_seoul, on='날짜', how='inner')  # how='inner'을 명시적으로 설정
# 데이터 확인
print(merged_data.head().T)

#[1] 시각화
# 비오는 날과 맑은 날의 배달 주문량 비교
merged_data['비'] = merged_data['PRCPT_QY_VALUE'] > 0
rainy_days = merged_data[merged_data['비']]
sunny_days = merged_data[~merged_data['비']]

# 온도와 습도가 배달 주문량에 미치는 영향 분석
correlation_temp = merged_data['TMPRT_VALUE'].corr(merged_data['sl_am'])
correlation_humidity = merged_data['HD_VALUE'].corr(merged_data['sl_am'])


plt.figure(figsize=(12, 6))

# 온도와 배달 주문량의 관계
plt.subplot(1, 2, 1)
sns.scatterplot(x='TMPRT_VALUE', y='sl_am', data=merged_data)
plt.title(f'온도와 배달 주문량의 상관관계: {correlation_temp:.2f}')
plt.xlabel('온도 (°C)')
plt.ylabel('배달 주문량')

# 습도와 배달 주문량의 관계
plt.subplot(1, 2, 2)
sns.scatterplot(x='HD_VALUE', y='sl_am', data=merged_data)
plt.title(f'습도와 배달 주문량의 상관관계: {correlation_humidity:.2f}')
plt.xlabel('습도 (%)')
plt.ylabel('배달 주문량')

plt.tight_layout()
plt.show()

print(f"비오는 날 평균 주문량: {rainy_days['sl_am'].mean()}")
print(f"맑은 날 평균 주문량: {sunny_days['sl_am'].mean()}")
print(f"온도와 주문량의 상관관계: {correlation_temp}")
print(f"습도와 주문량의 상관관계: {correlation_humidity}")



# 요일별 배달 주문량 집계
weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekday_avg = delivery_data.groupby('dayofweek')['sl_ct'].mean().reindex(weekday_order)

#[2] 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_avg.index, y=weekday_avg.values)
plt.title('요일별 평균 배달 주문량')
plt.xlabel('요일')
plt.ylabel('평균 배달 주문량')
plt.show()

# 요일별 배달 주문량 평균 출력
print("요일별 평균 배달 주문량:")
print(weekday_avg)


#[3] 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 특정 광역시도 선택 (예: 서울특별시)
selected_province = '서울특별시'

# 선택된 광역시도의 시군구 목록
selected_cities = industry_seoul[industry_seoul['광역시도'] == selected_province]['시군구명'].unique()

# 시군구별 업종별 배달횟수 집계
pivot_table = industry_seoul[industry_seoul['시군구명'].isin(selected_cities)].pivot_table(index='시군구명', columns='업종', values='배달횟수', aggfunc='sum')

# 시각화 설정
plt.figure(figsize=(15, 10))
pivot_table.plot(kind='bar', stacked=True, cmap='viridis')
plt.title(f'{selected_province} 시군구별 업종별 배달횟수')
plt.xlabel('시군구명')
plt.ylabel('배달횟수')
plt.xticks(rotation=45)
plt.legend(title='업종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# 클러스터링
from sklearn.metrics import silhouette_score
# 서울특별시 데이터 필터링
delivery_data = delivery_data.rename(columns={'행정동명(ADMI_NM)':'admd_nm'})
delivery_data.T

# 데이터 병합
merged_data2 = pd.merge(delivery_data, industry_data, on='ana_ccd_nm', how='inner')
merged_data2.columns
selected_columns = ['ana_ccd_nm', 'admd_nm', 'dayofweek', 'sl_am', 'sl_ct', '업종','배달횟수']
selected_data = merged_data2[selected_columns]
selected_data

# 자치구별 업종별 배달횟수 피벗 테이블 생성
industry_pivot = selected_data.pivot_table(index='ana_ccd_nm', columns='업종', values='배달횟수', aggfunc='sum', fill_value=0)

# 자치구별 배달 주문 요일 평균 계산
selected_data['dayofweek'] = pd.Categorical(selected_data['dayofweek'], categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], ordered=True)
dayofweek_avg = selected_data.groupby('ana_ccd_nm')['dayofweek'].apply(lambda x: x.mode().iloc[0]).reset_index()

# 자치구별 배달 주문 요일을 피벗 테이블로 변환
dayofweek_pivot = pd.pivot_table(dayofweek_avg, index='ana_ccd_nm', columns='dayofweek', aggfunc='size', fill_value=0)

# 피벗 테이블 합치기
merged_pivot = pd.concat([industry_pivot, dayofweek_pivot], axis=1).reset_index()

# 클러스터링을 위한 데이터 준비
X = merged_pivot.drop(columns=['ana_ccd_nm'])  # 클러스터링에 사용할 데이터
# 데이터 표준화 (StandardScaler를 사용하여 평균 0, 분산 1로 스케일링)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 최적의 클러스터 수 찾기 (엘보우 기법과 실루엣 점수)
sse = []
silhouette_scores = []
K = range(2, 11)  # 클러스터 개수를 2부터 10까지 시도

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Elbow Method 시각화
plt.figure(figsize=(10, 6))
plt.plot(K, sse, marker='o', label='SSE')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.show()


# 최적의 클러스터 수 설정 (엘보우 기법 및 실루엣 점수 결과에 따라 설정)
optimal_k = 4

# KMeans 클러스터링
kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(X)
clusters = kmeans.labels_

# 클러스터링 결과를 데이터프레임에 추가
merged_pivot['Cluster'] = clusters

# 클러스터별 자치구 출력 예시 (도봉구와 유사한 자치구 찾기)
target_district = '도봉구'
similar_districts = merged_pivot[merged_pivot['ana_ccd_nm'] == target_district]['Cluster'].values[0]
similar_districts = merged_pivot[merged_pivot['Cluster'] == similar_districts]['ana_ccd_nm'].tolist()
similar_districts.remove(target_district)  # 자기 자신은 제외

print(f"{target_district}와 유사한 자치구: {', '.join(similar_districts)}")

highest_sales_by_district = industry_pivot.idxmax(axis=1).to_dict()
def recommend_highest_sales_industry(district):
    return highest_sales_by_district.get(district, "해당 자치구에 대한 데이터가 없습니다.")

highest_sales_industry = recommend_highest_sales_industry(target_district)
print(f"{target_district}의 가장 매출이 높은 업종: {highest_sales_industry}")


# 실루엣 점수 계산
silhouette_avg = silhouette_score(X_scaled, clusters)
print(silhouette_avg)



## 선형회귀분석
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# 데이터 불러오기
age_data = pd.read_csv("city_age.csv", encoding='cp949')
delivery_data = pd.read_csv("delivery_seoul.csv", encoding='cp949')


# delivery_data의 열 이름 변경
delivery_data = delivery_data.rename(columns={'admd_nm': '행정동명(ADMI_NM)'})

# 데이터 병합
merged_data = pd.merge(age_data, delivery_data, on='행정동명(ADMI_NM)', how='inner')

# 병합된 데이터 확인
print(merged_data.head().T)

# 필요한 변수 선택
X = merged_data[['행정동코드(ADMI_CD)', 'dayofweek', 'sl_ct', '거주인구_수(RSPOP_CNT)']]
y = merged_data['sl_am']  # 예상 매출을 종속 변수로 설정

# 범주형 데이터를 더미 변수로 변환
X = pd.get_dummies(X, columns=['dayofweek'])

# 모델 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 사용자 입력
input_admi_nm = '쌍문1동'  # 사용자가 입력한 행정동명

# 입력된 행정동명을 가진 데이터 추출
user_data = merged_data[merged_data['행정동명(ADMI_NM)'] == input_admi_nm].iloc[0]

# 사용자 데이터에서 필요한 변수 선택 (종속 변수 제외)
# 예시로 행정동코드(ADMI_CD), dayofweek, sl_ct, 거주인구_수(RSPOP_CNT)을 선택했다고 가정합니다.
# 실제 필요한 변수에 따라 선택하는 부분을 수정해야 합니다.
user_X = user_data[['행정동코드(ADMI_CD)', 'dayofweek', 'sl_ct', '거주인구_수(RSPOP_CNT)']].to_frame().T

# 범주형 데이터 더미 변수로 변환
# dayofweek 컬럼을 가진 경우에만 get_dummies 함수를 사용하여 더미 변수로 변환합니다.
if 'dayofweek' in user_X.columns:
    user_X = pd.get_dummies(user_X, columns=['dayofweek'])

# 학습 데이터와 동일한 컬럼 구성을 확인하고, 없는 컬럼은 0으로 채워 넣습니다.
# X_train.columns에는 학습 데이터에서 사용된 모든 컬럼이 포함되어 있어야 합니다.
for col in X_train.columns:
    if col not in user_X.columns:
        user_X[col] = 0

# 입력 데이터를 모델 형식에 맞게 재정렬
# X_train.columns와 동일한 순서로 컬럼을 재정렬합니다.
user_X = user_X[X_train.columns]

# 다음 달 매출 예측
predicted_sales = model.predict(user_X.values.reshape(1, -1))

print(f"{input_admi_nm}의 다음 달 예상 매출: {predicted_sales[0]} 건")



#데이터 정보
delivery_data.info()
industry_data.info()
industry_seoul.info()
weather_data.info()
age_data.info()
merged_data.info()
