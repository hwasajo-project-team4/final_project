import streamlit as st
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, accuracy
from collections import defaultdict


def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

SEED = 42
reset_seeds(SEED)

df = pd.read_csv("아모레크롤링_스킨케어.csv")

# 제품별 평균 평점과 리뷰 수 계산
product_rating_avg = df.groupby('상품명')['별점'].mean()
product_rating_count = df.groupby('상품명').size()

# 가중 평점 계산
m = product_rating_count.quantile(0.6)
C = product_rating_avg.mean()
product_weighted_rating = (product_rating_count / (product_rating_count + m) * product_rating_avg) + (m / (product_rating_count + m) * C)

# 가중 평점을 데이터프레임에 추가
df['가중평점'] = df['상품명'].map(product_weighted_rating)

# 가상 유저 생성
df['가상유저'] = df['나이'] + ',' + df['성별'] + ',' + df['피부타입'] + ',' + df['피부트러블']

# 각 가상 유저별 리뷰 수 계산
user_review_counts = df['가상유저'].value_counts()

# 가상 유저와 상품명을 ID로 변환
user_to_id = {user: i for i, user in enumerate(df['가상유저'].unique())}
product_to_id = {product: j for j, product in enumerate(df['상품명'].unique())}
df['user_id'] = df['가상유저'].map(user_to_id)
df['product_id'] = df['상품명'].map(product_to_id)

# 가상유저별 총 구매횟수를 계산
user_total_purchase_count = df.groupby('가상유저').size().reset_index(name='총구매횟수')

# 구매횟수를 기반으로 10%씩 묶어 클래스를 생성
user_total_purchase_count['구매_클래스'] = pd.qcut(user_total_purchase_count['총구매횟수'], 10, labels=False)

# 원본 데이터에 구매 클래스 정보 추가
df = pd.merge(df, user_total_purchase_count[['가상유저', '구매_클래스']], on='가상유저', how='left')

train_df, test_df = train_test_split(df,test_size=0.2,random_state=SEED,stratify=df['구매_클래스'])

# Reader 객체 생성
reader = Reader(rating_scale=(0, 5))

# 학습 데이터와 테스트 데이터를 surprise의 데이터 형식으로 변환
train_data_surprise = Dataset.load_from_df(train_df[['user_id', 'product_id', '가중평점']], reader)
trainset = train_data_surprise.build_full_trainset()

# 테스트 데이터를 surprise의 데이터 형식으로 변환
testset = [(row['user_id'], row['product_id'], row['가중평점']) for i, row in test_df.iterrows()]

best_params = {'n_epochs': 100, 'lr_all': 0.005, 'reg_all': 0.2}
# SVD 알고리즘 사용하여 모델 학습
model = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'],random_state=SEED)
model.fit(trainset)

# 테스트 데이터에 대한 예측
predictions = model.test(testset)

# 평가 (RMSE)
rmse = accuracy.rmse(predictions)

id_to_user = {v: k for k, v in user_to_id.items()}
id_to_product = {v: k for k, v in product_to_id.items()}

def get_top_n_recommendations(predictions, n=5):
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        user_info = id_to_user[uid]
        product_name = id_to_product[iid]

        if user_info not in top_n:
            top_n[user_info] = []

        top_n[user_info].append((product_name, est))

    # 정렬, 중복 제거
    for user_info, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        top_n_without_duplicates = []
        for product_name, est in user_ratings:
            if product_name not in seen:
                seen.add(product_name)
                top_n_without_duplicates.append((product_name, est))

        top_n[user_info] = top_n_without_duplicates[:n]

    return top_n

def get_unrated_items(user, df):
    # 사용자가 평가한 아이템들
    rated_items = set(df[df['가상유저'] == user]['상품명'].tolist())
    # 전체 아이템들
    all_items = set(df['상품명'].tolist())
    # 평가하지 않은 아이템들
    unrated_items = all_items - rated_items
    return unrated_items

user_recommendations_with_rated = get_top_n_recommendations(predictions, n=5)

def content_based_recommendation_with_weights(age, gender, skin_type, skin_trouble, top_n=5, weight=0.1):
    # 사용자 정보와 일치하는 리뷰 데이터 필터링
    filtered_df = df[(df['나이'] == age) & (df['성별'] == gender) &
                     (df['피부타입'] == skin_type) & (df['피부트러블'] == skin_trouble)]

    # 상품별 평균 별점 계산
    product_rating_avg = filtered_df.groupby('상품명')['별점'].mean().reset_index()

    # 가중치 적용: 일치하는 특성이 있을 경우, 가중치를 더한다.
    feature_values = {'나이': age, '성별': gender, '피부타입': skin_type, '피부트러블': skin_trouble}
    for feature, feature_value in feature_values.items():
        feature_weight = filtered_df[filtered_df[feature] == feature_value].groupby('상품명')['별점'].count() * weight
        product_rating_avg = pd.merge(product_rating_avg, feature_weight.reset_index().rename(columns={'별점': f'{feature}_weight'}), on='상품명', how='left')

    # 최종 점수 계산 (평균 별점 + 가중치 합)
    product_rating_avg['final_score'] = product_rating_avg['별점'] + product_rating_avg[[f'{feature}_weight' for feature in ['나이', '성별', '피부타입', '피부트러블']]].sum(axis=1)

    # 최종 점수가 높은 상위 N개의 상품 추천
    recommended_products = product_rating_avg.sort_values(by='final_score', ascending=False).head(top_n)['상품명'].tolist()

    return recommended_products

def recommend_products_for_user(age, gender, skin_type, skin_trouble, top_n=5):
    # 가상 유저 ID를 생성
    virtual_user = f"{age},{gender},{skin_type},{skin_trouble}"

    # 가상 유저의 리뷰 수 확인
    user_review_count = df[df['가상유저'] == virtual_user].shape[0]

    # 가상유저별 총 구매횟수를 계산
    user_total_purchase_count = df.groupby('가상유저').size().reset_index(name='총구매횟수')

    # 구매횟수 상위 20%에 해당하는 임계값을 계산
    heavy_user_threshold = user_total_purchase_count['총구매횟수'].quantile(0.8)

    # 리뷰 수 상위 20% 이하인 경우 라이트 유저로 판단
    if user_review_count <= heavy_user_threshold:
        return content_based_recommendation_with_weights(age, gender, skin_type, skin_trouble, top_n=top_n)
    else:
        user_id = user_to_id[virtual_user]
        # CF 기반 추천 수행
        user_recommendations = user_recommendations_with_rated.get(virtual_user, [])
        recommended_products = [product_name for product_name, _ in user_recommendations[:top_n]]
        return recommended_products
    

st.sidebar.title('Cosmetic Recommend')
st.sidebar.header('추천받고 싶은 유형을 선택하세요')
if st.sidebar.checkbox("상품명 입력"):
    product = st.sidebar.text_input(label="상품명", value="default value")
    if st.sidebar.button("추천받기"):
        st.header(f"{product}와 유사한 제품입니다.")
        selected_product = st.selectbox('궁금한 제품을 선택하세요.', ['1.제품','2.제품','3.제품','4.제품','5.제품'])
        st.write('선택하신 제품은 {}입니다.'.format(selected_product))


if st.sidebar.checkbox("고객타입 입력"):
    gender = st.sidebar.selectbox("성별",["남성","여성"])
    age = st.sidebar.selectbox("나이",["10대","20대","30대","40대","50대 이상"])
    skintype = st.sidebar.selectbox("피부타입",["복합성","건성","수분부족지성","지성","중성","극건성"])
    skintrouble = st.sidebar.selectbox("피부트러블",["민감성","건조함","탄력없음","트러블","주름","모공","칙칙함","복합성"])
    if st.sidebar.button("추천받기"):
        st.header(f"{gender}, {age}, {skintype}, {skintrouble} 타입 고객님께 추천하는 제품입니다.")
        recommend_list = recommend_products_for_user(age,gender,skintype,skintrouble)
        for rec in recommend_list:
            st.write(rec)


