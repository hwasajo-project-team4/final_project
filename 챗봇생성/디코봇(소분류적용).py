import discord
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Okt
from collections import Counter
import asyncio
import pandas as pd
import time
import pandas as pd
import numpy as np
from eunjeon import Mecab
import matplotlib.pyplot as plt
import os
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split
from collections import defaultdict

mecab = Mecab()


new_df = pd.read_csv("키워드추출.csv")

new_df['키워드'].fillna("nan")

skincare = pd.read_csv("아모레크롤링_스킨케어.csv")

TOKEN = '본인의 디스코드봇 토큰'
CHANNEL_ID = '본인의 디스코드 채널 ID'


model_name = "noahkim/KoT5_news_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


product_rating_avg = skincare.groupby('상품명')['별점'].mean()
product_rating_count = skincare.groupby('상품명').size()

# 가중 평점 계산
m = product_rating_count.quantile(0.6)
C = product_rating_avg.mean()
product_weighted_rating = (product_rating_count / (product_rating_count + m) * product_rating_avg) + (m / (product_rating_count + m) * C)
# 가중 평점을 데이터프레임에 추가
skincare['가중평점'] = skincare['상품명'].map(product_weighted_rating)

# 가상 유저 생성
skincare['가상유저'] = skincare['나이'] + ',' + skincare['성별'] + ',' + skincare['피부타입'] + ',' + skincare['피부트러블']

# 각 가상 유저별 리뷰 계산
user_review_counts = skincare['가상유저'].value_counts()

# 가상 유저와 상품명을 ID로 변환
user_to_id = {user: i for i, user in enumerate(skincare['가상유저'].unique())}
product_to_id = {product: j for j, product in enumerate(skincare['상품명'].unique())}
skincare['user_id'] = skincare['가상유저'].map(user_to_id)
skincare['product_id'] = skincare['상품명'].map(product_to_id)

# 가상 유저별 총 구매횟수를 계산
user_total_purchase_count = skincare.groupby('가상유저').size().reset_index(name='총구매횟수')

# 구매 횟수를 기반으로 10%씩 묶어 클래스를 생성
user_total_purchase_count['구매_클래스'] = pd.qcut(user_total_purchase_count['총구매횟수'], 10, labels=False)

# 원본 데이터에 구매 클래스 정보 추가
skincare = pd.merge(skincare, user_total_purchase_count[['가상유저', '구매_클래스']], on='가상유저', how='left')

train_df, test_df = train_test_split(skincare,test_size=0.2,random_state=42,stratify=skincare['구매_클래스'])

# Reader 객체 생성
reader = Reader(rating_scale=(0, 5))

# 학습 데이터와 테스트 데이터를 surprise의 데이터 형식으로 변환
train_data_surprise = Dataset.load_from_df(train_df[['user_id', 'product_id', '가중평점']], reader)
trainset = train_data_surprise.build_full_trainset()

# 테스트 데이터를 surprise의 데이터 형식으로 변환
testset = [(row['user_id'], row['product_id'], row['가중평점']) for i, row in test_df.iterrows()]
best_params = {'n_epochs': 100, 'lr_all': 0.005, 'reg_all': 0.2}

# SVD 알고리즘 사용하여 모델 학습
model1 = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'],random_state=42)
model1.fit(trainset)

# 테스트 데이터에 대한 예측
predictions = model1.test(testset)

# 평가(RMSE)
rmse = accuracy.rmse(predictions)

id_to_user = {v: k for k, v in user_to_id.items()}
id_to_product = {v: k for k, v in product_to_id.items()}
     
RMSE: 0.0067

def get_top_n_recommendations(predictions, n=5):
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        user_info = id_to_user[uid]
        product_name = id_to_product[iid]

        if user_info not in top_n:
            top_n[user_info] = []

        top_n[user_info].append((product_name, est))

    #정렬, 중복 제거
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

def content_based_recommendation_with_weights(age, gender, skin_type, skin_trouble, df, top_n=5, weight=0.1):
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

# 상품 카테고리를 기반으로 상품 추천하는 함수
def recommend_products_by_category(df, category, top_n=5):
    # 카테고리에 해당하는 상품 필터링
    filtered_df = df[df['상품소분류'] == category]

    # 필터링된 상품들의 가중평점 평균 계산
    product_rating_avg = filtered_df.groupby('상품명')['가중평점'].mean().reset_index()

    # 가중평점 평균 기준으로 상위 N개 상품 추출
    recommended_products = product_rating_avg.sort_values(by='가중평점', ascending=False).head(top_n)['상품명'].tolist()

    return recommended_products

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        await self.change_presence(status=discord.Status.online, activity=discord.Game("구동중"))

        channel = self.get_channel(int(CHANNEL_ID))
        if channel:
            introduction_message = (
                "환영합니다! 저는 여러분을 도와드릴 챗봇입니다.\n"
                "저의 기능은 다음과 같습니다:\n"
                "- `!review`: 상품 리뷰 요약\n"
                "- `!keyword`: 키워드 추출\n"
                "- `!recommand`: 추천\n"
                "종료하려면 'q' 또는 'quit' 또는 '!quit'를 입력하세요."
            )
            await channel.send(introduction_message)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_matching_titles(self, title):
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in skincare['상품명'].unique().tolist() if condition(t)]
        return matching_titles
    
    async def get_product_title(self, message):
        while True:
            try:
                msg = await self.wait_for('message', timeout=60.0, check=lambda m: m.author == message.author and m.channel == message.channel)
                
                if msg.content in ['q', '!quit', 'quit']:
                    await msg.channel.send('대화를 종료합니다.')
                    await self.close()
                    return None
                
                title = msg.content.strip()
                
                if len(title) > 1:
                    matching_titles = self.find_matching_titles(title)
                    if len(matching_titles) == 0:
                        await message.channel.send("해당 상품이 없습니다. 상품명을 똑바로 입력해 주세요.")
                        continue
                    else:
                        return title
                else:
                    await message.channel.send("최소 두 글자 이상의 쿼리를 입력해주세요.")
                    
            except asyncio.TimeoutError:
                await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')
                return await self.get_product_title(message)

            


    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.channel.id != int(CHANNEL_ID):
            return


        if message.content.startswith('!review'):
                await message.channel.send("상품 리뷰 요약을 선택하셨습니다. 상품명을 입력해 주세요.")
                title = await self.get_product_title(message)
                if title:
                    await self.chatbot(message, title)
                return


        elif message.content.startswith('!keyword'):
            await message.channel.send("키워드 추출을 선택하셨습니다. 상품명을 입력해 주세요.")
            title = await self.get_product_title(message)
            if title:
                await self.keyword_bot(message, title)
            return

        elif message.content.startswith('!recommand'):
            await message.channel.send("상품 추천을 선택하셨습니다.")
            await self.recommend_chatbot(skincare, new_df, message)
        

        elif message.content in ['q', '!quit', 'quit']:
            await message.channel.send('대화를 종료합니다.')
            await self.close()
            return



    async def recommend_products_for_user(self, skincare, message, top_n=5):
        await message.channel.send("\n연령대를 입력하세요 (10대, 20대, 30대, 40대, 50대 이상): ")
        age = await self.wait_for_user_response(message)
        if not age: 
            return

        await message.channel.send("\n성별을 입력하세요 (여성, 남성): ")
        gender = await self.wait_for_user_response(message)
        if not gender:
            return

        await message.channel.send("\n피부 타입을 입력하세요 (지성, 건성, 복합성, 수분부족지성, 중성, 극건성): ")
        skin_type = await self.wait_for_user_response(message)
        if not skin_type:
            return

        await message.channel.send("\n피부 고민을 입력하세요 (모공, 건조함, 트러블, 민감성, 탄력없음, 칙칙함, 주름, 복합성): ")
        skin_trouble = await self.wait_for_user_response(message)
        if not skin_trouble:
            return

        # 카테고리 입력 추가
        await message.channel.send("\n카테고리를 입력하세요 (에센스 & 세럼, 로션 & 에멀젼, 스킨 & 토너, 크림, 아이 & 넥, 미스트, 선블럭, 선스틱, \n마스크 & 팩, 페이스 오일, 앰플, 메이크업 리무버, 클렌징 티슈, 클렌징 오일, 클렌징 워터, 클렌징 폼, 필링 & 스크럽): ")
        product_category = await self.wait_for_user_response(message)
        if not product_category:
            return

        # 가상 유저 ID를 생성
        virtual_user = f"{age},{gender},{skin_type},{skin_trouble}"

        # 가상 유저의 리뷰 수 확인
        user_review_count = skincare[skincare['가상유저'] == virtual_user].shape[0]

        # 가상유저별 총 구매횟수를 계산
        user_total_purchase_count = skincare.groupby('가상유저').size().reset_index(name='총구매횟수')

        # 구매횟수 상위 20%에 해당하는 임계값을 계산
        heavy_user_threshold = user_total_purchase_count['총구매횟수'].quantile(0.8)

        # 리뷰 수 상위 20% 이하인 경우 라이트 유저로 판단
        if user_review_count <= heavy_user_threshold:
            # 카테고리 기반 추천 수행
            recommended_products = recommend_products_by_category(skincare, product_category, top_n=top_n)
        else:
            user_id = user_to_id[virtual_user]
            # CF 기반 추천 수행
            user_recommendations = user_recommendations_with_rated.get(virtual_user, [])
            recommended_products = [product_name for product_name, _ in user_recommendations[:top_n]]

        # 추천된 상품을 디스코드 메시지로 전송
        await message.channel.send(f"추천 상품: {', '.join(recommended_products)}")


    
    async def chatbot(self, message, title, send_intro=True):
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in skincare['상품명'].unique().tolist() if condition(t)]
        
        if len(matching_titles) == 0:
            await message.channel.send("해당 상품이 없습니다. 상품명을 똑바로 입력해 주세요.")
            return
        elif len(title) == 1:
            await message.channel.send("최소 두 글자 이상의 쿼리를 입력해주세요.")
            return

        response = "일치하는 상품명 목록:\n"
        for idx, matching_title in enumerate(matching_titles, start=1):
            response += f"{idx}. {matching_title}\n"

        max_len = 2000

        if len(response) > max_len:
            for i in range(0, len(response), max_len):
                await message.channel.send(response[i:i + max_len])
        else:
            await message.channel.send(response)
        
        def check(m):
            return m.author == message.author and m.channel == message.channel

        try:
            while True:
                msg = await self.wait_for('message', timeout=60.0, check=check)
                if msg.content in ['q', '!quit', 'quit']:   
                    await msg.channel.send('대화를 종료합니다.')
                    await self.close()  
                    return
            
                try:
                    selected_idx = int(msg.content) - 1
                except ValueError:
                    await message.channel.send("숫자만 입력해 주세요.")
                    continue

                if 0 <= selected_idx < len(matching_titles):
                    selected_title = matching_titles[selected_idx]
            
                    reviews = skincare[skincare['상품명'] == selected_title]['리뷰'].to_list()
                    reviews_text = ' '.join(reviews)
                    
                    inputs = tokenizer(reviews_text, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = model.generate(inputs["input_ids"], num_beams=5, min_length=30, max_length=300, early_stopping=True)
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                    
                    await message.channel.send(f"{selected_title}의 전체 리뷰 요약: {summary}")
                    # 선택된 상품의 데이터 추출
                    reviews_df = skincare[skincare['상품명'] == selected_title]

                    # 상품에 대한 대표적인 사용자의 특성 정보 추출
                    age = reviews_df['나이'].mode().values[0]
                    types = reviews_df['피부타입'].mode().values[0]
                    trouble = reviews_df['피부트러블'].mode().values[0]
                    user_info = f"\n이 제품은 {age}, {types}이면서 {trouble} 고민을 가진 사람들이 많이 쓰는 제품입니다."
                    await message.channel.send(user_info)

                    # 비슷한 상품을 추천할지 물어보는 부분
                    while True:
                        await message.channel.send("비슷한 상품을 보여드릴까요? (Y/N)")
                        
                        try:
                            show_similar_msg = await self.wait_for('message', timeout=60.0, check=check)
                            user_response = show_similar_msg.content.lower()
                        except asyncio.TimeoutError:
                            user_response = 'n' 
                    
                        if user_response == 'y':
                            input_keywords_str = new_df[new_df['상품명'] == selected_title]['키워드'].iloc[0]
                            if pd.isna(input_keywords_str):  # 이 부분을 추가
                                await message.channel.send("해당 상품의 키워드 정보가 없습니다.")
                                break
                            input_keywords = input_keywords_str.split(", ")
                            
                            similar_products = {}
                            for _, row in new_df.iterrows():
                                product = row['상품명']
                                keywords_str = row['키워드']
                                if product == selected_title or pd.isna(keywords_str):
                                    continue
                                keywords = keywords_str.split(", ")
                                matching_keywords = set(input_keywords).intersection(set(keywords))
                                if matching_keywords:
                                    similar_products[product] = [keyword for keyword in matching_keywords]
                            
                            sorted_similar_products = sorted(similar_products.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                            
                            similar_response = "비슷한 제품:\n"
                            for product, matching_keywords in sorted_similar_products:
                                matching_keywords_str = ', '.join(matching_keywords)
                                similar_response += f"{product} (일치하는 키워드: {matching_keywords_str})\n"
                            
                            await message.channel.send(similar_response)
                            return
                        
                        elif user_response == 'n':
                            await message.channel.send("비슷한 상품 추천을 건너뜁니다.")
                            return
                        else:
                            await message.channel.send("올바르지 않은 입력입니다. 다시 입력해주세요.")
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except asyncio.TimeoutError:
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.chatbot(message, new_title)
        
        finally:
            if send_intro:
                await self.send_introduction_message(message.channel)



    async def keyword_bot(self, message, title):
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in new_df['상품명'].unique().tolist() if condition(t)]
        
        if len(matching_titles) == 0:
            await message.channel.send("해당 상품이 없습니다. 상품명을 똑바로 입력해 주세요.")
            return
        elif len(title) == 1:
            await message.channel.send("최소 두 글자 이상의 쿼리를 입력해주세요.")
            return

        response = "일치하는 상품명 목록:\n"
        for idx, matching_title in enumerate(matching_titles, start=1):
            response += f"{idx}. {matching_title}\n"

        max_len = 2000 
        if len(response) > max_len:
            for i in range(0, len(response), max_len):
                await message.channel.send(response[i:i + max_len])
        else:
            await message.channel.send(response)

        def check(m):
            return m.author == message.author and m.channel == message.channel

        try:
            while True:
                msg = await self.wait_for('message', timeout=30.0, check=check)
                if msg.content in ['q', '!quit', 'quit']:  
                    await msg.channel.send('대화를 종료합니다.')
                    await self.close()  
                    return
            
                try:
                    selected_idx = int(msg.content) - 1
                except ValueError:
                    await message.channel.send("숫자만 입력해 주세요.")
                    continue

            
                if 0 <= selected_idx < len(matching_titles):
                    selected_title = matching_titles[selected_idx]
                    await self.keyword(message, selected_title)

                    break
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except asyncio.TimeoutError:
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.keyword_bot(message, new_title) 
        
        finally:
            await self.send_introduction_message(message.channel)


    async def keyword(self, message, title):
        try:
            matching_keywords_str = new_df[new_df['상품명'] == title]['키워드'].iloc[0]
            if pd.isna(matching_keywords_str):  # Check if it's NaN
                await message.channel.send("해당 상품의 키워드 정보가 없습니다.")
                return
        except IndexError:
            await message.channel.send("해당 상품의 키워드 정보가 없습니다.")
            return

        matching_keywords = matching_keywords_str.split(', ')
        response = ""
        for keyword in matching_keywords:
            response += f"이 제품의 키워드 추출 결과는 {keyword} 입니다.\n"
            
        await message.channel.send(response)

    async def wait_for_user_response(self, message):
        def check(m):
            return m.author == message.author and m.channel == message.channel

        try:
            response = await self.wait_for('message', timeout=30.0, check=check)
            return response.content
        except asyncio.TimeoutError:
            await message.channel.send("응답 시간이 초과되었습니다. 처음부터 다시 시도해주세요.\n - `!review`: 상품 리뷰 요약\n- `!keyword`: 키워드 추출\n- `!recommand`: 추천\n종료하려면 'q' 또는 'quit' 또는 '!quit'를 입력하세요.")
            return None

    async def recommend_chatbot(self, skincare, new_df, message):       
        while True:
            await message.channel.send("\n1. 제품 추천 받기\n2. 상품 리뷰 확인하기\n3. 종료")
            response = await self.wait_for_user_response(message)
            if response in ['q', '!quit', 'quit']:
                await message.channel.send("대화를 종료합니다.")
                return

            if response == '1':
                # 추천 상품 리스트 받기
                await self.recommend_products_for_user(skincare, message)
            elif response == '2':
                first_time = True  # 첫 번째 입력인지를 판별하기 위한 변수
                while True:
                    if first_time:
                        await message.channel.send("상품명을 입력해 주세요.")
                        first_time = False
                    title = await self.wait_for_user_response(message)
                    
                    # 종료 명령어 확인
                    if not title:  # response가 None인 경우 (종료 명령어를 입력한 경우)
                        return
                    
                    # 최소 두 글자 이상의 쿼리를 입력받았는지 확인
                    if len(title) < 2:
                        await message.channel.send("최소 두 글자 이상의 쿼리를 입력해주세요.")
                        continue
                    
                    # 해당 상품명이 있는지 확인
                    keywords = title.split()
                    condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
                    matching_titles = [t for t in skincare['상품명'].unique().tolist() if condition(t)]
                    
                    if len(matching_titles) == 0:
                        await message.channel.send("해당 상품이 없습니다. 상품명을 똑바로 입력해 주세요.")
                        continue
                    
                    # 검증이 완료되면 chatbot 함수를 호출합니다.
                    await self.chatbot(message, title, send_intro=False)
                    break

            elif response == '3':
                await message.channel.send("추천 프로그램을 종료합니다. 완전히 종료하려면 q를 한번 더 입력해서 완전히 종료하세요. 아니라면 다시 아래 명령어들을 보고 명령어를 입력하세요.\n - `!review`: 상품 리뷰 요약\n- `!keyword`: 키워드 추출\n- `!recommand`: 추천")
                return
            else:
                await message.channel.send("올바른 선택지를 입력하세요.")





            
    async def send_introduction_message(self, channel):
        introduction_message = (
            "저는 여러분을 도와드릴 챗봇입니다.\n"
            "저의 기능은 다음과 같습니다:\n"
            "- `!review`: 상품 리뷰 요약\n"
            "- `!keyword`: 키워드 추출\n"
            "- `!recommand`: 추천\n"
            "종료하려면 'q' 또는 'quit' 또는 '!quit'를 입력하세요."
        )
        await channel.send(introduction_message)
 

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)

