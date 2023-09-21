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


TOKEN = '본인의 디스코드 챗봇 토큰'
CHANNEL_ID = '본인의 서버의 채널 아이디'


mecab = Mecab()
# 파일 경로
files = [
    f"뷰티푸드 별점.csv",
    f"소품도구 별점.csv",
    f"아모레크롤링_스킨케어_완료.csv",
    f"아모레크롤링_메이크업_완료.csv",
    f"향수 별점.csv"
]

df_list = []


for file in files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)


df = pd.concat(df_list, ignore_index=True)


new_df = pd.read_csv("키워드추출.csv")



model_name = "noahkim/KoT5_news_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        await self.change_presence(status=discord.Status.online, activity=discord.Game("대기중"))

        channel = self.get_channel(int(CHANNEL_ID))
        if channel:
            introduction_message = (
                "환영합니다! 저는 여러분을 도와드릴 챗봇입니다.\n"
                "저의 기능은 다음과 같습니다:\n"
                "- `!review`: 상품 리뷰 요약\n"
                "- `!keyword`: 키워드 추출\n"
                "- `ping`: 봇 상태 확인\n"
                "종료하려면 'q' 또는 'quit' 또는 '!quit'를 입력하세요."
            )
            await channel.send(introduction_message)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword_mode_per_user = {}
        self.timeout_mode_per_user = {}

    def find_matching_titles(self, title):
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in df['상품명'].unique().tolist() if condition(t)]
        return matching_titles
    
    async def get_product_title(self, message):
        while True:
            try:
                msg = await self.wait_for('message', timeout=30.0, check=lambda m: m.author == message.author and m.channel == message.channel)
                
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

        if self.keyword_mode_per_user.get(message.author.id, False):
            return

        if message.content.startswith('!review') or message.content.startswith('!keyword'):
            self.keyword_mode_per_user[message.author.id] = True

        
        if self.timeout_mode_per_user.get(message.author.id, False):
            return

        if message.content.startswith('!review'):
            await message.channel.send("상품 리뷰 요약을 선택하셨습니다. 상품명을 입력해 주세요.")
            title = await self.get_product_title(message)
            if title:
                await self.chatbot(message, title)
            self.keyword_mode_per_user[message.author.id] = False
            return


        elif message.content.startswith('!keyword'):
            await message.channel.send("키워드 추출을 선택하셨습니다. 상품명을 입력해 주세요.")
            title = await self.get_product_title(message)
            if title:
                await self.keyword_bot(message, title)
            self.keyword_mode_per_user[message.author.id] = False
            return


        elif message.content in ['q', '!quit', 'quit']:
            await message.channel.send('대화를 종료합니다.')
            await self.close()
            return
        
        elif message.content == 'ping':
            await message.channel.send('pong {0.author.mention}'.format(message))
        else:
            answer = self.get_answer(message.content)
            await message.channel.send(answer)

    def get_day_of_week(self):
        weekday_list = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
 
        weekday = weekday_list[datetime.today().weekday()]
        date = datetime.today().strftime("%Y년 %m월 %d일")
        result = '{}({})'.format(date, weekday)
        return result
 
    def get_time(self):
        return datetime.today().strftime("%H시 %M분 %S초")
 
    def get_answer(self, text):
        trim_text = text.replace(" ", "")
 
        answer_dict = {
            '안녕': '안녕하세요. MyBot입니다.',
            '요일': ':calendar: 오늘은 {}입니다'.format(self.get_day_of_week()),
            '시간': ':clock9: 현재 시간은 {}입니다.'.format(self.get_time()),
        }
 
        if trim_text == '' or trim_text is None:
            return "알 수 없는 질의입니다. 답변을 드릴 수 없습니다."
        elif trim_text in answer_dict.keys():
            return answer_dict[trim_text]
        else:
            for key in answer_dict.keys():
                if key.find(trim_text) != -1:
                    return "연관 단어 [" + key + "]에 대한 답변입니다.\n" + answer_dict[key]
 
            for key in answer_dict.keys():
                if answer_dict[key].find(text[1:]) != -1:
                    return "질문과 가장 유사한 질문 [" + key + "]에 대한 답변이에요.\n" + answer_dict[key]
 
        return text + "은(는) 없는 질문입니다."
    
    async def chatbot(self, message, title):
        self.keyword_mode_per_user[message.author.id] = True
        
        keywords = title.split()
        condition = lambda x: all(keyword.lower() in x.lower() for keyword in keywords)
        matching_titles = [t for t in df['상품명'].unique().tolist() if condition(t)]
        
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
            
                    reviews = df[df['상품명'] == selected_title]['리뷰'].to_list()
                    reviews_text = ' '.join(reviews)
                    
                    inputs = tokenizer(reviews_text, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = model.generate(inputs["input_ids"], num_beams=5, min_length=30, max_length=300, early_stopping=True)
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                    # self.keyword_mode_per_user[message.author.id] = False
                    
                    await message.channel.send(f"{selected_title}의 전체 리뷰 요약: {summary}")

                        # 비슷한 상품을 추천할지 물어보는 부분
                    await message.channel.send("비슷한 상품을 보여드릴까요? (Y/N)")
                    
                    try:
                        show_similar_msg = await self.wait_for('message', timeout=30.0, check=check)
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
                    elif user_response == 'n':
                        await message.channel.send("비슷한 상품 추천을 건너뜁니다.")
                    else:
                        await message.channel.send("올바르지 않은 입력입니다. 비슷한 상품 추천을 건너뜁니다.")
                    time.sleep(2)
                    await self.send_introduction_message(message.channel)
                    time.sleep(2)
                    break
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except asyncio.TimeoutError:
            self.timeout_mode_per_user[message.author.id] = True
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.chatbot(message, new_title)
        
        finally:
            self.keyword_mode_per_user[message.author.id] = False
            self.timeout_mode_per_user[message.author.id] = False
            await self.send_introduction_message(message.channel)



    async def keyword_bot(self, message, title):
        self.keyword_mode_per_user[message.author.id] = True
        
        
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
                    self.keyword_mode_per_user[message.author.id] = False
                    await self.keyword(message, selected_title)
                    time.sleep(2)
                    await self.send_introduction_message(message.channel)
                    time.sleep(2)
                    break
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except asyncio.TimeoutError:
            self.timeout_mode_per_user[message.author.id] = True
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.keyword_bot(message, new_title) 
        
        finally:
            self.keyword_mode_per_user[message.author.id] = False
            self.timeout_mode_per_user[message.author.id] = False 


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
    
    async def send_introduction_message(self, channel):
        introduction_message = (
            "저는 여러분을 도와드릴 챗봇입니다.\n"
            "저의 기능은 다음과 같습니다:\n"
            "- `!review`: 상품 리뷰 요약\n"
            "- `!keyword`: 키워드 추출\n"
            "- `ping`: 봇 상태 확인\n"
            "종료하려면 'q' 또는 'quit' 또는 '!quit'를 입력하세요."
        )
        await channel.send(introduction_message)
 

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)
