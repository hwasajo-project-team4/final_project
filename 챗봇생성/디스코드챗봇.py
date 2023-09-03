
import discord
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Okt
from collections import Counter
import asyncio
import pandas as pd
import time
TOKEN = '본인의 디스코드 챗봇 토큰'
CHANNEL_ID = '본인의 서버의 채널 아이디'


def load_dataframe():
    return pd.read_csv("아모레크롤링_스킨케어_완료.csv")

model_name = "noahkim/KoT5_news_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
okt = Okt()


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
        df = load_dataframe()
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
        df = load_dataframe()
        
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

                    self.keyword_mode_per_user[message.author.id] = False
                    
                    await message.channel.send(f"{selected_title}의 전체 리뷰 요약: {summary}")
                    time.sleep(2)
                    await self.send_introduction_message(message.channel)
                    time.sleep(2)
                    break
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except:
            self.timeout_mode_per_user[message.author.id] = True
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.chatbot(message, new_title)
        
        finally:
            self.keyword_mode_per_user[message.author.id] = False
            self.timeout_mode_per_user[message.author.id] = False 



    async def keyword_bot(self, message, title):
        self.keyword_mode_per_user[message.author.id] = True
        df = load_dataframe()
        
        
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
                    self.keyword_mode_per_user[message.author.id] = False
                    await self.keyword(message, selected_title)
                    time.sleep(2)
                    await self.send_introduction_message(message.channel)
                    time.sleep(2)
                    break
                else:
                    await message.channel.send(f"입력하신 숫자({selected_idx + 1}) 번째의 제품은 없습니다.")

        except:
            self.timeout_mode_per_user[message.author.id] = True
            await message.channel.send('시간이 초과되었습니다. 상품명을 다시 입력해 주세요.')

            new_title = await self.get_product_title(message) 
            if new_title:
                await self.keyword_bot(message, new_title) 
        
        finally:
            self.keyword_mode_per_user[message.author.id] = False
            self.timeout_mode_per_user[message.author.id] = False 


    async def keyword(self, message, title):
        df = load_dataframe()
        texts = df[df['상품명'] == title]['리뷰'].to_list()
        nouns_list = []
        
        for text in texts:
            nouns = okt.nouns(text)
            nouns_list.extend(nouns)

        count = Counter(nouns_list)
        noun_list = count.most_common(100)

        response = ""
        for v in noun_list:
            if any(keyword in v[0] for keyword in ['지성','수부지', '건성', '복합성','여드름','잡티','피부톤','탄력','피지','모공','블랙헤드','민감성','봄웜','웜톤','봄웜톤','봄라이트','봄라','봄브라이트','여쿨','여름쿨톤','여라','여름라이트','여름뮤트','여뮽','뮤트','뮤트톤','여름브라이트','가을웜톤','가을뮤트','갈딥','가을딥','갈웜','겨쿨','겨울브라이트','겨울딥','쿨톤']):
                response += f"이 제품은 {v[0]} 타입에게 추천하는 제품입니다.\n"
        
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
