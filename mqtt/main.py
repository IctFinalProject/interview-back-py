from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from contextlib import asynccontextmanager
from database import model as db
import json
import uvicorn

from typing import List

# 데이타베이스 연결
conn = db.connectDb() # 모델 호출

@asynccontextmanager
async def lifespan(app: FastAPI):
    start() # 서버 시작
    yield
    shutdown() # 서버 종료

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 리액트 앱의 도메인을 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

mqtt_broker = "localhost"   # 브로커 ip주소
mqtt_port = 1883    # 브로커 포트번호, 1883으로 이용할것
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")

mqtt_client.on_connect = on_connect
mqtt_client.connect(mqtt_broker, mqtt_port, 60) #mqtt에 연결
mqtt_client.loop_start()



def start():
    print("service is started.")
    # mqtt_client.subscribe("mqtt/chat/#") # chat 하위루트는 모두 구독

def shutdown():
    # print("service is stopped.")
    mqtt_client.loop_stop()  #시작된 루프 끝내기, mqtt에서 연결끊기
    mqtt_client.disconnect()


clients = {}  # 연결된 클라이언트들(토픽) 관리 차원

class ChatMessage(BaseModel):
    text: str
    chatroomId: int
    sender: str
    senderId: str
    timestamp: str
    type: str
    userIds: List[int]

class CommentMessage(BaseModel):
    text: str
    bbsId: str
    bbsTitle: str
    sender: str
    senderId: str
    receiverId: str
    contentId: str
    timestamp: str
    type: str

class ChatAlarm(BaseModel):
    type: str
    userId: int # receiverId
    chatroomId: int

class BbsAlarm(BaseModel):
    id: int

@app.post("/sendMessage")
async def send_message(message: ChatMessage):
    message_id = db.getMaxMessageId(conn, 'chat_messages') + 1
    alarm_id = db.getMaxMessageId(conn, 'alarm') + 1

    # MQTT로 메시지 publish
    mqtt_topic = f"mqtt/chat/{message.chatroomId}"
    mqtt_message = json.dumps({**message.dict(), 'messageId': message_id, 'alarmId': alarm_id})

    # publish
    mqtt_client.publish(mqtt_topic, mqtt_message)

    for user_id in message.userIds:
        mqtt_alarm_topic = f"mqtt/member/{user_id}"  # 각 user_id에 대해 토픽 생성
        mqtt_client.publish(mqtt_alarm_topic, mqtt_message)  # 해당 토픽으로 메시지 발행

        alarm_isinserted = db.saveAlarm(conn, {
            'id': db.getMaxMessageId(conn, 'alarm') + 1,
            'sender': message.sender,
            'senderId': message.senderId,
            'chatroomId': message.chatroomId,
            'text': message.text,
            'title': 'id값으로 제목 갖고오기',
            'type': message.type,
            'contentId': message_id,
            'receiverId': user_id
        })

    # DB에 메시지 저장
    isinserted = db.saveMessage(conn, {
        'id': message_id,
        'topic': mqtt_topic,
        'text': message.text,
        'senderId': message.senderId
    })

    isupdated = db.updateLastMessage(conn, {
        'topic': mqtt_topic,
        'sender': message.sender,
        'text': message.text
    })

    return {"status": "Message sent", "db_inserted": isinserted}

@app.post("/sendComment")
async def send_comment(message: CommentMessage):

    print('sendCommnet에서 message 출력: ', message)

    alarm_id = db.getMaxMessageId(conn, 'alarm') + 1

    # MQTT로 메시지 publish
    mqtt_topic = f"mqtt/member/{message.receiverId}"
    # mqtt_message = json.dumps(message.dict())
    mqtt_message = json.dumps({**message.dict(), 'alarmId': alarm_id})

    print('sendCommnet에서 mqtt_message 출력: ', mqtt_message)

    # publish
    mqtt_client.publish(mqtt_topic, mqtt_message)

    if(message.type == 'bbs'):
        alarm_isinserted = db.saveCommentAlarm(conn, {
            'id': db.getMaxMessageId(conn, 'alarm') + 1,
            'bbsId': message.bbsId,
            'title': message.bbsTitle,
            'text': message.text,
            'sender': message.sender,
            'senderId': message.senderId,
            'receiverId': message.receiverId,
            'contentId': message.contentId,
            'type': message.type
        })

    return {"status": "Comment Alarm sent"}


# chatroomId랑 receiverId 겹치는 거 isRead = 1로 변경 후 mqtt/chat/{alarm.userId} 로 메세지 보내기 (걍 뭐라도)
@app.post("/readChatAlarm")
async def read_chat_message(alarm: ChatAlarm):
    topic = f"mqtt/chat/{alarm.userId}"


@app.post("/readBbsAlarm")
async def read_bbs_alarm(alarm: BbsAlarm):
    db.readCommentAlarm(conn, {
        'id': alarm.id
    })
    return 'test'



### 기존에 subscribe를 한 후 받은 메세지를 db에 저장
# def on_message(client, userdata, msg):
#     topic = msg.topic
#     message = json.loads(msg.payload.decode())
#     # recipient = message.get("recipient")
#     # print(f"Received message on topic {msg.topic}: {message}")
#     # isinserted = db.saveMessage(conn,{'content':message['content']})
#     # print('타입 확인 type(topic): ', type(topic))
#     isinserted = db.saveMessage(conn, {'topic':topic,
#                                        'text':message['text'],
#                                        'senderId':message['senderId']})
#
#     isupdated = db.updateLastMessage(conn, {'topic':topic, 'sender':message['sender'], 'text':message['text']})
#
#     # print(f"{message['content']}, {isinserted}")
#     print(f"Topic '{msg.topic}'에서 메시지를 받았습니다: {msg.payload.decode()}")
#
#     # if recipient in clients:
#     #     clients[recipient].append(message)
#
#
# mqtt_client.on_message = on_message

if __name__ == "__main__":
    # uvicorn.run(app, host="localhost", port=8000) # fast-api 서버를 돌릴 포트 연결
    uvicorn.run(app, host="0.0.0.0", port=8000)  # fast-api 서버를 돌릴 포트 연결