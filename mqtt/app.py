from fastapi import FastAPI, WebSocket
from paho.mqtt.client import Client
import json
from contextlib import asynccontextmanager

app = FastAPI()

mqtt_client = Client()

# MQTT 브로커에 연결 (로컬 호스트)
try:
    mqtt_client.connect("localhost", 1883, 60)
except ConnectionRefusedError as e:
    print('서버에 연결할 수 없습니다.')
    print('1. 서버의 ip주소와 포트번호가 올바른지 확인하십시오.')
    print('2. 서버 실행 여부를 확인하십시오.')
    print('에러: ', e)

# MQTT 메시지 수신 콜백 함수
def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    data = json.loads(payload)
    print(f"Received message: {data}")
    for connection in websocket_connections:
        connection.send_text(json.dumps(data))

mqtt_client.on_message = on_message
mqtt_client.subscribe("chat/messages")

websocket_connections = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def startup_event():
        mqtt_client.loop_start()
        print("MQTT client started")
    yield

    async def shutdown_event():
        mqtt_client.loop_stop()
        print("MQTT client stopped")

@app.websocket("/mqtt")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    websocket_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"Received WebSocket message: {message}")
            mqtt_client.publish("chat/messages", json.dumps(message))
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        websocket_connections.remove(websocket)
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)