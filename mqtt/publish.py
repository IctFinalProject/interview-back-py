import random
import time
from paho.mqtt import client as mqtt_client
from datetime import datetime
import json

# broker = 'broker.emqx.io'
broker = '192.168.0.137'
port = 1883
topic = "python/mqtt"
client_id = f'publish-{random.randint(0, 1000)}'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("MQTT 브로커에 연결되었습니다!")
        else:
            print(f"연결 실패, 반환 코드: {rc}")

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, message):
    result, _ = client.publish(topic, message)
    if result == mqtt_client.MQTT_ERR_SUCCESS:
        print(f"메시지를 Topic '{topic}'로 전송했습니다:")
    else:
        print(f"메시지 전송 실패: {topic}")

def run():
    client = connect_mqtt()
    client.loop_start()
    time.sleep(1)
    while True:
        user_input = input("전송할 메시지를 입력하세요 : ")
        if user_input == 'q':
            break
        # publish(client, user_input)
        publish(client, json.dumps({'text': user_input, 'sender': 'inPython', 'timestamp': datetime.now().isoformat()}))
    client.loop_stop()

if __name__ == '__main__':
    run()