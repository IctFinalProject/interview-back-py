import random
from paho.mqtt import client as mqtt_client

# broker = 'broker.emqx.io'
broker = '192.168.0.137'
port = 1883
topic = "mqtt/chat/#"
client_id = f'subscribe-{random.randint(0, 100)}'


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("MQTT 브로커에 연결되었습니다!")  # MQTT Broker에 성공적으로 연결되었을 때 출력
        else:
            print(f"연결 실패, 반환 코드: {rc}")  # 연결 실패 시 반환 코드 출력

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def on_message(client, userdata, msg):
    print(f"Topic '{msg.topic}'에서 메시지를 받았습니다: {msg.payload.decode()}")


def run():
    client = connect_mqtt()
    client.on_message = on_message
    client.subscribe(topic)
    client.loop_forever()


if __name__ == '__main__':
    run()