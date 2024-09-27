import os
from configparser import ConfigParser
from cx_Oracle import connect
import datetime



def connectDb():#데이타베이스 연결
    config = ConfigParser()
    # 현재 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # database.ini 파일의 절대 경로 생성
    config_path = os.path.join(current_dir, 'database.ini')
    config.read(config_path,encoding='utf8')
    print(config.sections())
    # 데이타베이스 연결
    return connect(user=config['ORACLE']['user'],password=config['ORACLE']['password'],dsn=config['ORACLE']['url'],encoding="UTF-8")

def close(conn):#커넥션 닫기
    if conn:conn.close()

def saveMessage(conn,dict_): #입력 처리:성공시 1,실패시 0
    with conn.cursor() as cursor:
        try:
            # cursor.execute('INSERT INTO chat_messages (is_active,chatroom_id,message_id,message_send_date,user_id,message_content) VALUES(DEFAULT,:topic,seq_message.nextval,SYSDATE,1,:content)',dict_)
            topic_str = dict_['topic']
            dict_['topic'] = int(topic_str.split('/')[-1])


            cursor.execute(
                'INSERT INTO chat_messages(id, chatroom_id, created_time, message, users_id, deleted_time, is_deleted) VALUES(:id, :topic, SYSDATE, :text, :senderId, null, 0)',
                dict_)

            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print('입력시 오류:',e,sep='')
            return 0

def updateLastMessage(conn,dict_): #입력 처리:성공시 1,실패시 0
    with conn.cursor() as cursor:
        current_time = datetime.datetime.now()
        try:
            topic_str = dict_['topic']
            dict_['topic'] = int(topic_str.split('/')[-1])

            last_message = f"{dict_['sender']} : {dict_['text']}"
            # print(last_message)
            dict_['last_message'] = last_message

            # cursor.execute(
            #     'UPDATE chatroom SET last_message = :last_message WHERE id = :topic',
            #     dict_)
            cursor.execute(
                'UPDATE chatroom SET last_message = :last_message, updated_time = :updated_time WHERE id = :topic',
                {'last_message': dict_['last_message'], 'topic': dict_['topic'], 'updated_time': current_time}
            )

            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print('last_message 업데이트 오류:', e, sep='')
            return 0

def getMaxMessageId(conn, table_name): # 가장 큰 ID 값을 조회하는 함수
    with conn.cursor() as cursor:
        try:
            cursor.execute(f'SELECT MAX(id) FROM {table_name}')
            result = cursor.fetchone()
            max_id = result[0] if result[0] is not None else 0
            return max_id
        except Exception as e:
            print('ID 조회 오류:', e, sep='')
            return 0

def saveAlarm(conn,dict_): # 입력 처리:성공시 1,실패시 0
    with conn.cursor() as cursor:
        try:
            print('dict_: ', dict_)

            chatroom_title = getChatroomTitle(conn, dict_['chatroomId'])
            content = f"{dict_['sender']} : {dict_['text']}"

            cursor.execute(
                '''
                INSERT INTO alarm (
                    id, type, title, content, sender_id, receiver_id, chatroom_id, content_id, is_read, is_disabled, created_time
                ) VALUES (
                    :id, :type, :title, :content, :senderId, :receiverId, :chatroomId, :contentId, :isRead, :isDisabled, SYSDATE
                )
                ''',
                {
                    'id': dict_['id'],  # 알람 ID
                    'type': dict_['type'],  # 알람 타입 (예: 'chat')
                    'title': chatroom_title,  # chatroom_title을 title로 사용
                    'content': content,  # 알람 내용
                    'senderId': dict_['senderId'],  # 보낸 사람 ID
                    'receiverId': dict_['receiverId'],  # 받는 사람 ID
                    'chatroomId': dict_['chatroomId'],  # 채팅방 ID
                    'contentId': dict_['contentId'],  # 메시지 또는 게시물 ID
                    'isRead': 0,  # 읽지 않은 상태
                    'isDisabled': 0  # 비활성화되지 않은 상태
                }
            )

            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print('알람 저장 시 오류:', e, sep='')
            return 0

def saveCommentAlarm(conn,dict_): # 입력 처리:성공시 1,실패시 0
    with conn.cursor() as cursor:
        try:
            print('dict_: ', dict_)
            content = f"{dict_['sender']} : {dict_['text']}"

            cursor.execute(
                '''
                INSERT INTO alarm (
                    id, type, title, content, sender_id, receiver_id, bbs_id, content_id, is_read, is_disabled, created_time
                ) VALUES (
                    :id, :type, :title, :content, :senderId, :receiverId, :bbsId, :contentId, :isRead, :isDisabled, SYSDATE
                )
                ''',
                {
                    'id': dict_['id'],  # 알람 ID
                    'bbsId': dict_['bbsId'],
                    'type': dict_['type'],  # 알람 타입 (예: 'chat')
                    'title': dict_['title'],  # chatroom_title을 title로 사용
                    'content': content,  # 알람 내용
                    'senderId': dict_['senderId'],  # 보낸 사람 ID
                    'receiverId': dict_['receiverId'],  # 받는 사람 ID
                    'contentId': dict_['contentId'],  # 메시지 또는 게시물 ID
                    'isRead': 0,  # 읽지 않은 상태
                    'isDisabled': 0  # 비활성화되지 않은 상태
                }
            )

            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print('알람 저장 시 오류:', e, sep='')
            return 0

def readCommentAlarm(conn,dict_):
    with conn.cursor() as cursor:
        try:

            cursor.execute(
                'UPDATE alarm SET is_read = 1 WHERE id = :id',
                {'id': dict_['id']}
            )

            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print('bbs Alarm isRead 업데이트 오류:', e, sep='')
            return 0

def getChatroomTitle(conn, chatroom_id):  # chatroomId로 chatroom_title 조회하는 함수
    with conn.cursor() as cursor:
        try:
            cursor.execute('SELECT chatroom_title FROM chatroom WHERE id = :chatroom_id', {'chatroom_id': chatroom_id})
            result = cursor.fetchone()

            return result[0]

        except Exception as e:
            print('Chatroom title 조회 오류:', e, sep='')
            return None