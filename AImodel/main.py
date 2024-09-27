# The-Interview-Buster-main\main.py

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cx_Oracle
from datetime import datetime
import os
import cv2
import subprocess
import time
from tqdm import tqdm
from google.cloud import storage
from google.oauth2 import service_account
import requests
import moviepy.editor as mp
import anthropic
import json
from collections import Counter
from konlpy.tag import Okt
import librosa
import numpy as np
from typing import Dict, Any
import logging
import tempfile

# 필요한 추가 임포트
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Dict

# 로컬 모듈 임포트
from models.head_eye import head_eye

client = anthropic.Anthropic(api_key="YOUR-API-KEY")

# Oracle 데이터베이스 연결 설정
dsn = cx_Oracle.makedsn("192.168.0.100", 1521, service_name="XEPDB1")
pool = cx_Oracle.SessionPool(user="VIP", password="VIP", dsn=dsn, min=2, max=5, increment=1, encoding="UTF-8")

# Clova Speech API 설정
CLOVA_API_URL = 'https://naveropenapi.apigw.ntruss.com/recog/v1/stt'
CLOVA_CLIENT_ID = 'YOUR_CLIENT_ID'
CLOVA_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'

# 감정 분석 및 요약 API URL 설정
SENTIMENT_API_URL = 'https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze'
SUMMARY_API_URL = 'https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize'

# Spring Boot 백엔드 URL 설정
SPRING_BACKEND_URL = "http://localhost:8080/api/video-analysis/result"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalysisRequest(BaseModel):
    video_id: int
app = FastAPI()

# 스레드 풀 및 작업 큐 설정
executor = ThreadPoolExecutor(max_workers=5)  # 동시 작업 수 증가
task_queue: asyncio.Queue = asyncio.Queue(maxsize=10)  # 큐 크기 제한
task_statuses: Dict[int, Dict[str, Any]] = {}

# 큐 처리를 위한 비동기 함수
async def process_queue():
    while True:
        video_id = await task_queue.get()
        try:
            task_statuses[video_id] = {"status": "processing", "progress": 0}
            await background_video_analysis(video_id)
            task_statuses[video_id] = {"status": "completed", "progress": 100}
        except Exception as e:
            task_statuses[video_id] = {"status": "failed", "error": str(e)}
        finally:
            task_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.post("/analyze-video/")
async def request_video_analysis(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    if task_queue.full():
        raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")
    
    await task_queue.put(request.video_id)
    task_statuses[request.video_id] = {"status": "queued", "progress": 0}
    return {"message": "Video analysis queued", "video_id": request.video_id}

@app.get("/job-status/{video_id}")
async def get_job_status(video_id: int):
    if video_id in task_statuses:
        return task_statuses[video_id]
    else:
        return {"status": "not found"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    오디오 파일을 업로드하고 분석하는 엔드포인트
    """
    file_path = f'./uploads/{file.filename}'
    with open(file_path, 'wb') as f:
        f.write(await file.read())
    
    transcription_result = transcribe_with_clova(file_path)
    
    if transcription_result:
        text = transcription_result
        sentiment_result = analyze_sentiment(text)
        summary_result = summarize_text(text)
        
        return {
            "transcription": text,
            "sentiment_analysis": sentiment_result,
            "text_summary": summary_result
        }
    else:
        return {"error": "Transcription failed"}

@app.on_event("shutdown")
def shutdown_event():
    """
    애플리케이션 종료 시 Oracle 커넥션 풀을 종료하는 함수
    """
    pool.close()

def start_video_analysis(video_id: int, background_tasks: BackgroundTasks):
    """
    비디오 분석을 시작하는 함수
    """
    video_info = get_video_info(video_id)
    if not video_info:
        return JSONResponse(
            status_code=404,
            content={"error": "Video not found"}
        )
    
    background_tasks.add_task(background_video_analysis, video_id)
    return {
        "message": "Video analysis started",
        "video_id": video_id
    }

async def background_video_analysis(video_id: int):
    start_time = time.time()
    try:
        logger.info(f"Starting background video analysis for video_id: {video_id}")
        task_statuses[video_id] = {"status": "in_progress", "progress": 0}
        
        # 비디오 정보 가져오기
        video_info = await asyncio.to_thread(get_video_info, video_id)
        if not video_info:
            logger.error(f"Video info not found for video_id: {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")
        
        task_statuses[video_id]["progress"] = 10
        
        # 비디오 다운로드
        logger.info(f"Downloading video for video_id: {video_id}")
        local_video_path = await asyncio.to_thread(download_video_from_gcs, video_info['file_path'], '/tmp')
        logger.info(f"Video downloaded to: {local_video_path}")
        
        task_statuses[video_id]["progress"] = 30
        
        # 비디오 길이 계산
        logger.info(f"Calculating video duration for video_id: {video_id}")
        answer_duration = await asyncio.to_thread(calculate_answer_duration, local_video_path)
        logger.info(f"Video duration: {answer_duration} seconds")
        
        # 데이터베이스 업데이트
        logger.info(f"Updating video duration in database for video_id: {video_id}")
        await update_video_duration(video_id, answer_duration)
        
        task_statuses[video_id]["progress"] = 50
        
        # 회사명과 질문 텍스트 가져오기
        logger.info(f"Getting company name and question text for video_id: {video_id}")
        company_name = await get_company_name(video_id)
        question_text = await asyncio.to_thread(get_question_text, video_id)
        
        # 비디오 분석
        logger.info(f"Starting video analysis for video_id: {video_id}")
        results, speech_results = await asyncio.to_thread(analyze_video, local_video_path)
        logger.info(f"Video analysis completed for video_id: {video_id}")
        
        task_statuses[video_id]["progress"] = 70
        
        # 분석 결과 저장
        if results:
            logger.info(f"Saving analysis results for video_id: {video_id}")
            await asyncio.to_thread(save_analysis_result, video_id, results)
        
        if speech_results:
            logger.info(f"Saving speech analysis results for video_id: {video_id}")
            await asyncio.to_thread(save_speech_analysis_result, video_id, speech_results)
        
        task_statuses[video_id]["progress"] = 80
        
        # Claude 분석
        claude_analysis = None
        if "transcription" in speech_results:
            logger.info(f"Starting Claude analysis for video_id: {video_id}")
            claude_analysis = await analyze_interview_with_claude(
                speech_results["transcription"],
                question_text,
                company_name,
                answer_duration
            )
            logger.info(f"Claude analysis completed for video_id: {video_id}")
            await save_claude_analysis_result(video_id, claude_analysis)
        
        task_statuses[video_id]["progress"] = 90
        
        # 통합된 분석 결과
        integrated_results = integrate_analysis_data(results, speech_results, claude_analysis)
        
        # 분석 완료 메시지 전송
        success = await asyncio.to_thread(send_analysis_complete_to_spring, video_id, integrated_results)
        if success:
            logger.info(f"Analysis complete message sent successfully for video_id: {video_id}")
        else:
            logger.error(f"Failed to send analysis complete message for video_id: {video_id}")
        
        logger.info(f"Analysis completed for video_id: {video_id}")
        
    except Exception as e:
        logger.error(f"Error in background_video_analysis for video_id {video_id}: {str(e)}", exc_info=True)
        # 에러 발생 시에도 Spring 백엔드에 알림
        error_message = {
            "videoId": video_id,
            "message": f"분석 중 오류 발생: {str(e)}"
        }
        await asyncio.to_thread(send_analysis_complete_to_spring, video_id, error_message)
    finally:
        # 분석 상태 업데이트
        task_statuses[video_id] = {"status": "completed", "progress": 100}
        
        # 분석 시간 계산 및 출력
        end_time = time.time()
        analysis_duration = end_time - start_time
        logger.info(f"Total analysis time for video_id {video_id}: {analysis_duration:.2f} seconds")

async def get_company_name(video_id: int) -> str:
    def _get_company_name(video_id: int) -> str:
        with pool.acquire() as connection:
            cursor = connection.cursor()
            try:
                # videos, resume, users 테이블을 LEFT JOIN하여 정보를 가져옵니다.
                cursor.execute("""
                    SELECT r.DESIRED_COMPANY, u.USERNAME
                    FROM VIP.VIDEOS v
                    LEFT JOIN VIP.RESUME r ON v.RESUME_ID = r.RESUME_ID
                    LEFT JOIN VIP.USERS u ON v.USER_ID = u.ID
                    WHERE v.ID = :id
                """, id=video_id)
                result = cursor.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail=f"Video with id {video_id} not found")
   
                desired_company, username = result
                print("회사 이름",desired_company)
                if desired_company:
                    return desired_company
                elif username:
                    return f"Company for {username}"
                else:
                    return "Unknown Company"
            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.error(f"Database error in get_company_name for video_id {video_id}: "
                             f"Error Code: {error.code}, "
                             f"Error Message: {error.message}")
                raise HTTPException(status_code=500, detail=f"Database error: {error.message}")
            finally:
                cursor.close()

    try:
        return await asyncio.to_thread(_get_company_name, video_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_company_name for video_id {video_id}: {str(e)}")
        return "Unknown Company"
    
def get_question_text(video_id: int) -> str:
    with pool.acquire() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT question_text FROM videos WHERE id = :id", id=video_id)
            result = cursor.fetchone()
            if result:
                lob = result[0]
                if lob is not None:
                    # CLOB 데이터를 문자열로 변환
                    return lob.read()
                else:
                    return ""
            else:
                raise HTTPException(status_code=404, detail="Question text not found")
        finally:
            cursor.close()

def get_video_info(video_id: int):
    """
    Oracle DB에서 video_id에 해당하는 파일 경로와 파일 이름을 조회하는 함수
    """
    logger.info(f"Fetching video info for video_id: {video_id}")
    with pool.acquire() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT FILE_PATH, FILE_NAME
                FROM videos
                WHERE ID = :id
            """, id=video_id)
            result = cursor.fetchone()
            if result:
                logger.info(f"Video info found: {result}")
                return {"file_path": result[0], "file_name": result[1]}
            logger.warning(f"No video info found for video_id: {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching video info: {e}")
            raise
        finally:
            cursor.close()

def download_video_from_gcs(gcs_url: str, local_path: str) -> str:
    """
    GCS에서 비디오 파일을 로컬로 다운로드하는 함수
    """
    logger.info(f"Downloading video from GCS: {gcs_url}")
    bucket_name = "cloud-storage-upload"
    gcs_file_path = gcs_url.split(f"https://storage.googleapis.com/{bucket_name}/")[1]
    
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)

    local_file_path = os.path.join(local_path, gcs_file_path)
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    blob.download_to_filename(local_file_path)

    logger.info(f"Video downloaded to {local_file_path}")
    return local_file_path

def analyze_video(file_path: str):
    """
    비디오 파일을 분석하는 함수 (head-eye 분석 포함)
    """
    try:
        logger.info(f"Starting video analysis for file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file {file_path} does not exist.")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video source {file_path}")
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        progress_bar = tqdm(total=total_frames * 2, desc="Video Analysis Progress")

        analysis_functions = [
            (head_eye, "Head-eye"),
        ]

        results = {}
        all_frames = []
        for func, name in analysis_functions:
            try:
                if name == "Head-eye":
                    output_frames, message, head_score, eye_score = func(file_path)
                    results[f"{name.lower()}_head_score"] = head_score
                    results[f"{name.lower()}_eye_score"] = eye_score
                
                results[f"{name.lower()}_message"] = message
                all_frames.extend(output_frames)
                progress_bar.update(total_frames)  # Update progress by 50%
                print(f"{name} frames: {len(output_frames)}")
            except Exception as e:
                print(f"Error in {name} analysis: {e}")
                if name == "Head-eye":
                    results[f"{name.lower()}_head_score"] = 0
                    results[f"{name.lower()}_eye_score"] = 0
                results[f"{name.lower()}_message"] = f"Error: {str(e)}"
            
            progress_bar.update(total_frames)  # Update progress by another 50%

        progress_bar.close()

        analyzed_file_path = save_analyzed_video(file_path, all_frames)

        valid_scores = [score for key, score in results.items() if key.endswith('_score') and isinstance(score, (int, float))]
        results["avg_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        results["analyzed_file_path"] = analyzed_file_path

        audio_analysis = extract_and_analyze_audio(file_path)
        results["audio_analysis"] = audio_analysis

        speech_results = {}
        logger.info("Extracting audio from video")
        audio_file_path = os.path.splitext(file_path)[0] + ".wav"
        extract_audio_from_video(file_path, audio_file_path)
        
        logger.info("Transcribing audio")
        transcription = transcribe_with_clova(audio_file_path)
        
        if transcription:
            logger.info("Performing sentiment analysis")
            sentiment_result = analyze_sentiment(transcription)
            
            logger.info("Summarizing text")
            summary_result = summarize_text(transcription)
            
            # CLOVA 결과 처리 및 피드백 생성
            transcription_feedback = generate_transcription_feedback(transcription)
            sentiment_feedback = generate_sentiment_feedback(sentiment_result)
            summary_feedback = generate_summary_feedback(summary_result)
            overall_feedback = generate_overall_feedback(sentiment_result)  # 여기를 수정
            
            speech_results = {
                "transcription": transcription,
                "sentiment_analysis": sentiment_result or {},
                "text_summary": summary_result or {},
                "feedback": {
                    "transcription_feedback": transcription_feedback,
                    "sentiment_feedback": sentiment_feedback,
                    "summary_feedback": summary_feedback,
                    "overall_feedback": overall_feedback
                }
            }
        else:
            logger.warning("No transcription available or error in speech recognition")
            speech_results = {}

        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        logger.info("Video analysis completed")
        print("Results: ", results)
        print("Speech Results: ", speech_results)
        return results, speech_results
        
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}", exc_info=True)
        raise

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    """
    Google Cloud Storage에 비디오 파일을 업로드하는 함수
    """
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    storage_client = storage.Client.from_service_account_json(credentials_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # MIME 타입을 video/mp4로 설정하여 업로드
    blob.upload_from_filename(local_file_path, content_type='video/mp4')

    print(f"File {local_file_path} uploaded to {destination_blob_name}.")
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"

def save_analyzed_video(original_file_path, all_frames):
    """
    분석된 프레임을 하나의 비디오로 저장하고 GCS에 업로드하는 함수
    """
    if len(all_frames) == 0:
        print("No frames to save.")
        return None

    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    analyzed_file_path = os.path.join(temp_dir, f"analyzed_{timestamp}.mp4")

    try:
        height, width, _ = all_frames[0].shape
        fps = 30
        print(f"Video size: {width}x{height}, FPS: {fps}")

        frame_files = []
        for idx, frame in enumerate(all_frames):
            frame_file = os.path.join(temp_dir, f'frame_{idx:04d}.png')
            cv2.imwrite(frame_file, frame)
            frame_files.append(frame_file)

        ffmpeg_path = r'C:\Users\ict02-17\Desktop\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe'
        
        ffmpeg_command = (
            f'"{ffmpeg_path}" -y -framerate {fps} -i "{temp_dir}/frame_%04d.png" '
            f'-c:v libx264 -pix_fmt yuv420p "{analyzed_file_path}"'
        )
        
        process = subprocess.Popen(ffmpeg_command, shell=True, cwd=temp_dir, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"FFmpeg stdout: {stdout.decode()}")
            print(f"FFmpeg stderr: {stderr.decode()}")
            raise Exception(f"FFmpeg command failed with return code {process.returncode}")
        
        if not os.path.exists(analyzed_file_path):
            raise FileNotFoundError(f"File {analyzed_file_path} was not created.")
        
        print(f"Analyzed video saved at {analyzed_file_path}")

        bucket_name = "cloud-storage-upload"
        destination_blob_name = f"analyzed_videos/analyzed_{timestamp}.mp4"
        gcs_url = upload_to_gcs(analyzed_file_path, bucket_name, destination_blob_name)

        os.remove(analyzed_file_path)

        for frame_file in frame_files:
            os.remove(frame_file)

        return gcs_url

    except Exception as e:
        print(f"Error saving analyzed video: {e}")
        return None

def save_analysis_result(video_id: int, results: dict):
    with pool.acquire() as connection:
        cursor = connection.cursor()
        try:
            # 새로운 분석 ID 생성
            cursor.execute("SELECT SEQ_VIDEO_ANALYSIS_ID.NEXTVAL FROM DUAL")
            next_id = cursor.fetchone()[0]

            # NumPy 타입을 Python 기본 타입으로 변환
            converted_results = convert_numpy_types(results)

            # 헤드-아이 분석 메시지 추출
            head_eye_message = converted_results.get("head-eye_message", "")
            # 오디오 분석 결과 추출
            audio_analysis = converted_results.get("audio_analysis", {}).get("audio_analysis", {})

            # 안전하게 값 추출
            def safe_get(d, key, default=0):
                value = d.get(key, default)
                return float(value) if isinstance(value, (int, float, np.number)) else default

            # 분석 결과를 저장하는 SQL 쿼리 실행
            cursor.execute(""" 
                INSERT INTO video_analysis 
                (id, video_id, head_score, eye_score, avg_score, analysis_date, analyzed_file_path, 
                audio_pitch, audio_tempo, audio_volume, audio_spectral_centroid,
                audio_feedback, head_eye_feedback)
                VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13)
            """, (
                next_id,
                video_id,
                round(safe_get(converted_results, "head-eye_head_score")),
                round(safe_get(converted_results, "head-eye_eye_score")),
                round(safe_get(converted_results, "avg_score")),
                datetime.now(),
                converted_results.get("analyzed_file_path", ""),
                safe_get(audio_analysis, "average_pitch"),
                safe_get(audio_analysis, "tempo"),
                safe_get(audio_analysis, "average_volume"),
                safe_get(audio_analysis, "average_spectral_centroid"),
                "\n".join(converted_results.get("audio_analysis", {}).get("feedback_messages", [])),
                head_eye_message
            ))
            
            # 변경사항 커밋
            connection.commit()
            print(f"Analysis result saved successfully for video_id: {video_id}")
        except (cx_Oracle.DatabaseError, ValueError, KeyError, TypeError) as e:
            # 오류 발생 시 로그 기록 및 롤백
            print(f"Error in save_analysis_result: {e}")
            print(f"Converted results: {converted_results}")
            connection.rollback()
        finally:
            # 커서 종료
            cursor.close()

def save_speech_analysis_result(video_id: int, results: dict):
    with pool.acquire() as connection:
        cursor = connection.cursor()
        try:
            # 새로운 음성 분석 ID 생성
            cursor.execute("SELECT SEQ_VIDEO_SPEECH_ANALYSIS_ID.NEXTVAL FROM DUAL")
            next_id = cursor.fetchone()[0]

            # 음성 인식 결과 추출
            transcription = results.get("transcription", "")
            # 감정 분석 결과 추출
            sentiment_overall = results.get("sentiment_analysis", {}).get("document", {}).get("sentiment", "")
            # 감정 분석 신뢰도 추출 및 소수점 없는 정수로 변환
            sentiment_confidence = round(float(results.get("sentiment_analysis", {}).get("document", {}).get("confidence", {}).get("positive", 0)) * 100)
            # 요약 결과 추출
            summary = results.get("text_summary", {}).get("summary", "")

            # 피드백 정보 추출
            feedback = results.get("feedback", {})
            feedback_str = json.dumps(feedback)  # 피드백을 JSON 문자열로 변환

            # 음성 분석 결과를 저장하는 SQL 쿼리 실행
            cursor.execute("""
                INSERT INTO video_speech_analysis 
                (id, video_id, transcription, sentiment_overall, sentiment_confidence, summary, feedback, created_at)
                VALUES (:1, :2, :3, :4, :5, :6, :7, :8)
            """, (
                next_id,
                video_id,
                transcription,
                sentiment_overall,
                sentiment_confidence,
                summary,
                feedback_str,
                datetime.now()
            ))
            connection.commit()
            print(f"Speech analysis result saved successfully for video_id: {video_id}")
        except cx_Oracle.DatabaseError as e:
            print(f"Database error: {e}")
            connection.rollback()
        except Exception as e:
            print(f"Error in save_speech_analysis_result: {e}")
            connection.rollback()
        finally:
            cursor.close()

async def analyze_interview_with_claude(transcription, question, company_name, answer_duration):
    
    keywords = extract_keywords(transcription, top_n=50)
    json_format = '''
    {
        "content_analysis": {
            "logic_score": 0,
            "key_ideas": "",
            "specific_examples": "",
            "consistency_score": 0
        },
        "sentiment_analysis": {
            "tone": "",
            "confidence_score": 0
        },
        "language_pattern_analysis": {
            "professional_vocab_score": 0,
            "repetitive_expressions": "",
            "grammar_structure_score": 0,
            "industry_specific_terms": []
        },
        "tone_tension_analysis": {
            "consistency_score": 0,
            "tension_detected": "",
            "hesitations": 0,
            "long_pauses": 0
        },
        "insight_analysis": {
            "creativity_score": 0,
            "problem_solving_score": 0
        },
        "star_analysis": {
            "situation_score": 0,
            "task_score": 0,
            "action_score": 0,
            "result_score": 0
        },
        "company_fit_analysis": {
            "alignment_score": 0,
            "company_knowledge": "",
            "industry_understanding": ""
        },
        "question_comprehension": {
            "relevance_score": 0,
            "missed_points": []
        },
        "answer_duration_analysis": {
            "duration": 0,
            "pace_score": 0,
            "comment": ""
        },
        "keywords": [],
        "overall_quality": 0,
        "improvement_suggestions": "",
        "comprehensive_evaluation": ""
    }
    '''

    try:
        message = await asyncio.to_thread(
            client.messages.create,
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            system="당신은 면접 답변을 분석하는 경험 많은 인사 담당자입니다. 분석 결과를 JSON 형식으로 제공해주세요.",
            messages=[
                {"role": "user", "content": f"""다음 면접 질문과 답변, 그리고 회사명을 주어진 기준에 따라 상세히 분석해주세요:

    질문: {question}
    답변: {transcription}
    회사명: {company_name}
    답변 시간: {answer_duration}초

    분석 기준:
    1. 내용 분석:
       - 논리성 평가 (1-10점)
       - 핵심 아이디어 도출
       - 구체적인 예시 제공 여부
       - 답변의 일관성 (1-10점)
    2. 감정 분석:
       - 감정 톤 (긍정/부정/중립)
       - 자신감 표현 정도 (1-10점)
    3. 언어 사용 패턴 분석:
       - 전문적 어휘 사용 정도 (1-10점)
       - 반복적인 표현 감지
       - 문법 및 문장 구조의 적절성 (1-10점)
       - 업계 관련 전문 용어 사용
    4. 어조 및 텐션 분석:
       - 어조의 일관성 (1-10점)
       - 긴장감 감지 여부
       - 주저함이나 긴 휴지의 횟수
    5. 인사이트 분석:
       - 창의성 평가 (1-10점)
       - 문제 해결 접근 방식의 논리성 (1-10점)
    6. STAR 방식 분석:
       - Situation, Task, Action, Result 각각의 명확성 (1-10점)
    7. 회사 적합성 분석:
       - 회사에 대한 이해도 (1-10점)
       - 회사와 산업에 대한 지식 평가
    8. 질문 이해도 분석:
       - 질문 관련성 점수 (1-10점)
       - 놓친 중요 포인트 목록
    9. 답변 시간 분석:
       - 주어진 답변 시간의 적절성
       - 답변 속도 평가 (1-10점)
       - 답변 시간에 대한 코멘트

    각 항목에 대해 점수와 함께 간단한 설명을 제공해주세요.
    전체적인 답변의 품질을 100점 만점으로 평가하고, 개선을 위한 제안사항을 추가해주세요.

    마지막으로, 'comprehensive_evaluation' 항목에 다음 내용을 포함한 종합적인 평가를 제공해주세요:
    1. 답변 구조와 내용: STAR 기법 활용도, 구체적 예시 제공 여부
    2. 경험과 성장: 어려움 극복 사례, 학습 능력 및 성장 과정 설명
    3. 회사와의 연관성: 회사 가치관/미션과의 연결, 업계 트렌드 이해도
    4. 질문 이해도: 질문 요점 파악 능력, 모든 부분에 대한 답변 여부
    5. 전문성과 자신감: 기술 용어 사용의 적절성, 자신감 있는 답변 여부
    6. 개선을 위한 구체적 제안: 답변 길이 조절, 비언어적 커뮤니케이션 개선 등

    이 종합 평가는 면접자에게 실질적인 피드백을 제공하고 향후 면접 준비에 도움을 줄 수 있도록 구체적이고 실행 가능한 조언을 포함해야 합니다.

    추가로, 다음 키워드를 고려하여 분석해주세요:
    주요 키워드: {keywords}

    응답은 다음과 같은 JSON 형식으로 제공해주세요:
    {json_format}"""}
            ]
        )

        content = message.content[0].text if isinstance(message.content, list) else message.content
        analysis_result = json.loads(content)
        analysis_result["keywords"] = [keyword for keyword, _ in keywords]
        analysis_result["answer_duration_analysis"]["duration"] = answer_duration
    except json.JSONDecodeError:
        print(f"JSON parsing failed. Raw content: {content}")
        analysis_result = {
            "error": "JSON parsing failed",
            "content": content,
            "keywords": [keyword for keyword, _ in keywords],
            "overall_quality": 0,
            "improvement_suggestions": "분석 실패로 인해 개선 제안을 제공할 수 없습니다.",
            "answer_duration_analysis": {"duration": answer_duration, "pace_score": 0, "comment": "분석 실패"},
            "comprehensive_evaluation": "분석 실패로 인해 종합 평가를 제공할 수 없습니다."
        }
    except Exception as e:
        print(f"Unexpected error in analyze_interview_with_claude: {e}")
        analysis_result = {
            "error": "Unexpected error",
            "content": str(e),
            "keywords": [keyword for keyword, _ in keywords],
            "overall_quality": 0,
            "improvement_suggestions": "분석 중 예기치 않은 오류가 발생했습니다.",
            "answer_duration_analysis": {"duration": answer_duration, "pace_score": 0, "comment": "분석 실패"},
            "comprehensive_evaluation": "분석 중 예기치 않은 오류로 인해 종합 평가를 제공할 수 없습니다."
        }
    
    print("claude 면접분석:", analysis_result)
    return analysis_result
# 주석: 1분 내외의 짧은 답변에서 분석이 어려울 수 있는 항목들
"""
1. 내용 분석:
   - 논리성 평가: 짧은 답변에서는 완전한 논리 구조를 파악하기 어려울 수 있음
   - 구체적인 예시 제공 여부: 시간 제약으로 구체적인 예시를 제공하기 어려울 수 있음
   - 답변의 일관성: 짧은 답변에서는 일관성을 평가하기에 충분한 내용이 없을 수 있음

2. 감정 분석:
   - 대체로 분석 가능하나, 짧은 답변에서는 감정의 변화를 관찰하기 어려울 수 있음

3. 언어 사용 패턴 분석:
   - 반복적인 표현 감지: 짧은 답변에서는 반복을 관찰하기 어려울 수 있음
   - 업계 관련 전문 용어 사용: 시간 제약으로 충분한 전문 용어를 사용하지 못할 수 있음

4. 어조 및 텐션 분석:
   - 어조의 일관성: 짧은 답변에서는 일관성을 평가하기 어려울 수 있음
   - 주저함이나 긴 휴지의 횟수: 짧은 답변에서는 이를 관찰할 충분한 시간이 없을 수 있음

5. 인사이트 분석:
   - 창의성 평가: 짧은 답변에서는 창의적인 아이디어를 충분히 표현하기 어려울 수 있음
   - 문제 해결 접근 방식의 논리성: 복잡한 문제 해결 과정을 설명하기에 시간이 부족할 수 있음

6. STAR 방식 분석:
   - 1분 내외의 답변으로는 Situation, Task, Action, Result를 모두 충분히 설명하기 어려울 수 있음

7. 회사 적합성 분석:
   - 회사와 산업에 대한 지식 평가: 짧은 시간 내에 충분한 지식을 표현하기 어려울 수 있음

8. 질문 이해도 분석:
   - 대체로 분석 가능하나, 복잡한 질문의 경우 짧은 답변으로는 완전한 이해도를 판단하기 어려울 수 있음

9. 답변 시간 분석:
   - 대체로 분석 가능하나, 1분 내외의 답변에서는 '적절한' 답변 시간을 정의하기 어려울 수 있음

전반적으로, 1분 내외의 답변에서는 깊이 있는 분석이 어려울 수 있으며, 특히 복잡한 사고나 
구체적인 예시를 요구하는 항목들에서 분석의 한계가 있을 수 있습니다.
"""
async def save_claude_analysis_result(video_id: int, analysis_result: dict):
    logging.info(f"Saving Claude analysis result for video_id: {video_id}")
    if not isinstance(video_id, int) or not isinstance(analysis_result, dict):
        raise ValueError("Invalid input parameters")

    def _save_claude_analysis_result(video_id: int, analysis_result: dict):
        with pool.acquire() as connection:
            cursor = connection.cursor()
            try:
                # 새로운 Claude 분석 ID 생성
                cursor.execute("SELECT SEQ_CLAUDE_ANALYSIS_ID.NEXTVAL FROM DUAL")
                next_id = cursor.fetchone()[0]

                # 분석 데이터를 JSON 문자열로 변환
                analysis_data_str = json.dumps(analysis_result)
                # 개선 제안 추출
                improvement_suggestions_str = analysis_result.get('improvement_suggestions', '')
                # 키워드 추출 및 JSON 문자열로 변환
                keywords_str = json.dumps(analysis_result.get('keywords', []))

                # Claude 분석 결과를 저장하는 SQL 쿼리 실행
                cursor.execute("""
                    INSERT INTO claude_analysis 
                    (id, video_id, analysis_data, created_at, overall_score, improvement_suggestions, keywords)
                    VALUES (:1, :2, :3, :4, :5, :6, :7)
                """, (
                    next_id,
                    video_id,
                    analysis_data_str,
                    datetime.now(),
                    # 전체 점수를 소수점 없는 정수로 변환
                    round(float(analysis_result.get('overall_quality', 0))),
                    improvement_suggestions_str,
                    keywords_str
                ))
                # 변경사항 커밋
                connection.commit()
                logging.info(f"Claude analysis result saved successfully for video_id: {video_id}")
            except cx_Oracle.DatabaseError as e:
                # 데이터베이스 오류 처리
                logging.error(f"Database error while saving Claude analysis for video_id {video_id}: {e}")
                connection.rollback()
                raise
            except Exception as e:
                # 기타 예외 처리
                logging.error(f"Unexpected error in save_claude_analysis_result for video_id {video_id}: {e}", exc_info=True)
                connection.rollback()
                raise
            finally:
                # 커서 종료
                cursor.close()

    try:
        # 비동기적으로 데이터베이스 작업 실행
        await asyncio.to_thread(_save_claude_analysis_result, video_id, analysis_result)
    except Exception as e:
        # 최종 예외 처리 및 로깅
        logging.error(f"Error in save_claude_analysis_result for video_id {video_id}: {e}", exc_info=True)
        raise

def extract_keywords(text, top_n=50):
    okt = Okt()
    tokens = okt.pos(text)  # pos 메소드를 사용하여 품사 태깅
    
    # 명사, 형용사, 동사만 선택하고 2글자 이상인 단어만 선택
    valid_tokens = [word for word, pos in tokens if (pos in ['Noun', 'Adjective', 'Verb']) and len(word) > 1]
    
    word_count = Counter(valid_tokens)
    return word_count.most_common(top_n)

def extract_audio_from_video(video_path, output_audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def analyze_audio(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    
    # 음정 분석
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    avg_pitch = np.mean(pitches) if len(pitches) > 0 else 0
    
    # 템포 분석
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # 볼륨 분석 (RMS)
    rms = librosa.feature.rms(y=y)[0]
    avg_volume = np.mean(rms)
    
    # 스펙트럼 중심 분석
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroids)
    
    # 분석 결과
    audio_analysis = {
        "average_pitch": avg_pitch,
        "tempo": tempo,
        "average_volume": avg_volume,
        "average_spectral_centroid": avg_spectral_centroid
    }
    
    # 피드백 메시지 생성 (위임)
    feedback_messages = generate_audio_feedback(audio_analysis)
    
    # 결과 반환
    return {
        "audio_analysis": audio_analysis,
        "feedback_messages": feedback_messages
    }
def generate_audio_feedback(audio_analysis: dict):
    feedback_messages = []
    
    # 음정 평가
    avg_pitch = audio_analysis.get("average_pitch", 0)

    # 평균 음정을 10으로 나눈 값으로 피드백 생성
    avg_pitch_display = int(avg_pitch / 10)
    if avg_pitch < 140:
        feedback_messages.append(f"음정이 낮습니다. 평균 음정이 {int(avg_pitch_display)}Hz로, 자신감이 부족한 인상을 줄 수 있습니다. 좀 더 높은 톤으로 말하는 것이 좋습니다.\n")
    elif avg_pitch > 180:
        feedback_messages.append(f"음정이 너무 높습니다. 평균 음정이 {int(avg_pitch_display)}Hz로, 긴장되거나 흥분한 인상을 줄 수 있습니다. 조금 더 차분한 톤으로 말하는 것이 좋습니다.\n")
    else:
        feedback_messages.append(f"음정이 적절합니다. 평균 음정이 {int(avg_pitch_display)}Hz로 안정적입니다.\n")
    
    # 템포 평가
    tempo = audio_analysis.get("tempo", 0)
    if tempo < 80:
        feedback_messages.append(f"말하는 속도가 느립니다. 템포가 {int(tempo)}bpm로 측정되었습니다. 조금 더 빠르게 말해보세요.\n")
    elif tempo > 150:
        feedback_messages.append(f"말하는 속도가 빠릅니다. 템포가 {int(tempo)}bpm로 측정되었습니다. 조금 더 천천히 말하는 것이 좋습니다.\n")
    else:
        feedback_messages.append(f"말하는 속도가 적절합니다. 템포가 {int(tempo)}bpm로 자연스럽습니다.\n")
    
    # 볼륨 평가
    avg_volume = audio_analysis.get("average_volume", 0)
    if avg_volume < 0.02:
        feedback_messages.append("목소리가 너무 작습니다. 좀 더 큰 소리로 말하는 것이 좋습니다.\n")
    elif avg_volume > 0.1:
        feedback_messages.append("목소리가 너무 큽니다. 조금 더 낮은 소리로 말하는 것이 좋습니다.\n")
    else:
        feedback_messages.append("목소리의 크기가 적절합니다. 현재 상태를 유지하면서 연습하시면 됩니다.\n")
    
    # 스펙트럼 중심 평가
    avg_spectral_centroid = audio_analysis.get("average_spectral_centroid", 0)
    if avg_spectral_centroid < 2000:
        feedback_messages.append(f"소리가 다소 무겁게 들립니다. 평균 스펙트럼 중심이 {int(avg_spectral_centroid)}Hz로, 조금 더 밝고 명확하게 말해보세요.\n")
    elif avg_spectral_centroid > 4000:
        feedback_messages.append(f"소리가 다소 날카롭게 들립니다. 평균 스펙트럼 중심이 {int(avg_spectral_centroid)}Hz로, 조금 더 차분하게 말하는 것이 좋습니다.\n")
    else:
        feedback_messages.append(f"소리의 톤이 적절합니다. 평균 스펙트럼 중심이 {int(avg_spectral_centroid)}Hz로 자연스럽습니다.\n")
    
    return feedback_messages


def extract_and_analyze_audio(video_path):
    audio_file_path = os.path.splitext(video_path)[0] + ".wav"
    extract_audio_from_video(video_path, audio_file_path)
    
    audio_analysis = analyze_audio(audio_file_path)
    
    os.remove(audio_file_path)
    
    return audio_analysis

def transcribe_with_clova(audio_file_path):
    headers = {
        'X-NCP-APIGW-API-KEY-ID': CLOVA_CLIENT_ID,
        'X-NCP-APIGW-API-KEY': CLOVA_CLIENT_SECRET,
        'Content-Type': 'application/octet-stream'
    }

    url_with_params = CLOVA_API_URL + '?lang=Kor&completion=sync'

    try:
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            
        response = requests.post(
            url_with_params,
            headers=headers,
            data=audio_data
        )

        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Content: {response.content}")
        
        if response.status_code == 200:
            result = response.json()
            if 'text' in result:
                return result['text']
            else:
                print("No 'text' field in the response")
                return None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def analyze_sentiment(text):
    headers = {
        'X-NCP-APIGW-API-KEY-ID': CLOVA_CLIENT_ID,
        'X-NCP-APIGW-API-KEY': CLOVA_CLIENT_SECRET,
        'Content-Type': 'application/json'
    }
    data = {
        'content': text
    }
    
    response = requests.post(SENTIMENT_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Sentiment Analysis Error: {response.status_code}, {response.text}")
        return None

def summarize_text(text):
    if len(text.split()) < 10:  # 단어가 10개 미만인 경우
        print("Text too short for summarization")
        return {"summary": "Text too short for summarization"}

    headers = {
        'X-NCP-APIGW-API-KEY-ID': CLOVA_CLIENT_ID,
        'X-NCP-APIGW-API-KEY': CLOVA_CLIENT_SECRET,
        'Content-Type': 'application/json'
    }
    data = {
        'document': {
            'content': text
        },
        'option': {
            'language': 'ko',
            'model': 'news',
            'tone': 0,
            'summaryCount': 3
        }
    }

    try:
        response = requests.post(SUMMARY_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Summary Error: {response.status_code}, {response.text}")
            return {"summary": "Summarization failed"}
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return {"summary": "Summarization request failed"}

def generate_transcription_feedback(transcription):
    feedback = "음성 인식 결과에 대한 피드백:\n"
    word_count = len(transcription.split())
    
    if word_count < 50:
        feedback += "답변이 다소 짧습니다. 더 자세한 설명을 제공하면 좋을 것 같습니다.\n"
    elif word_count > 200:
        feedback += "답변이 매우 길습니다. 핵심 내용을 간결하게 전달하는 연습이 필요할 수 있습니다.\n"
    else:
        feedback += "답변 길이가 적절합니다.\n"
    
    if '음...' in transcription or '어...' in transcription:
        feedback += "말할 때 '음...', '어...' 등의 간투사를 줄이는 것이 좋습니다.\n"
    
    return feedback

def generate_sentiment_feedback(sentiment_result):
    feedback = "감정 분석 결과에 대한 피드백:\n"
    sentiment = sentiment_result['document']['sentiment']
    confidence = sentiment_result['document']['confidence']
    
    if sentiment == 'positive':
        feedback += f"긍정적인 어조로 답변했습니다. (확신도: {confidence['positive']:.2f})\n"
        if confidence['positive'] < 0.7:
            feedback += "더 자신감 있는 어조로 말하면 좋을 것 같습니다.\n"
    elif sentiment == 'negative':
        feedback += f"다소 부정적인 어조로 답변했습니다. (확신도: {confidence['negative']:.2f})\n"
        feedback += "  가능한 긍정적이고 건설적인 표현을 사용하는 것이 좋습니다.\n"
    else:
        feedback += "중립적인 어조로 답변했습니다.\n"
    
    return feedback

def generate_summary_feedback(summary_result):
    feedback = "답변 요약 및 피드백:\n"
    summary = summary_result['summary']
    
    feedback += f"- 요약: {summary}\n"
    feedback += "- 핵심 내용을 잘 전달했는지 확인해 보세요.\n"
    feedback += "- 면접관의 질문에 직접적으로 답변했는지 검토해 보세요.\n"
    
    return feedback

def generate_overall_feedback(sentiment_result):
    overall_feedback = "감정 분석 피드백:\n"
    sentiment = sentiment_result['document']['sentiment']
    confidence = sentiment_result['document']['confidence']

    if sentiment == 'positive':
        overall_feedback += f"긍정적인 어조로 답변했습니다. 이는 면접관에게 좋은 인상을 줄 수 있습니다. (확신도: {confidence['positive']:.2f})\n"
        if confidence['positive'] < 0.7:
            overall_feedback += "다만, 더 자신감 있는 어조로 말하면 좋을 것 같습니다. 자신의 능력과 경험을 확신을 가지고 표현해 보세요.\n"
        overall_feedback += "긍정적인 태도를 유지하면서 다음을 고려해 보세요:\n"
        overall_feedback += "- 구체적인 성과나 경험을 언급하여 긍정적인 면을 뒷받침하세요.\n"
        overall_feedback += "- 열정을 표현할 때는 목소리의 톤과 속도를 약간 높이는 것도 좋습니다.\n"
        overall_feedback += "- 긍정적인 body language를 사용하세요. 예를 들어, 적절한 미소, 열린 자세 등이 도움이 될 수 있습니다.\n"
    elif sentiment == 'negative':
        overall_feedback += f"다소 부정적인 어조로 답변했습니다. 이는 면접관에게 좋지 않은 인상을 줄 수 있습니다. (확신도: {confidence['negative']:.2f})\n"
        overall_feedback += "가능한 긍정적이고 건설적인 표현을 사용하세요. 어려움이나 실패 경험을 언급할 때도 그로부터 배운 점이나 극복 과정에 초점을 맞추세요.\n"
        overall_feedback += "부정적인 어조를 개선하기 위해 다음을 시도해 보세요:\n"
        overall_feedback += "- 문제나 어려움을 언급할 때는 항상 해결책이나 학습 경험과 함께 이야기하세요.\n"
        overall_feedback += "- '못했다', '실패했다' 대신 '도전했다', '경험했다'와 같은 중립적이거나 긍정적인 단어를 사용하세요.\n"
        overall_feedback += "- 과거의 부정적 경험을 현재의 강점으로 연결 지어 설명해보세요.\n"
    else:
        overall_feedback += "중립적인 어조로 답변했습니다. 상황에 따라 적절할 수 있지만, 때로는 열정과 관심을 더 표현하는 것이 도움될 수 있습니다.\n"
        overall_feedback += "중립적인 어조를 보완하기 위해 다음을 고려해 보세요:\n"
        overall_feedback += "- 특히 지원 동기나 향후 계획을 이야기할 때는 열정을 조금 더 표현해 보세요.\n"
        overall_feedback += "- 본인의 강점이나 주요 성과를 언급할 때는 자신감 있는 어조를 사용하세요.\n"
        overall_feedback += "- 답변에 개인적인 이야기나 경험을 포함시켜 감정적 연결을 만들어보세요.\n"

    overall_feedback += "\n추가 조언:\n"
    overall_feedback += "- 면접관의 질문을 주의 깊게 듣고, 질문의 의도에 맞게 답변하세요.\n"
    overall_feedback += "- 답변 중 잠시 생각할 시간이 필요하다면, '잠시 생각할 시간을 주시겠습니까?'라고 정중히 요청하세요.\n"
    overall_feedback += "- 어조뿐만 아니라 표정과 자세도 일치시켜 진정성 있는 커뮤니케이션을 하세요.\n"
    overall_feedback += "- 면접 후반부로 갈수록 긴장이 풀려 어조가 변할 수 있으니, 일관된 태도를 유지하도록 노력하세요.\n"

    return overall_feedback
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()[0] if obj.size == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def send_analysis_complete_to_spring(video_id: int, analysis_results: Dict[str, Any]) -> bool:
    try:
        headers = {
            'Content-Type': 'application/json',
        }

        message = {
            "videoId": video_id,
            "message": "분석이 끝났습니다",
            "analysisResults": analysis_results
        }

        json_message = json.dumps(message, cls=NumpyEncoder)

        logger.info(f"Sending analysis complete message to Spring backend for video_id: {video_id}")
        logger.info(f"URL: {SPRING_BACKEND_URL}")
        logger.info(f"Message: {json_message}")

        response = requests.post(SPRING_BACKEND_URL, data=json_message, headers=headers)

        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")

        if response.status_code == 200:
            logger.info(f"Analysis complete message sent successfully to Spring backend for video_id: {video_id}")
            return True
        else:
            logger.error(f"Failed to send analysis complete message to Spring backend. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error sending analysis complete message to Spring backend: {e}", exc_info=True)
        return False

def integrate_analysis_data(video_analysis, speech_analysis, claude_analysis):
    return {
        "video_analysis": video_analysis,
        "speech_analysis": speech_analysis,
        "claude_analysis": claude_analysis if claude_analysis is not None else {}
    }
    
def calculate_answer_duration(video_path: str) -> float:
    try:
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"Error calculating video duration: {e}")
        return 0.0
async def update_video_duration(video_id: int, duration: float):
    def _update_video_duration(video_id: int, duration: float):
        with pool.acquire() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute("""
                    UPDATE VIP.VIDEOS 
                    SET ANSWER_DURATION = :duration 
                    WHERE ID = :id
                """, duration=duration, id=video_id)
                connection.commit()
                logger.info(f"Updated duration for video_id {video_id}: {duration} seconds")
            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.error(f"Database error updating duration for video_id {video_id}: "
                             f"Error Code: {error.code}, "
                             f"Error Message: {error.message}")
                connection.rollback()
                raise
            finally:
                cursor.close()

    await asyncio.to_thread(_update_video_duration, video_id, duration)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)