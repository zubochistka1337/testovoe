import json
import cv2
from fastapi import FastAPI, status, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from contextlib import asynccontextmanager
from datetime import date, datetime
from multiprocessing import Process
import redis
import redis.asyncio as aioredis
import os
import asyncio
from typing import cast, Annotated
from ultralytics import YOLO
import time
import sqlite3
import os.path
import numpy as np
import uuid
from pydantic import BaseModel
import logging
import psutil


REDIS_HOST = 'redis'
REDIS_PORT = 6379
REDIS_DB = 0
PID_KEY = 'pid'
STOP_KEY = 'stop'
STATUS_KEY = 'status'
IMG_CHANNEL = 'img_channel'
redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
aio_redis_conn = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
IMG_FOLDER = './img_humans'


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists('database.db'):
        with sqlite3.connect('database.db') as con:
            cur = con.cursor()
            stmt = '''CREATE TABLE IF NOT EXISTS humans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            img_url TEXT,
            created_at TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER)'''
            cur.execute(stmt)
            con.commit()
    if not os.path.exists(IMG_FOLDER):
        os.mkdir(IMG_FOLDER)
    if not os.path.exists('./settings.json'):
        with open('./settings.json', 'w') as outfile:
            new_settings = {'width': 640, 'height': 480, 'x1': 0, 'y1': 0}
            json.dump(new_settings, outfile, indent=4)
    try:
        redis_conn.ping()
        redis_conn.delete(PID_KEY)
        redis_conn.delete(STOP_KEY)
        redis_conn.delete(STATUS_KEY)
    except redis.exceptions.ConnectionError:
        print('redis connection error, shutting down')
        exit()
    yield
app = FastAPI(title='Human detection test task', lifespan=lifespan)


def save_img(img: np.ndarray) -> str:
    img_uuid_name = str(uuid.uuid4()) + '.jpg'
    img_uuid_path = IMG_FOLDER + '/' + img_uuid_name
    cv2.imwrite(img_uuid_path, img)
    return img_uuid_path


def camera_runtime():
    logger_file = 'output.log'
    logger = logging.getLogger(logger_file)
    try:
        model = YOLO('./best.pt')
        redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        pid = os.getpid()
        redis_conn.set(PID_KEY, pid)

        with open('settings.json', 'r', encoding='UTF-8') as f:
            settings_dict = json.load(f)
        width = settings_dict['width']
        height = settings_dict['height']
        settings_x1 = settings_dict['x1']
        settings_y1 = settings_dict['y1']

        cap = cv2.VideoCapture(0)
        freeze_time = False
        capture_time = False
        start_capture_time = None
        start_freeze_time = None
        while True:
            if redis_conn.get(STOP_KEY):
                redis_conn.delete(STOP_KEY)
                redis_conn.delete(PID_KEY)
                redis_conn.delete(STATUS_KEY)
                cap.release()
                break

            if freeze_time:
                if time.time() >= start_freeze_time + 5:  # сброс фриз тайма
                    freeze_time = False
                    print('freeze_time_stop')
                    continue
                else:
                    continue

            status, frame = cap.read()
            if not status:
                redis_conn.delete(STOP_KEY)
                redis_conn.delete(PID_KEY)
                cap.release()
                break

            cropped_frame = frame[settings_y1:settings_y1 + height, settings_x1:settings_x1 + width]
            results = model.predict(cropped_frame, verbose=False)
            boxes = results[0].boxes
            if not boxes or len(boxes) > 1:  # сброс удержания человека в кадре
                capture_time = False
                continue

            if capture_time:
                if time.time() >= start_capture_time + 5:  # если человек в кадре в течении 5 секунд
                    print('capture_stop')
                    freeze_time = True
                    img_path = save_img(cropped_frame)
                    img_name = img_path.split('/')[-1]
                    redis_conn.publish(IMG_CHANNEL, img_name)
                    with sqlite3.connect('database.db') as con:
                        cur = con.cursor()
                        stmt = '''INSERT INTO humans (img_url, created_at, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?)'''
                        xyxy = boxes.xyxy.tolist()[0]
                        x1, y1, x2, y2 = map(int, xyxy)
                        cur.execute(stmt, (img_path, date.today().strftime('%Y-%m-%d'), x1, y1, x2, y2))
                        con.commit()
                    start_freeze_time = time.time()
                    capture_time = False
                    print('freeze_time_start')
                else:
                    # counter = round(time.time() - start_capture_time, 3)
                    # print(f'Capture time: {counter} seconds')
                    continue
            else:
                print('capture_start')
                capture_time = True
                start_capture_time = time.time()

    except Exception as e:
        logger.error(e)
        redis_conn.delete(STOP_KEY)
        redis_conn.delete(PID_KEY)


def is_process_alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


@app.get('/', response_class=HTMLResponse)
def index():
    with open('index.html', 'r', encoding='UTF-8') as f:
        return HTMLResponse(f.read())


@app.post('/start')
async def start():
    if redis_conn.get(STATUS_KEY) == '1':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Process already running')
    pid = redis_conn.get(PID_KEY)
    if pid:
        pid = int(pid)
        if is_process_alive(pid):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Process already running')
        else:
            redis_conn.delete(PID_KEY)
            redis_conn.delete(STOP_KEY)
            p = Process(target=camera_runtime)
            p.start()
    else:
        redis_conn.set(STATUS_KEY, 1)
        redis_conn.delete(STOP_KEY)
        p = Process(target=camera_runtime)
        p.start()


@app.post('/stop')
async def stop():
    if not redis_conn.get(PID_KEY):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Camera runtime already stopped')
    redis_conn.set(STOP_KEY, 1)
    redis_conn.delete(PID_KEY)
    redis_conn.delete(STATUS_KEY)


async def event_gen():
    pubsub = aio_redis_conn.pubsub()
    await pubsub.subscribe(IMG_CHANNEL)
    async for message in pubsub.listen():
        if message['type'] == 'message':
            img_name = message['data']
            yield f'data: {img_name}\n\n'
            await asyncio.sleep(1)


@app.get('/events')
async def events():
    return StreamingResponse(event_gen(), media_type='text/event-stream')


@app.get("/img/{image_name}")
async def get_image(image_name: str):
    img_path = IMG_FOLDER + '/' + image_name
    if not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


class ResponseURLSModel(BaseModel):
    image_urls: list[str] | None


@app.get('/humans', response_model=ResponseURLSModel)
async def get_humans(start_date: Annotated[str | None, Query(example='2025-04-04')],
                     end_date: Annotated[str | None, Query(example='2025-05-05')]
                     ):
    with sqlite3.connect('database.db') as con:
        cur = con.cursor()

        query = "SELECT img_url FROM humans"
        params = ()

        if start_date and end_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid start or end date')

            query += " WHERE created_at BETWEEN ? AND ?"
            params = (start_date, end_date)
        elif start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid start date')

            query += " WHERE created_at >= ?"
            params = (start_date,)
        elif end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid end date')
            query += " WHERE created_at <= ?"
            params = (end_date,)

        cur.execute(query, params)
        results = cur.fetchall()

        image_urls = [row[0] for row in results]

        return {'image_urls': image_urls}

