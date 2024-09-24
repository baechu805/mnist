import jigeum.seoul
from mnist.db import select, dml
import random
import os
import requests

def get_job_img_task():
   sql = """
   SELECT 
    num, file_name, file_path
   FROM image_processing
   WHERE prediction_result IS NULL
   ORDER BY num -- 가장 오래된 요청
   LIMIT 1 -- 하나씩
   """
   r = select(sql, 1)
   
   if len(r) > 0:
       return r[0]
   else:
       return None

def prediction(file_path, num):
    sql = """UPDATE image_processing
    SET prediction_result=%s,
        prediction_model='n72',
        prediction_time=%s
    WHERE num=%s
    """
    presult = random.randint(0, 9)
    dml(sql, presult, jigeum.seoul.now(), num)

    return presult


def run():
  """image_processing 테이블을 읽어서 가장 오래된 요청 하나씩을 처리"""
  
  # STEP 1
  # image_processing 테이블의 prediction_result IS NULL 인 ROW 1 개 조회 - num 갖여오기
  job = get_job_img_task()
  
  if job is None:
      print(f"{jigeum.seoul.now()} - job is None")
      return

  num = job['num']
  file_name = job['file_name']
  file_path = job['file_path']

  # STEP 2
  # RANDOM 으로 0 ~ 9 중 하나 값을 prediction_result 컬럼에 업데이트
  # 동시에 prediction_model, prediction_time 도 업데이트
  presutl = prediction(file_path, num)

  # STEP 3
  # LINE 으로 처리 결과 전송
  send_line_noti(file_name, presutl)

  print(jigeum.seoul.now())

def send_line_noti(file_name, presult):

    KEY = os.environ.get('API_TOKEN')
    url = "https://notify-api.line.me/api/notify"
    data = {"message": "성공적으로 저장했습니다!"}
    headers = {"Authorization": "Bearer " + KEY}
    response = requests.post(url, data=data, headers=headers)
    print(response.text)
    print("SEND LINE NOTI")
