import pymysql.cursors

def get_conn():
  conn = pymysql.connect(host='172.18.0.1', port = 53306,
                            user = 'mnist', password = '1234',
                            database = 'mnistdb',
                            cursorclass=pymysql.cursors.DictCursor)
  return conn


def select(query: str, size = -1):
  conn = get_conn()
  with conn:
      with conn.cursor() as cursor:
          cursor.execute(query)
          result = cursor.fetchmany(size)

  return result


def dml(sql, *values):
  conn = get_conn()

  with conn:
    with conn.cursor() as cursor:
        cursor.execute(sql, values)
        conn.commit()
        return cursor.rowcount
