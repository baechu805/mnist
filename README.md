# mnist

### Run
```
$ uvicorn src.mnist.main:app --reload
```

### DB
```bash
sudo docker run -d \
        --name mnist-mariadb \
        -e MARIADB_USER=mnist \
        --env MARIADB_PASSWORD=1234 \
        --env MARIADB_DATABASE=mnistdb \
        --env MARIADB_ROOT_PASSWORD=my-secret-pw \
        -p 53306:3306 \
        mariadb:latest
```

### Table
```
$ sudo docker exec -it mnist-mariadb bash
root@5d871c9bab8f:/# mariadb -u mnist -p1234
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 3
Server version: 11.5.2-MariaDB-ubu2404 mariadb.org binary distribution

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mnistdb            |
+--------------------+
2 rows in set (0.012 sec)

MariaDB [(none)]> use mnistdb;
Database changed
MariaDB [mnistdb]> CREATE TABLE image_processing (
    -> num INT AUTO_INCREMENT PRIMARY KEY COMMENT '자동 증가 숫자',
    -> file_name VARCHAR(100) NOT NULL COMMENT '원본 파일명',
    -> file_path VARCHAR(255) NOT NULL COMMENT '저장 전체 경로 및 변환 파일명',
    -> request_time VARCHAR(50) NOT NULL COMMENT '요청시간',
    -> request_user VARCHAR(50) NOT NULL COMMENT '요청 사용자',
    -> prediction_model VARCHAR(100) COMMENT '예측 사용 모델',
    -> prediction_result VARCHAR(50) COMMENT '예측 결과',
    -> prediction_time VARCHAR(50) COMMENT '예측 시간'
    -> )
    -> ;
Query OK, 0 rows affected (0.033 sec)

MariaDB [mnistdb]> desc image_processing;
+-------------------+--------------+------+-----+---------+----------------+
| Field             | Type         | Null | Key | Default | Extra          |
+-------------------+--------------+------+-----+---------+----------------+
| num               | int(11)      | NO   | PRI | NULL    | auto_increment |
| file_name         | varchar(100) | NO   |     | NULL    |                |
| file_path         | varchar(255) | NO   |     | NULL    |                |
| request_time      | varchar(50)  | NO   |     | NULL    |                |
| request_user      | varchar(50)  | NO   |     | NULL    |                |
| prediction_model  | varchar(100) | YES  |     | NULL    |                |
| prediction_result | varchar(50)  | YES  |     | NULL    |                |
| prediction_time   | varchar(50)  | YES  |     | NULL    |                |
+-------------------+--------------+------+-----+---------+----------------+
8 rows in set (0.013 sec)

MariaDB [mnistdb]> SELECT * FROM image_processing;
Empty set (0.004 sec)

MariaDB [mnistdb]>
```

### CRUD
```sql
INSERT INTO image_processing(file_name, file_path,
request_time, request_user)
VALUES('123.png', '/a/b/c/123.png', '2024-09-20 10:20:11', 'n99');

UPDATE image_processing  
SET prediction_model = 'n00', prediction_result='5', prediction_time='2024-09-20 11:11:11' 
WHERE num = 1;

SELECT * FROM image_processing WHERE num = 1;

DELETE FROM image_processing WHERE num = 1;
```
