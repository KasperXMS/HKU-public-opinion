import pymysql
import time

db = pymysql.connect(host='host name', port=3306, user='user name', password='password', database='database name')
cursor = db.cursor()
query = """SELECT * FROM tweet_raw WHERE keyword="University of Hong Kong" and (text LIKE "%HKU %" or text LIKE "%University of Hong Kong %") and (text NOT LIKE "%Chinese University of Hong Kong %" and text NOT LIKE "%City University of Hong Kong %" and text NOT LIKE "%Education University of Hong Kong %" and text NOT LIKE "%Hang Seng University of Hong Kong %") ORDER BY time DESC  LIMIT 0, 50"""
cursor.execute(query)
res = cursor.fetchall()
print("Total of ", len(res), " data")
for row in res:
    print(row)
    sql = "INSERT INTO tweet_hku(tweet_id, time, text, retweet_count, favorite_count, reply_count, quote_count, language, location, keyword, positive, neutral, negative) VALUES ('{}', '{}', \"{}\", {}, {}, {}, {}, '{}', '{}', '{}', {}, {}, {})".format(
        row[0], row[1].strftime("%Y-%m-%d %H:%M:%S"), row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12])
    try:
        cursor.execute(sql)
        db.commit()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert success for tweet ', row[0])
    except:
        db.rollback()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert FAILED for tweet ', row[0],
              '; SQL: ', sql)

        sql = "UPDATE tweet_hku SET retweet_count={}, favorite_count={}, reply_count={}, quote_count={} WHERE tweet_id='{}'".format(
            row[3], row[4], row[5], row[6],  row[0])
        try:
            cursor.execute(sql)
            db.commit()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Update success for tweet ', row[0])
        except:
            db.rollback()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert FAILED for tweet ', row[0],
                  '; SQL: ', sql)

db.close()
