import pymysql
import time

def insert_tweet(tweet_info, keyword):
    db = pymysql.connect(host='host name', port=3306, user='user name', password='password', database='database name')
    cursor = db.cursor()
    query = "SELECT * FROM tweet WHERE tweet_id='{}'".format(tweet_info[0])
    cursor.execute(query)
    res = cursor.fetchall()
    if len(res) == 0:
        sql = "INSERT INTO tweet(tweet_id, time, text, retweet_count, favorite_count, reply_count, quote_count, language, location, keyword) VALUES ('{}', '{}', \"{}\", {}, {}, {}, {}, '{}', '{}', '{}')".format(
            tweet_info[0], tweet_time_convert(tweet_info[1]), tweet_info[2], tweet_info[3], tweet_info[4],
            tweet_info[5], tweet_info[6], tweet_info[7], tweet_info[8], keyword)
        try:
            cursor.execute(sql)
            db.commit()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert success for tweet ', tweet_info[0])
        except:
            db.rollback()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert FAILED for tweet ', tweet_info[0], '; SQL: ', sql)

    else:
        sql = "UPDATE tweet SET retweet_count={}, favorite_count={}, reply_count={}, quote_count={}, location='{}' WHERE tweet_id='{}'".format(
            tweet_info[3], tweet_info[4], tweet_info[5], tweet_info[6], tweet_info[8], tweet_info[0])
        try:
            cursor.execute(sql)
            db.commit()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Update success for tweet ', tweet_info[0])
        except:
            db.rollback()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ' - Insert FAILED for tweet ', tweet_info[0], '; SQL: ', sql)

    db.close()

def tweet_time_convert(raw_time):
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    year = raw_time.split(' ')[-1]
    month = months[raw_time.split(' ')[1]]
    day = raw_time.split(' ')[2]
    times = raw_time.split(' ')[3]
    return year + '-' + month + '-' + day + ' ' + times