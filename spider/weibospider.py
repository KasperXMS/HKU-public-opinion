import re

import requests
import json
from bs4 import BeautifulSoup
import pymysql
import time

def req1():
    url = "http://passport.weibo.com/visitor/genvisitor"
    params = {
        'cb': 'gen_callback',
        'fp': '%7B%22os%22%3A%221%22%2C%22browser%22%3A%22Chrome85%2C0%2C4183%2C121%22%2C%22fonts%22%3A%22undefined%22%'
              '2C%22screenInfo%22%3A%221536*864*24%22%2C%22plugins%22%3A%22Portable%20Document%20Format%3A%3Ainternal-'
              'pdf-viewer%3A%3AChromium%20PDF%20Plugin%7C%3A%3Amhjfbmdgcfjbbpaeojofohoefgiehjai%3A%3AChromium%20PDF%20V'
              'iewer%7C%3A%3Ainternal-nacl-plugin%3A%3ANative%20Client%22%7D'
    }
    r = requests.get(url=url, params=params)
    response_json = r.text[36:len(r.text)-2]
    response1 = json.loads(response_json)
    return response1['data']['tid']

def req2(tid):
    url = "http://passport.weibo.com/visitor/visitor"
    params = {
        'a': 'incarnate',
        't': tid,
        'cb': 'cross_domain',
        'from': 'weibo'
    }
    r = requests.get(url=url, params=params)
    response_json = r.text[36:len(r.text)-2]
    response1 = json.loads(response_json)
    return response1['data']['sub'], response1['data']['subp']

def req3(sub, subp, keyword):
    url = 'https://s.weibo.com/realtime'
    params = {
        'q': keyword,
        'rd': 'realtime',
        'tw': 'realtime',
        'Refer': 'weibo_realtime'
    }
    cookies = {
        'SUB': sub,
        'SUBP': subp
    }
    r = requests.get(url=url, params=params, cookies=cookies)
    return r.text

def get_user_location(user_url):
    tid = req1()
    sub, subp = req2(tid)
    url = 'https://weibo.com/ajax/profile/detail?uid=' + re.compile(r'/\d+\?').findall(user_url)[0][1:-1]
    cookies = {
        'SUB': sub,
        'SUBP': subp
    }
    r = requests.get(url=url, cookies=cookies)
    r = json.loads(r.text)
    return r['data']['location']

def insertToDB(url, text, comments, likes, location, keyword):
    db = pymysql.connect(host='hostname', user='username', password='password', database='database')
    cursor = db.cursor()
    query = "SELECT * FROM weibo WHERE mid='{}'".format(url)
    cursor.execute(query)
    res = cursor.fetchall()
    if len(res) == 0:
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        insert = "INSERT INTO weibo (mid, text, comments, likes, time, location, keyword) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(url, text, comments, likes, currentTime, location, keyword)
        print(insert)
        try:
            cursor.execute(insert)
            db.commit()
        except:
            print("insert error!")
            db.rollback()
    else:
        update = "UPDATE weibo SET text='{}', comments={}, likes={}, location='{}' WHERE mid='{}'".format(text, comments, likes, location, url)
        print(update)
        try:
            cursor.execute(update)
            db.commit()
        except:
            print("update error!")
            db.rollback()

    db.close()

def resolve2upload(raw_html, keyword):
    soup = BeautifulSoup(raw_html, 'html.parser')
    res = soup.select('#pl_feedlist_index > div > div > div > div.card-feed > div.content')
    res1 = soup.select('#pl_feedlist_index > div > div')
    res2 = soup.select('#pl_feedlist_index > div > div > div > div.card-act > ul > li > a > button > span.woo-like-count')
    res3 = soup.select('#pl_feedlist_index > div > div > div > div.card-act > ul > li:nth-child(2) > a')
    res4 = soup.select('#pl_feedlist_index > div > div > div > div.card-feed > div.avator > a')
    for i in range(0, len(res1)):
        # print(res[i])
        mid = res1[i]['mid']
        # if href[0:5] == 'https' and True:
        print(mid)
        text = ''
        max_text_len = -1
        for element in res[i].find_all('p'):
            if len(element.get_text()) > max_text_len:
                text = element.get_text()
                max_text_len = len(element.get_text())

        text = text.strip()
        print(text)
        comments = res3[i].get_text().replace('评论', '0')
        likes = res2[i].get_text().replace('赞', '0')
        print("comments: {}, likes: {}".format(comments, likes))
        user_profile = 'https:' + res4[i]['href']
        location = get_user_location(user_profile)
        insertToDB(mid, text, comments, likes, location, keyword)

def weibo(keyword):
    tid = req1()
    sub, subp = req2(tid)
    raw_html = req3(sub, subp, keyword)
    resolve2upload(raw_html, keyword)

if __name__ == "__main__":
    f = open('keyword_cn.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.replace('\n', '')
        weibo(line)




