from twitter_search import retrieve
from insert_to_db import insert_tweet

f = open('keywords.txt', 'r')
keywords = f.readlines()
f.close()

for keyword in keywords:
    keyword = keyword.replace('\n', '')
    raws = retrieve(keyword)
    for raw in raws:
        insert_tweet(raw, keyword)
