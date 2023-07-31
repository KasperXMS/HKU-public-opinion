# HKU Project: A Platform For HKU Related Public Opinions Monitoring And Analysis

With the development of the Internet, the analysis of public opinion has become particularly important. People also judge companies or organizations based on public opinion. As students of the University of Hong Kong (HKU), we would like to help the university to monitor public opinion on the Internet in real time. We designed a platform for public opinion monitoring, whose main functions include sentiment analysis of HKU-related comments in Chinese and English, statistics of public opinion data, and geographic distribution of commenters. We collected HKU related comments on twitter and Weibo as our dataset and built a database. In terms of sentiment analysis models, we trained various models, and finally chose RoBERTa and RoBERTa WWM as the English and Chinese sentiment analysis models respectively, and they achieved an accuracy of 93.6 and 86.3 in the English and Chinese test sets, which is able to analyze the public opinion of the University of Hong Kong very well.



### File structure

`hku-senti-react`: Source code of front-end page to present visualized data.

`java`: Source code of back-end server, providing data API.

`project-english-models`: Source code of training and evaluation of our English sentiment analysis model.

`sentiment_cn`: Source code of training and evaluation of our Chinese sentiment analysis model.

`spider`: Source code of Weibo and Twitter spider.



### Usage

See the README file in each sub-directory.

### 