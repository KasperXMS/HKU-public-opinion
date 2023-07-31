import json
import time
import requests


def retrieve(text):
    url = "https://twitter.com/i/api/graphql/nK1dw4oV3k4w5TdtcAdSww/SearchTimeline?"
    data = {'variables': '{"rawQuery":"' + text + '","count":20,"querySource":"recent_search_click","product":"Latest"}',
            'features': '{"rweb_lists_timeline_redesign_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":false,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_media_download_video_enabled":false,"responsive_web_enhance_cards_enabled":false}'
            }
    cookies = {'auth_token': "c92b4d781fa6c9bc0ebc1de5175697aa2addb7b6",
               'ct0': "c72cb6b82a5cfd61eb5c3879c9e7696f200a55f5a90c05ca0e0a0523601c7067914a5da5716d0813c724be5b2bac7c23c06735bc047c6c72e7843bcf3115aca2c6872722c518148194df90084d142011"}
    headers = {
        "X-Csrf-Token": "c72cb6b82a5cfd61eb5c3879c9e7696f200a55f5a90c05ca0e0a0523601c7067914a5da5716d0813c724be5b2bac7c23c06735bc047c6c72e7843bcf3115aca2c6872722c518148194df90084d142011",
        "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs=1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.127 Safari/537.36",
        "X-Twitter-Auth-Type": "OAuth2Session",
        "X-Twitter-Active-User": "yes",
    }

    result = []
    response = requests.get(url=url, headers=headers, cookies=cookies, data=data)
    response_object = json.loads(response.text)
    entries = response_object['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]['entries']
    for (i, entry) in enumerate(entries):
        if entry['entryId'].startswith('tweet'):
            tweet_info = entry['content']['itemContent']['tweet_results']['result']['legacy']
            user_location = entry['content']['itemContent']['tweet_results']['result']['core']['user_results']['result']['legacy']['location']
            result.append([entry['entryId'][6:], tweet_info['created_at'], tweet_info['full_text'].replace('\"', '\''),
                           tweet_info['retweet_count'], tweet_info['favorite_count'], tweet_info['reply_count'],
                           tweet_info['quote_count'], tweet_info['lang'].replace('\n', ' '),
                           user_location])

    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " - retrieved a patch of tweets, keyword: ", text,
    #       "; size: ", len(result))
    return result


if __name__ == '__main__':
    print(retrieve('University of Hong Kong'))
