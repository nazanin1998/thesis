import json
from copy import deepcopy
import pandas


def cross_join(left, right):
    new_rows = [] if right else left
    for left_row in left:
        for right_row in right:
            temp_row = deepcopy(left_row)
            for key, value in right_row.items():
                temp_row[key] = value
            new_rows.append(deepcopy(temp_row))
    return new_rows


def flatten_list(data):
    for elem in data:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


def json_to_dataframe(data_in):
    def flatten_json(data, prev_heading=''):
        if isinstance(data, dict):
            rows = [{}]
            for key, value in data.items():
                rows = cross_join(rows, flatten_json(value, prev_heading + '.' + key))
        elif isinstance(data, list):
            rows = []
            for item in data:
                [rows.append(elem) for elem in flatten_list(flatten_json(item, prev_heading))]
        else:
            rows = [{prev_heading[1:]: data}]
        return rows

    return pandas.DataFrame(flatten_json(data_in))


if __name__ == '__main__':
    json_data = """{
  "contributors": None,
  "truncated": False,
  "text": "Charlie Hebdo became well known for publishing the Muhammed cartoons two years ago",
  "in_reply_to_status_id": null,
  "id": 552784600502915072,
  "favorite_count": 41,
  "source": "<a href=\"http:\/\/twitter.com\" rel=\"nofollow\">Twitter Web Client<\/a>",
  "retweeted": false,
  "coordinates": null,
  "entities": {
    "symbols": [],
    "user_mentions": [],
    "hashtags": [],
    "urls": []
  },
  "in_reply_to_screen_name": null,
  "id_str": "552784600502915072",
  "retweet_count": 202,
  "in_reply_to_user_id": null,
  "favorited": false,
  "user": {
    "follow_request_sent": false,
    "profile_use_background_image": true,
    "profile_text_color": "5A5A5A",
    "default_profile_image": false,
    "id": 331658004,
    "profile_background_image_url_https": "https:\/\/pbs.twimg.com\/profile_background_images\/337316083\/bbc_twitter_template1280.jpg",
    "verified": true,
    "profile_location": null,
    "profile_image_url_https": "https:\/\/pbs.twimg.com\/profile_images\/1497949200\/DanielSandfordSmall_normal.jpg",
    "profile_sidebar_fill_color": "FFFFFF",
    "entities": {
      "url": {
        "urls": [
          {
            "url": "http:\/\/t.co\/tPNR3GoVZJ",
            "indices": [
              0,
              22
            ],
            "expanded_url": "http:\/\/news.bbc.co.uk",
            "display_url": "news.bbc.co.uk"
          }
        ]
      },
      "description": {
        "urls": []
      }
    },
    "followers_count": 41591,
    "profile_sidebar_border_color": "CCCCCC",
    "id_str": "331658004",
    "profile_background_color": "FFFFFF",
    "listed_count": 1657,
    "is_translation_enabled": false,
    "utc_offset": 14400,
    "statuses_count": 15128,
    "description": "I am Home Affairs Correspondent for BBC News. Police, prisons, law, crime and terrorism. Before that Moscow Correspondent so still tweet about Russia\/Ukraine.",
    "friends_count": 2268,
    "location": "",
    "profile_link_color": "1F527B",
    "profile_image_url": "http:\/\/pbs.twimg.com\/profile_images\/1497949200\/DanielSandfordSmall_normal.jpg",
    "following": false,
    "geo_enabled": true,
    "profile_banner_url": "https:\/\/pbs.twimg.com\/profile_banners\/331658004\/1360223450",
    "profile_background_image_url": "http:\/\/pbs.twimg.com\/profile_background_images\/337316083\/bbc_twitter_template1280.jpg",
    "name": "Daniel Sandford",
    "lang": "en",
    "profile_background_tile": false,
    "favourites_count": 0,
    "screen_name": "BBCDanielS",
    "notifications": false,
    "url": "http:\/\/t.co\/tPNR3GoVZJ",
    "created_at": "Fri Jul 08 14:32:54 +0000 2011",
    "contributors_enabled": false,
    "time_zone": "Moscow",
    "protected": false,
    "default_profile": false,
    "is_translator": false
  },
  "geo": null,
  "in_reply_to_user_id_str": null,
  "lang": "en",
  "created_at": "Wed Jan 07 11:11:33 +0000 2015",
  "in_reply_to_status_id_str": null,
  "place": null
}"""
    data = json.loads(json_data)
    df = json_to_dataframe(data)

    print(df.to_csv())
