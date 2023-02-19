import numpy

from lib.models.user import User

# is_rumour , thread, in_reply_tweet, event, tweet_id, is_source_tweet, in_reply_user, user_id, tweet_length,
# symbol_count, user_mentions, urls_count, media_count, hashtags_count, retweet_count
# favorite_count, mentions_count, is_truncated, created, has_smile_emoji, sensitive
# has_place, has_coords, has_quest, has_exclaim, has_quest_or_exclaim, user.tweets_count
# user.verified, user.followers_count, user.listed_count, user.desc_length, user.handle_length,
# user.name_length, user.notifications, user.friends_count, user.time_zone, user.has_bg_img,
# user.default_pic, user.created_at, user.location, user.profile_sbcolor, user.profile_bgcolor,
# user.utc_dist, hasperiod, number_punct, negativewordcount, positivewordcount, capitalratio,
# contentlength, sentimentscore, Noun, has_url_in_text
#

#  thread
# media_count,
# has_smile_emoji, sensitive
# has_place, has_coords, has_quest, has_exclaim, has_quest_or_exclaim,
#  user.handle_length,
# user.has_bg_img,
# user.default_pic, user.profile_sbcolor, user.profile_bgcolor,
# user.utc_dist, hasperiod, number_punct, negativewordcount, positivewordcount, capitalratio,
# contentlength, sentimentscore, Noun, has_url_in_text
#
from lib.read_datasets.pheme.file_dir_handler import FileDirHandler


class ReactionCsvModel:
    def to_json(self, is_rumour, event, is_source_tweet, reaction_text):
        user = self.user
        text = self.text
        user.verified = self.__normalize_boolean(user.verified)
        user.notifications = self.__normalize_boolean(user.notifications)
        user.protected = self.__normalize_boolean(user.protected)
        user.default_profile = self.__normalize_boolean(user.default_profile)
        user.default_profile_image = self.__normalize_boolean(user.default_profile_image)
        user.follow_request_sent = self.__normalize_boolean(user.follow_request_sent)
        user.profile_background_tile = self.__normalize_boolean(user.profile_background_tile)
        user.profile_use_background_image = self.__normalize_boolean(user.profile_use_background_image)
        self.truncated = self.__normalize_boolean(self.truncated)
        return {
            'event': event,
            'text': text,
            'reaction_text': reaction_text,
            'is_rumour': is_rumour,
            'tweet_id': self.id,
            'tweet_length': len(text),
            'created': self.created_at,
            'is_truncated': self.truncated,
            'symbol_count': len(self.symbols),
            'user_mentions': len(self.user_mentions),
            'mentions_count': len(self.user_mentions),
            'urls_count': len(self.urls),
            'is_source_tweet': is_source_tweet,
            'retweet_count': self.retweet_count,
            'favorite_count': self.favorite_count,
            'hashtags_count': len(self.hashtags),
            'in_reply_user_id': self.in_reply_to_user_id,
            'in_reply_tweet_id': self.in_reply_to_status_id,
            'media': self.media,
            'hashtags': self.hashtags,

            'user.id': user.id,
            'user.name': user.name,
            'user.verified': user.verified,
            'user.protected': user.protected,
            'user.name_length': len(user.name),
            'user.location': user.location,
            'user.description': user.description,
            'user.time_zone': user.time_zone,
            'user.created_at': user.created_at,
            'user.listed_count': user.listed_count,
            'user.default_profile': user.default_profile,
            'user.tweets_count': user.statuses_count,
            'user.statuses_count': user.statuses_count,
            'user.friends_count': user.friends_count,
            'user.favourites_count': user.favourites_count,
            'user.followers_count': user.followers_count,
            'user.notifications': user.notifications,
            'user.profile_text_color': user.profile_text_color,
            'user.follow_request_sent': user.follow_request_sent,
            'user.default_profile_image': user.default_profile_image,
            'user.profile_background_tile': user.profile_background_tile,
            'user.profile_image_url_https': user.profile_image_url_https,
            'user.profile_sidebar_fill_color': user.profile_sidebar_fill_color,
            'user.profile_use_background_image': user.profile_use_background_image,
            'user.profile_background_image_url_https': user.profile_background_image_url_https

        }

    def __init__(self, id, source, truncated, favorited, retweeted, in_reply_to_user_id, retweet_count,
                 contributors,
                 text, media,
                 in_reply_to_status_id, urls, hashtags, user_mentions, created_at, favorite_count, user, symbols):

        self.id = id
        self.text = text
        self.user = user
        self.source = source
        self.favorited = favorited
        self.symbols = symbols
        self.truncated = truncated
        self.user_mentions = user_mentions
        self.hashtags = hashtags
        self.urls = urls
        self.media = media
        self.retweeted = retweeted
        self.created_at = created_at
        self.retweet_count = retweet_count
        self.contributors = contributors
        self.favorite_count = favorite_count
        self.in_reply_to_user_id = in_reply_to_user_id
        self.in_reply_to_status_id = in_reply_to_status_id

    @staticmethod
    def from_json(js_obj):
        id = js_obj['id_str']
        text = js_obj['text']
        source = js_obj['source']
        truncated = js_obj['truncated']
        user = User.from_json(js_obj['user'])
        entities = js_obj['entities']
        urls = entities['urls']
        symbols = entities['symbols']
        hashtags = numpy.NaN
        try:
            hashtags = Tweet.__extract_hashtag_str(hashtag_list=entities['hashtags'])
        except:
            pass
        media = numpy.NaN
        try:
            media = Tweet.__extract_media_str(media_list=entities['media'])
        except:
            pass
        user_mentions = entities['user_mentions']
        favorited = js_obj['favorited']
        retweeted = js_obj['retweeted']
        created_at = js_obj['created_at']
        retweet_count = js_obj['retweet_count']
        contributors = js_obj['contributors']
        favorite_count = js_obj['favorite_count']
        in_reply_to_user_id = js_obj['in_reply_to_user_id']
        in_reply_to_status_id = js_obj['in_reply_to_status_id']

        return Tweet(id=id, symbols=symbols, user_mentions=user_mentions, hashtags=hashtags, urls=urls, text=text,
                     source=source, truncated=truncated,
                     user=user, favorited=favorited,
                     retweeted=retweeted,
                     media=media,
                     created_at=created_at, retweet_count=retweet_count, contributors=contributors,
                     favorite_count=favorite_count, in_reply_to_user_id=in_reply_to_user_id,
                     in_reply_to_status_id=in_reply_to_status_id)

    @staticmethod
    def __extract_media_str(media_list):
        media_str = ''
        for media in media_list:
            if media['type'] == 'photo':
                if media_str == '':
                    media_str = media['media_url_https']
                else:
                    media_str = media_str + ',' + media['media_url_https']
        return media_str

    @staticmethod
    def __extract_hashtag_str(hashtag_list):
        hashtag_str = ''
        for hashtag in hashtag_list:
            if hashtag_str == '':
                hashtag_str = hashtag['text']
            else:
                hashtag_str = hashtag_str + ',' + hashtag['text']
        return hashtag_str

    @staticmethod
    def __normalize_boolean(input_bool):
        if input_bool:
            return 0
        elif not input_bool:
            return 1
        if input_bool == 'FALSE':
            return 1
        elif input_bool == 'TRUE':
            return 0

    @staticmethod
    def tweet_file_to_obj(path):
        tweet_json_obj = FileDirHandler.read_json_file(path=path)
        if tweet_json_obj is None:
            return None
        return Tweet.from_json(tweet_json_obj)
    #
    # {
    #   "contributors": null,
    #   "truncated": false,
    #   "text": "Charlie Hebdo\u2019s Last Tweet Before Shootings http:\/\/t.co\/9Oa2xAqOcM http:\/\/t.co\/skJHNEQcn0",
    #   "in_reply_to_status_id": null,
    #   "id": 552784898743099392,
    #   "favorite_count": 20,
    #   "source": "<a href=\"https:\/\/about.twitter.com\/products\/tweetdeck\" rel=\"nofollow\">TweetDeck<\/a>",
    #   "retweeted": false,
    #   "coordinates": null,
    #   "entities": {
    #     "symbols": [],
    #     "user_mentions": [],
    #     "hashtags": [],
    #     "urls": [
#       {
#         "url": "http:\/\/t.co\/9Oa2xAqOcM",
#         "indices": [
#           44,
#           66
#         ],
#         "expanded_url": "http:\/\/order-order.com\/2015\/01\/07\/charlie-hebdos-last-tweet-before-shootings\/",
#         "display_url": "order-order.com\/2015\/01\/07\/cha\u2026"
#       }
#     ],
#     "media": [
#       {
#         "expanded_url": "http:\/\/twitter.com\/GuidoFawkes\/status\/552784898743099392\/photo\/1",
#         "display_url": "pic.twitter.com\/skJHNEQcn0",
#         "url": "http:\/\/t.co\/skJHNEQcn0",
#         "media_url_https": "https:\/\/pbs.twimg.com\/media\/B6vi4x3CYAABfFl.jpg",
#         "id_str": "552784844367749120",
#         "sizes": {
#           "small": {
#             "h": 343,
#             "resize": "fit",
#             "w": 340
#           },
#           "large": {
#             "h": 505,
#             "resize": "fit",
#             "w": 500
#           },
#           "medium": {
#             "h": 505,
#             "resize": "fit",
#             "w": 500
#           },
#           "thumb": {
#             "h": 150,
#             "resize": "crop",
#             "w": 150
#           }
#         },
#         "indices": [
#           67,
#           89
#         ],
#         "type": "photo",
#         "id": 552784844367749120,
#         "media_url": "http:\/\/pbs.twimg.com\/media\/B6vi4x3CYAABfFl.jpg"
#       }
#     ]
#   },
#   "in_reply_to_screen_name": null,
#   "id_str": "552784898743099392",
#   "retweet_count": 144,
#   "in_reply_to_user_id": null,
#   "favorited": false,
#   "user": {
#     "follow_request_sent": false,
#     "profile_use_background_image": true,
#     "profile_text_color": "451EC7",
#     "default_profile_image": false,
#     "id": 465973,
#     "profile_background_image_url_https": "https:\/\/pbs.twimg.com\/profile_background_images\/126781614\/pink-co-conspirators.jpg",
#     "verified": true,
#     "profile_location": null,
#     "profile_image_url_https": "https:\/\/pbs.twimg.com\/profile_images\/588973879395229696\/2DXPltjM_normal.jpg",
#     "profile_sidebar_fill_color": "0A0909",
#     "entities": {
#       "url": {
#         "urls": [
#           {
#             "url": "http:\/\/t.co\/2tQYEIO1cg",
#             "indices": [
#               0,
#               22
#             ],
#             "expanded_url": "http:\/\/order-order.com",
#             "display_url": "order-order.com"
#           }
#         ]
#       },
#       "description": {
#         "urls": [
#           {
#             "url": "https:\/\/t.co\/OEbchV0UNB",
#             "indices": [
#               0,
#               23
#             ],
#             "expanded_url": "https:\/\/www.facebook.com\/fawkespage",
#             "display_url": "facebook.com\/fawkespage"
#           },
#           {
#             "url": "http:\/\/t.co\/4ex2flvpsg",
#             "indices": [
#               29,
#               51
#             ],
#             "expanded_url": "http:\/\/guyfawk.es\/1eVzH8o",
#             "display_url": "guyfawk.es\/1eVzH8o"
#           }
#         ]
#       }
#     },
#     "followers_count": 139847,
#     "profile_sidebar_border_color": "FFFFFF",
#     "id_str": "465973",
#     "profile_background_color": "EDE6E6",
#     "listed_count": 3015,
#     "is_translation_enabled": false,
#     "utc_offset": null,
#     "statuses_count": 58363,
#     "description": "https:\/\/t.co\/OEbchV0UNB | G+ http:\/\/t.co\/4ex2flvpsg",
#     "friends_count": 30,
#     "location": "Guido.Fawkes@Order-Order.com",
#     "profile_link_color": "FA0505",
#     "profile_image_url": "http:\/\/pbs.twimg.com\/profile_images\/588973879395229696\/2DXPltjM_normal.jpg",
#     "following": false,
#     "geo_enabled": true,
#     "profile_banner_url": "https:\/\/pbs.twimg.com\/profile_banners\/465973\/1413303571",
#     "profile_background_image_url": "http:\/\/pbs.twimg.com\/profile_background_images\/126781614\/pink-co-conspirators.jpg",
#     "name": "Guido Fawkes",
#     "lang": "en",
#     "profile_background_tile": true,
#     "favourites_count": 216,
#     "screen_name": "GuidoFawkes",
#     "notifications": false,
#     "url": "http:\/\/t.co\/2tQYEIO1cg",
#     "created_at": "Tue Jan 02 19:22:19 +0000 2007",
#     "contributors_enabled": false,
#     "time_zone": null,
#     "protected": false,
#     "default_profile": false,
#     "is_translator": false
#   },
#   "geo": null,
#   "in_reply_to_user_id_str": null,
#   "possibly_sensitive": false,
#   "lang": "en",
#   "created_at": "Wed Jan 07 11:12:44 +0000 2015",
#   "in_reply_to_status_id_str": null,
#   "place": null,
#   "extended_entities": {
#     "media": [
#       {
#         "expanded_url": "http:\/\/twitter.com\/GuidoFawkes\/status\/552784898743099392\/photo\/1",
#         "display_url": "pic.twitter.com\/skJHNEQcn0",
#         "url": "http:\/\/t.co\/skJHNEQcn0",
#         "media_url_https": "https:\/\/pbs.twimg.com\/media\/B6vi4x3CYAABfFl.jpg",
#         "id_str": "552784844367749120",
#         "sizes": {
#           "small": {
#             "h": 343,
#             "resize": "fit",
#             "w": 340
#           },
#           "large": {
#             "h": 505,
#             "resize": "fit",
#             "w": 500
#           },
#           "medium": {
#             "h": 505,
#             "resize": "fit",
#             "w": 500
#           },
#           "thumb": {
#             "h": 150,
#             "resize": "crop",
#             "w": 150
#           }
#         },
#         "indices": [
#           67,
#           89
#         ],
#         "type": "photo",
#         "id": 552784844367749120,
#         "media_url": "http:\/\/pbs.twimg.com\/media\/B6vi4x3CYAABfFl.jpg"
#       }
#     ]
#   }
# }
