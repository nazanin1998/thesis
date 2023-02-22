class UserModel:
    def __init__(self, id, verified, name, friends_count, statuses_count, listed_count, followers_count,
                 follow_request_sent, notifications, created_at, time_zone,
                 default_profile_image, profile_text_color, favourites_count,
                 profile_use_background_image, profile_image_url_https,
                 profile_background_image_url_https, profile_background_tile, default_profile, location,
                 profile_sidebar_fill_color, protected,
                 description):
        self.id = id
        self.name = name
        self.location = location
        self.protected = protected
        self.verified = verified
        self.time_zone = time_zone
        self.created_at = created_at
        self.notifications = notifications
        self.statuses_count = statuses_count
        self.friends_count = friends_count
        self.favourites_count = favourites_count
        self.description = description
        self.default_profile = default_profile
        self.profile_background_tile = profile_background_tile
        self.listed_count = listed_count
        self.followers_count = followers_count
        self.profile_text_color = profile_text_color
        self.follow_request_sent = follow_request_sent
        self.default_profile_image = default_profile_image
        self.profile_image_url_https = profile_image_url_https
        self.profile_sidebar_fill_color = profile_sidebar_fill_color
        self.profile_use_background_image = profile_use_background_image
        self.profile_background_image_url_https = profile_background_image_url_https

    @staticmethod
    def from_json(js_obj):
        id = js_obj['id_str']
        name = js_obj['name']
        protected = js_obj['protected']
        notifications = js_obj['notifications']
        verified = js_obj['verified']
        time_zone = js_obj['time_zone']
        created_at = js_obj['created_at']
        description = js_obj['description']
        statuses_count = js_obj['statuses_count']
        friends_count = js_obj['friends_count']
        favourites_count = js_obj['favourites_count']
        listed_count = js_obj['listed_count']
        followers_count = js_obj['followers_count']
        location = js_obj['location']
        default_profile = js_obj['default_profile']
        profile_background_tile = js_obj['profile_background_tile']
        profile_text_color = js_obj['profile_text_color']
        follow_request_sent = js_obj['follow_request_sent']
        default_profile_image = js_obj['default_profile_image']
        profile_image_url_https = js_obj['profile_image_url_https']
        profile_sidebar_fill_color = js_obj['profile_sidebar_fill_color']
        profile_use_background_image = js_obj['profile_use_background_image']
        profile_background_image_url_https = js_obj['profile_background_image_url_https']

        return UserModel(id=id, verified=verified, protected=protected, default_profile=default_profile,
                         description=description,
                         statuses_count=statuses_count,
                         friends_count=friends_count, favourites_count=favourites_count, listed_count=listed_count,
                         followers_count=followers_count, created_at=created_at,
                         location=location, name=name, notifications=notifications,
                         profile_background_tile=profile_background_tile,
                         time_zone=time_zone,
                         profile_text_color=profile_text_color, follow_request_sent=follow_request_sent,
                         default_profile_image=default_profile_image, profile_image_url_https=profile_image_url_https,
                         profile_sidebar_fill_color=profile_sidebar_fill_color,
                         profile_use_background_image=profile_use_background_image,
                         profile_background_image_url_https=profile_background_image_url_https)
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
    #     "followers_count": 139847,
    #     "id_str": "465973",
    #     "listed_count": 3015,
    #     "is_translation_enabled": false,
    #     "statuses_count": 58363,
    #     "description": "https:\/\/t.co\/OEbchV0UNB | G+ http:\/\/t.co\/4ex2flvpsg",
    #     "friends_count": 30,
    #     "location": "Guido.Fawkes@Order-Order.com",
    #     "profile_image_url": "http:\/\/pbs.twimg.com\/profile_images\/588973879395229696\/2DXPltjM_normal.jpg",
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
#     "profile_sidebar_border_color": "FFFFFF",
#     "profile_background_color": "EDE6E6",
#     "utc_offset": null,
#     "profile_link_color": "FA0505",
#     "following": false,
#     "geo_enabled": true,
#     "lang": "en",


#     "profile_banner_url": "https:\/\/pbs.twimg.com\/profile_banners\/465973\/1413303571",
#     "profile_background_image_url": "http:\/\/pbs.twimg.com\/profile_background_images\/126781614\/pink-co-conspirators.jpg",
#     "name": "Guido Fawkes",
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
