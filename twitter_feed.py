from tweepy import Stream
from tweepy import OAuthHandler
import json



#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

class listener(Stream):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]

        print(tweet)

        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])