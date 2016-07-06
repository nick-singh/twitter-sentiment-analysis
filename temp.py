import csv, re, json


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def get_feature_list(wordlists):
	all_words = []
	for (w, s) in wordlists:
		all_words.extend(w)
	return list(set(all_words))


def processTweets(tweet):
	#process the tweet

	tweet = tweet.decode('utf-8','ignore').encode("utf-8")

	#Convert to lowercase
	tweet = tweet.lower()	

	#convert www.* or https?:// to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
	#Convert @username to AT_USER
	tweet = re.sub('@[^\s]+','',tweet)
	#remove additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
	#Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	#trim
	tweet = tweet.strip('\'"')

	#split into list of words
	tweet = tweet.split(" ")

	#remove all emoticons
	tweet = [s for s in tweet if not emoticon_re.search(s)]

	#joing list to make sentance
	tweet = " ".join(tweet)

	return tweet


tweet = "Hello , :) :( :-( :-D"
print processTweets(tweet)	