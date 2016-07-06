import csv, re, json
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet as wn 
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from multiprocessing import Process, Queue, Pool, cpu_count



stop_words = set(stopwords.words('english'))
tknizr = TweetTokenizer()
# lmz = WordNetLemmatizer()

pos_data_file = "datasets/pos-800000.csv"
neg_data_file = "datasets/neg-800000.csv"
test = "datasets/testdata.manual.2009.06.14.csv"

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

	return tweet

def replaceTowOrMore(word):
	#look for 2 ore more repetitions of character and replace with the character itself
	pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
	return pattern.sub(r"\1\1", word)



def processed(row):		
	# clean the tweets
	clean_tweet = processTweets(row[5])

	# convert tweet into word list
	tknized_string = tknizr.tokenize(clean_tweet)

	# remove all non alpha charactrs and normalize word
	filtered_tweet = [replaceTowOrMore(w) for w in tknized_string if not w in stop_words and w.isalpha()]
	
	print (filtered_tweet, row[0])
	return (filtered_tweet, row[0])


def process_all_tweets(data_file):
	print "Reading file"
	with open(data_file, 'rb') as csvfile:
		r = csv.reader(csvfile, delimiter=',')
		pool = Pool(cpu_count() * 120)
		print "Starting process.."
		all_results = pool.map(processed, r)
		pool.close()
		pool.join()
		print len(all_results)


		results = data_file+".clean_tweet.json"	
		print "writing clean data to file %s.."	% results
		json.dump(all_results, open(results,'w'))

		print "extracting unique features....."
		feature_list = get_feature_list(all_results)

		all_features = data_file+".feature_list.json"
		print "writing features to file %s"	 % all_features	
		json.dump(feature_list, open(all_features, 'w'))		



print "Processing %s" % pos_data_file
process_all_tweets(pos_data_file)	

print "Processing %s" % neg_data_file
process_all_tweets(neg_data_file)	
# processed()
