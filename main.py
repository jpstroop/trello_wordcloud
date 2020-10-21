from json import dumps, load
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from regex import match, sub

stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")
addl_stops = ["anne", "anne's", "may", "n't", "'s", "barbara", "jon", "'fall",
    "e.g"]
short_words = ["us", "pu", "hr", "ux"]

def is_word(w):
    '''Return False if a word should be excluded'''
    if w.isdigit(): return False
    if all([match(r"\p{P}|`", l) for l in w]): return False
    if w in addl_stops: return False
    if len(w) < 3 and w not in short_words: return False
    return True

def pad_slashes(s):
    return sub(r"\/", " / ", s)

with open('./llt_swot.json') as f:
    cards = load(f)['cards']
open_cards = [c for c in cards if not c['closed']]

words = []
root_words_lookup = {}
for card in open_cards:
    card_words = []
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['name']))]
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['desc']))]
    # create a lookup where the stemmed version is the key, and the value is
    # a list of the variants. We append the first variant to the list, so as
    # to avoid non-words
    for word in card_words:
        stemmed = stemmer.stem(word).lower()
        # TODO: split on "/" as well, and provide an override lookup for
        # the preferred version e.g. "prof."
        if stemmed not in root_words_lookup:
            root_words_lookup[stemmed] = []
        if word not in root_words_lookup[stemmed]:
            root_words_lookup[stemmed].append(word.lower())
        if stemmed not in stopwords and is_word(stemmed):
            words.append(root_words_lookup[stemmed][0])

for word in words:
    print(word)
