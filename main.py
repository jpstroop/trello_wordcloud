from json import dumps, load
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from regex import match, sub

# TODO: https://www.geeksforgeeks.org/generating-word-cloud-python/

stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")


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

with open('./training.json') as f:
    data = load(f)
    root_words_lookup = data["root_words_lookup"]
    addl_stops = data["addl_stops"]
    short_words = data["short_words"]

words = []
for card in open_cards:
    card_words = []
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['name']))]
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['desc']))]
    # create a lookup where the stemmed version is the key, and the value is
    # a list of the variants. We append the first variant to the list, so as
    # to avoid non-words
    for word in card_words:
        stemmed = stemmer.stem(word).lower()
        word = word.lower()
        if stemmed not in stopwords and is_word(stemmed):
            # append the first label
            words.append(root_words_lookup[stemmed][0])



# for word in sorted(set(words)):
print(len(set(words)))
