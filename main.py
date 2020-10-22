from collections import Counter
from json import dumps, load
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from PIL import Image
from regex import match, sub
from wordcloud import WordCloud
import numpy as np

# TODO: WordCloud
# See: https://towardsdatascience.com/simple-wordcloud-in-python-2ae54a9f58e5
# See: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
# See: https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
# TODO: Synonyms?

with open('./llt_swot.json') as f:
    cards = load(f)['cards']

with open('./training.json') as f:
    data = load(f)
    root_words_lookup = data["root_words_lookup"]
    addl_stops = data["addl_stops"]
    short_words = data["short_words"]

def is_word(w):
    '''Return False if a word should be excluded'''
    if w.isdigit(): return False
    if all([match(r"\p{P}|`", l) for l in w]): return False
    if len(w) < 3 and w not in short_words: return False
    return True

def pad_slashes(s):
    return sub(r"\/", " / ", s)

def to_wordcloud(word_list, min_occurs=3, mask=None):
    words = {w: c for w, c in Counter(word_list).items() if c >= min_occurs}
    return WordCloud(width=3000, height=2000, random_state=2, colormap='Set2',
        background_color='black', min_font_size=10, max_words=len(words),
        prefer_horizontal=0.6, stopwords=[], mask=mask,
        collocations=False).generate_from_frequencies(words)

stopwords = stopwords.words("english") + addl_stops
stemmer = SnowballStemmer("english")

words = []
for card in filter(lambda c: not c['closed'], cards):
    card_words = []
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['name']), language='english')]
    [card_words.append(w) for w in word_tokenize(pad_slashes(card['desc']), language='english')]
    for word in card_words:
        stemmed = stemmer.stem(word).lower()
        word = word.lower()
        if stemmed not in stopwords and is_word(stemmed):
            # append the first label; can adjust/add in training.json
            words.append(root_words_lookup[stemmed][0])


# mask = np.array(Image.open('logo.png'))
# wc = to_wordcloud(words, mask=mask, min_occurs=2)
wc = to_wordcloud(words)
wc.to_file("wordcloud.png")
