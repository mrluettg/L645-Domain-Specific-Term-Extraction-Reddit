import spacy
import json
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

## reads base path
## returns dictionary: subreddit -> list of sentences
FILEPATH = "D:\\L645_corpora\\project_corpora\\"
TESTCORPUS = "StarWarsEU"
BASECORPUS = "reddit-corpus-small"



punct = [",", ".", "<", ">", "/", "?", ";", ":", '\'', "\"", "[", "]", "{", "]", "\\", "!", "@", "#", "$", "%", "^",
             "(", ")", "`", "“", "”", " ", "\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t"]
stop_punct = stopwords.words('english') + punct


def read_json(filepath):
    file = open(filepath, "r")
    subreddit_sentences = {}
    all_posts = json.load(file)
    for data in all_posts:
        text = data["text"]
        meta = data["meta"]
        subreddit = meta["subreddit"]
        sents = sent_tokenize(text)
        if subreddit in subreddit_sentences:
            subreddit_sentences[subreddit] += sents
        else:
            subreddit_sentences[subreddit] = sents
    return subreddit_sentences


##reads subreddit path.
##returns list of sentences.
def read_jsonl(filepath):
    file = open(filepath, "r")
    sentences = []
    for line in file:
        data = json.loads(line)
        text = data["text"]
        sents = sent_tokenize(text)
        sentences += sents
    return sentences


def tokenizeSubtree(token):
    str = ""
    for descendant in token.subtree:
        str += descendant.text + "_"
    str = str[:-1]
    return str


def tokenizeLefts(token):
    lst = []
    prev_descendants = []
    reverse_descendents = []
    for descendant in token.lefts:
        reverse_descendents.append(descendant)
    reverse_descendents.reverse()
    for descendant in reverse_descendents:
        descendant_subtree = tokenizeSubtree(descendant)
        str = descendant_subtree + "_"
        for prev_desc in prev_descendants:
            str += prev_desc + "_"
        str += token.text
        lst.append(str)
        prev_descendants.insert(0, descendant_subtree)
    return lst


def tokenizeRights(token):
    lst = []
    str = token.text
    for descendant in token.rights:
        descendant_subtree = tokenizeSubtree(descendant)
        str += "_" + descendant_subtree
        lst.append(str)
    return lst

def tokenizeLst(lst):
    str = ""
    for word in lst:
        str += word + "_"
    str = str[:-1]
    return str

def add(tokenLst, token):
    #print("before add",  type(tokenLst), type(token))
    if token.count("_") == 0 or token.count("_") > 6:
        return tokenLst
    if token.count("\n") > 0 or token.count("\t") > 0:
        return tokenLst
    if token in tokenLst:
        return tokenLst
    token_words = token.split("_")
    if token_words[0] in stop_punct:
        no_punct_stop = token_words[1:]
        return add(tokenLst, tokenizeLst(no_punct_stop))
    if token_words[len(token_words) - 1] in stop_punct:
        no_punct_stop = token_words[:-1]
        return add(tokenLst, tokenizeLst(no_punct_stop))
    tokenLst.append(token)
    return tokenLst

#not recursive anymore
def recursiveTokenization(token):
    LIMIT = 10
    nounTags = ["PRON", "PROPN", "NOUN", "NUM"]
    tokens = []
    subtree = token.subtree
    count = sum(1 for x in subtree)
    if token.pos_ in nounTags and count > 1:
        #ok we're going to limit the length to say, 10 to limit runtime.
        tokens = add(tokens, tokenizeSubtree(token))
        if(count > 2):
            left_tokens = tokenizeLefts(token)
            right_tokens = tokenizeRights(token)
            combined_tokens = left_tokens + right_tokens
            for multitoken in combined_tokens:
                tokens = add(tokens, multitoken)
        #for child in token.children:
            #tokens += recursiveTokenization(child)
    return tokens


def MWETokenizer(doc):
    nounTags = ["PRON", "PROPN", "NOUN", "NUM"]
    # string to token object
    # for token in doc: print(token.text, [child for child in token.children])
    tokens = []
    token_pos = {}
    for token in doc:
        # print(token, token.pos_, [child.text for child in token.children])
        if token.pos_ in nounTags and token.text != "_":
            child_tokens = recursiveTokenization(token)
            for child in child_tokens:
                tokens = add(tokens, child)
        token_pos[token.text] = token.pos_
    #for token_str in tokens: print(token_str)
    return tokens
#all frequencies.
#returns dictionary (subreddit(string) -> dictionary(term(string) -> frequency))



def subreddit_multigram_frequencies(subreddit_sentences):
    nlp = spacy.load("en_core_web_sm")
    term_freqs = {}
    incrementer = 0
    size = len(subreddit_sentences)
    for sentence in subreddit_sentences:
        doc = nlp(sentence)
        unigrams = word_tokenize(sentence)
        multigrams = MWETokenizer(doc)
        allgrams = unigrams + multigrams
        for gram in allgrams:
            gram = gram.lower()
            if gram in term_freqs:
                term_freqs[gram] = term_freqs[gram] + 1
            else:
                term_freqs[gram] = 1
        if(incrementer % 1000 == 0): print(str(incrementer) + "/" + str(size) + " sentences")
        incrementer += 1
    return term_freqs


def write_subreddit_freq(subreddit_name, subreddit_frequencies):
    file = open(FILEPATH + "MWE_frequencies\\" + subreddit_name + "_MWE_frequencies.txt", 'w', encoding='utf8')
    for key, value in subreddit_frequencies.items():
        print(key, str(value))
        file.write(key + "\t" + str(value) + "\n")
    file.close()


def combinedCorpora():
  compare_subreddit = read_jsonl(FILEPATH + TESTCORPUS + "\\utterances.jsonl")
  all_subreddits = read_json(FILEPATH + BASECORPUS + "\\utterances.json")
  all_subreddits[TESTCORPUS] = compare_subreddit
  num_subreddits = len(all_subreddits)
  incrementer = 1
  for key, value in all_subreddits.items():
      print("subreddit " + str(incrementer) + "/" + str(num_subreddits) + ": " + key)
      freqs = subreddit_multigram_frequencies(value)
      write_subreddit_freq(key, freqs)
      incrementer += 1

def singleTestCorpus(max):
    compare_subreddit_sentences = read_jsonl(FILEPATH + TESTCORPUS + "\\utterances.jsonl")
    cut_sentences = []
    for i in range(0, max):
        cut_sentences.append(compare_subreddit_sentences[i])
    freqs = subreddit_multigram_frequencies(cut_sentences)
    write_subreddit_freq(TESTCORPUS, freqs)




#tokens = MWETokenizer("we're going to apply text classification and word embeddings to named entity taggers")
#tokens += MWETokenizer("I like the chimp eating an apple on the tree.")
#tokens += MWETokenizer("We're going to Buskirk and Chumley Theater")
#for token in tokens: print(token)

##run()
##read_json(FILEPATH + BASECORPUS + "\\utterances.json")
##read_jsonl(FILEPATH + TESTCORPUS + "\\utterances.jsonl")

##tokens = MWETokenizer("we're going to apply text classification and word embeddings to named entity taggers")
##for token in tokens: print(token)
singleTestCorpus(100000)

