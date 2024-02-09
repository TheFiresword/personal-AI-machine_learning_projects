import string
import nltk
from nltk.corpus import wordnet
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> VG | NG | NG VG | VG PG | NG VG Conj S 
NG -> N | Det N | Det N Adv | Det NAdj N | Det N  P  NG | Det NAdj N  P  NG
PG -> P NG
VG -> V | V NG | V Adv | V Adv NG | Adv V | Adv V NG | V P NG 
NAdj -> Adj | Adj NAdj
"""

NG = "NG"

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)


def parse(sentence):
    s = preprocess(sentence)
    parser = nltk.ChartParser(grammar)
    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    relevant_words = []
    # Print each tree with noun phrase chunks
    for tree in trees:
        for np in np_chunk(tree):
            relevant_words.append(" ".join(np.flatten()))
    return relevant_words


def add_grammar_rule(wordlist):
    convert_dic = {wordnet.NOUN: ""}
    global TERMINALS
    for word in wordlist:
        synset = wordnet.synsets(word)
        if synset:
            match synset[0].pos():
                case wordnet.NOUN:
                    TERMINALS += f'N -> "{word}" \n'
                case wordnet.VERB:
                    TERMINALS += f'V -> "{word}" \n'
                case wordnet.ADJ:
                    TERMINALS += f'Adj -> "{word}" \n'
                case wordnet.ADV | wordnet.ADJ_SAT:
                    TERMINALS += f'Adv -> "{word}" \n'
                case "i":
                    TERMINALS += f'P -> "{word}" \n'
            print(synset)
    print(TERMINALS)


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    all_words = []
    for word in nltk.word_tokenize(sentence):
        word = word.lower()
        if any(letter in word for letter in string.ascii_letters):
            all_words.append(word)
    return all_words


def np_chunk(tree: nltk.Tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    npclist = []
    for substree in tree.subtrees(lambda x: x.label() == NG):
        if [subsub.label() for subsub in substree.subtrees()].count(NG) > 1:
            continue
        npclist.append(substree)
    return npclist


add_grammar_rule(["pinch", "mom", "at", "dance"])
