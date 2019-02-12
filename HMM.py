"""
This python script includes preprocessing function for the HMM, MEMM,
and baseline system models.

Authors: Sena Katako, Vivian Gao, Xianyi Han
"""
import preprocess
import bigram
import numpy as np

TAGS = {'O': 0, 'PER': 1, 'LOC': 2, 'ORG': 3, 'MISC': 4}
IDX2TAGS = ['O', 'PER', 'LOC', 'ORG', 'MISC']

F1 = "train.txt"
F2 = "trainNE.txt"

class HMM:

    def __init__(self, train, trainNE):

        self._bigramNE = bigram.BigramLM(trainNE)
        self._lexicon, self._vocab = preprocess.tagDictHMM(train)

        #Handling Unknown words for Lexicon
        self._unknowns = assignUnk(self._vocab)
        addUnknown(self._lexicon, self._unknowns)


    def assignTags(self, sentence):
        """
        a function that assigns tags to a given sentence (as a string)
        and returns the assigned tags as a list.
        """
        tokens = sentence.strip().split()
        length = len(tokens)

        matProb = np.zeros((5, length))             #matrix for the probability
        matBP = np.zeros(5, str(length) + 'int8')   #matrix for the backpointers
        self.matInit(tokens[0], matProb)

        for i in range(1, length):
            token = tokens[i]
            prevProb = matProb[:, i-1]

            for tag, idx in TAGS.items():
                tempPrev = prevProb.copy()

                #Multiplies transition probabilities
                for preTag, j in TAGS.items():
                    transProb = self.getTransProb(preTag, tag)
                    tempPrev[j] = tempPrev[j] * transProb

                #Fill in the two matrixes
                lexProb = self.getLexProb(token, tag)
                matProb[idx, i] = tempPrev.max() * lexProb
                matBP[idx, i] = tempPrev.argmax()

        return getPath(matProb, matBP)


    def matInit(self, firstWord, matProb):
        """
        a function that assign the initial probabilities to a
        HMM probability matrix.
        """
        for tag, idx in TAGS.items():
            lexProb = self.getLexProb(firstWord, tag)
            transProb = self.getTransProb("<s>", tag)
            matProb[idx, 0] = lexProb * transProb


    def getLexProb(self, word, tag):
        """
        a function that returns the probability of [word] given [tag]
        """
        if word not in self._vocab:
            word = '<UNK>'
        if word not in self._lexicon[tag]:
            num = 0
        else:
            num = self._lexicon[tag][word]
        den = self._bigramNE._tokens[tag]
        return num/den


    def getTransProb(self, firstTag, secondTag):
        """
        a function that returns the transion (bigram) probabilities
        of a given pair of tags.
        """
        return self._bigramNE.calBiProb(firstTag, secondTag)


def getPath(matProb, matBP):
    """
    a function that returns the best path, aka assigned tags, for a
    given viberti matrix with all backpointers and probabilities
    generated.
    """
    if len(matProb[1]) == 1:
        return [IDX2TAGS[matProb.argmax()]]

    path = []
    currentIndex = len(matProb[1])-1

    #Get tag of last column
    lastColumn = matProb[:, currentIndex]
    maxValueIndex = lastColumn.argmax()
    path.insert(0, IDX2TAGS[maxValueIndex])

    #follows the backpointer and assign tags to the rest of the sentence
    backpointer = matBP[maxValueIndex][currentIndex]
    while currentIndex > 0:
        currentIndex -= 1
        tag = IDX2TAGS[backpointer]
        path.insert(0, tag)
        backpointer = matBP[backpointer][currentIndex]

    return path


def assignUnk(vocab):
    """
    a function that decides the list of unknowns from a given vocab
    """
    unknowns = []
    unkFreq = 0; unkNum = 0;
    max = 20;

    for k, v in list(vocab.items()):
        if v == 1:
            unknowns.append(k)
            unkFreq += v
            unkNum += 1
            vocab.pop(k)

        if unkNum == max:
            vocab['<UNK>'] = unkFreq
            break

    return unknowns


def addUnknown(lexicon, unknowns):
    """
    a function that adds unknown words to a built lexicon and returns
    the total number of tokens after removal
    """
    for tag in lexicon:
        for word, count in list(lexicon[tag].items()):
            if word in unknowns:
                if '<UNK>' not in lexicon[tag]:
                    lexicon[tag]['<UNK>'] = count
                else:
                    lexicon[tag]['<UNK>'] += count
                lexicon[tag].pop(word)


def evaluateHMM(train, trainNE, test):
    """
    a function that returns the accuracy of the HMM model using a given
    validation set.
    """
    model = HMM(train, trainNE)
    lines = preprocess.readFile(test)
    lineNum = 1
    correct = 0; total = 0

    for line in lines:
        if (lineNum % 3) == 1:
            tags = model.assignTags(line)
            addBio(tags)
        elif (lineNum % 3) == 0:
            answers = line.strip().split()
            #Following line just for testing
            assert len(tags) == len(answers)
            for i in range(len(tags)):
                if tags[i] == answers[i]:
                    correct += 1
                total += 1
        lineNum += 1

    return correct, total


def addBio(tags):
    """
    a function that adds the B and I prefixes to a list of tags.
    """
    #print(tags)
    preNE = None; NEcontinues = False
    for i in range(len(tags)):
        tag = tags[i]
        if tag != 'O':
            if NEcontinues:
                if tag == preNE:
                    tags[i] = 'I-' + tag
                else:
                    tags[i] = 'B-' + tag
                    preNE = tag
            else:
                #print(tag)
                tags[i] = 'B-' + tag
                preNE = tag; NEcontinues = True
        elif NEcontinues:
            preNE = None; NEcontinues = False


def main():
    print("Hello user !")

    train = 'tempTrain.txt'
    trainNE = 'temptrainNE.txt'
    test = 'tempTest.txt'
    correct, total = evaluateHMM(train, trainNE, test)

    print("Out of " + str(total) + " tokens")
    print("HMM gets " + str(correct) + " corrects !")
    print("The accuracy is " + str(correct/total))


if __name__ == "__main__":
	main()
