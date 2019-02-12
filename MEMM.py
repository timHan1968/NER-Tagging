import preprocess
import nltk
import numpy as np

#Feature set Format:
#<preToken, preNE, curToken, curPOS>
PRETOKEN = '<t>'
PRENE = '<n>'

TAGS = {'O': 0, 'PER': 1, 'LOC': 2, 'ORG': 3, 'MISC': 4}
IDX2TAGS = ['O', 'PER', 'LOC', 'ORG', 'MISC']

class MEMM:

    def __init__(self, train):

        self._trainCorpus = preprocess.MEMMpreprocess(train)

        print("Training the MaxEnt classifier...")
        self.classifier = genMaxEnt(self._trainCorpus)
        print("Done! MEMM model ready.")


    def assignTags(self, sentence, POS):
        """
        a function that assigns tags to a given sentence (as a string)
        and returns the assigned tags as a list.
        """
        tokens = sentence.strip().split()
        POStags = POS.strip().split()
        length = len(tokens)

        matProb = np.zeros((5, length))             #matrix for the probability
        matBP = np.zeros(5, str(length) + 'int8')   #matrix for the backpointers
        self.matInit(tokens[0], POStags[0], matProb)

        for col in range(1, length):
            curToken = tokens[col]; curPOS = POStags[col]
            preToken = tokens[col-1]; prevProb = matProb[:, col-1]

            for tag, row in TAGS.items():
                tempPrev = prevProb.copy()

                for preTag, j in TAGS.items():
                    features = {preToken: True, preTag: True, curToken: True,
                    curPOS: True}
                    transProb = self.getProb(features, tag)
                    tempPrev[j] = tempPrev[j] * transProb

                #Fill in the two matrixes
                matProb[row, col] = tempPrev.max()
                matBP[row, col] = tempPrev.argmax()

        return getPath(matProb, matBP)


    def matInit(self, firstWord, firstPOS, matProb):
        """
        a function that assign the initial probabilities to a
        MEMM probability matrix.
        """
        features = {PRETOKEN: True, PRENE: True, firstWord: True, firstPOS: True}
        for tag, idx in TAGS.items():
            matProb[idx, 0] = self.getProb(features, tag)


    def getProb(self, features, NE):
        """
        a function that returns the probability of a word with given
        [features] being labled [NE] by a maxEnt [classifier].
        """
        distribution = self.classifier.prob_classify(features)
        return distribution.prob(NE)



def genMaxEnt(trainTokens):
    """
    a function that generates the maxEnt classifier of nltk using a given
    trianing corpus.
    """
    #Note: For numIt, the default iterations of nltk's maxEnt package is
    #100 which would take nearly 2 hours to train, but 30 tends to give
    #good enough accuracy and takes only about 30 min.
    numIt = 1
    return nltk.MaxentClassifier.train(trainTokens, max_iter = numIt)


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


def evaluateMEMM(train, test):
    """
    a function that returns the accuracy of the MEMM model using a given
    validation set.
    """
    model = MEMM(train)
    lines = preprocess.readFile(test)
    lineNum = 1
    correct = 0; total = 0

    for line in lines:
        if (lineNum % 3) == 1:
            sentence = line

        elif (lineNum % 3) == 2:
            tags = model.assignTags(sentence, line)
            addBio(tags)

        elif (lineNum % 3) == 0:
            answers = line.strip().split()
            #Following line just for testing
            #assert len(tags) == len(answers)
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
    test = 'tempTest.txt'
    correct, total = evaluateMEMM(train, test)

    print("Out of " + str(total) + " tokens")
    print("HMM gets " + str(correct) + " corrects !")
    print("The accuracy is " + str(correct/total))


if __name__ == "__main__":
	main()
