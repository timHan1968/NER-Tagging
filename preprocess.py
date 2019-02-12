"""
This python script includes preprocessing function for the HMM, MEMM,
and baseline system models.

Authors: Sena Katako, Vivian Gao, Xianyi Han
"""
PRETOKEN = '<t>'
PRENE = '<n>'
UNK = '<UNK>'


def baselineDict(filepath):
    """
    The function reads a test corpus and processes it into a dictionary
    that contains all tokens and their NER tag (only the last one).
    """
    lines = readFile(filepath)
    lineNum = 1
    dict = {}

    for line in lines:
        if (lineNum % 3) == 1:
            #Line contains tokens
            tokens = line.strip().split()
        elif (lineNum % 3) == 0:
            #Line contains BIO tags
            tags = line.strip().split()
            assert len(tokens) == len(tags)
            for i in range(len(tokens)):
                dict[tokens[i]] = tags[i]
        lineNum += 1

    return dict


def tagDictHMM(filepath):
    """
    The function reads a test corpus and processes it into a nested dictionary
    that contains frequencies of all tag|word combinations.
    """
    lines = readFile(filepath)
    lineNum = 1
    dict = {'O':{}, 'PER':{}, 'LOC':{}, 'ORG':{}, 'MISC':{}}
    vocab = {}

    for line in lines:
        if (lineNum % 3) == 1:
            tokens = line.strip().split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = 1
                else:
                    vocab[token] += 1
        elif (lineNum % 3) == 0:
            tags = line.strip().split()
            for i in range(len(tokens)):
                if tags[i][:1] == 'O':
                    tagNE = 'O'
                else:
                    tagNE = tags[i][2:]
                token = tokens[i]
                if token not in dict[tagNE]:
                    dict[tagNE][token] = 1
                else:
                    dict[tagNE][token] += 1
        lineNum += 1
    return dict, vocab


def MEMMpreprocess(train):
    """
    Preprocessing function for the MEMM model. Returns a list of
    tuples which contains the feature set (dictionary) and the
    labled class.
    """
    #Feature set Format:
    #<preToken, preNE, curToken, curPOS>
    lines = readFile(train)
    lineNum = 1
    corpus = []

    for line in lines:
        if (lineNum % 3) == 1:
            tokens = line.strip().split()
            tokens.insert(0, PRETOKEN)

        elif (lineNum % 3) == 2:
            POStags = line.strip().split()

        else:   #Third line with BIO tags
            NEtags = line.strip().split()
            NEtags.insert(0, PRENE)

            for i in range(len(NEtags)):
                if NEtags[i][:1] != 'O' and NEtags[i] != PRENE:
                    NEtags[i] = NEtags[i][2:]

            for i in range(1, len(tokens)):
                features = {tokens[i-1]: True, NEtags[i-1]: True,
                tokens[i]: True, POStags[i-1]: True}
                corpus.append((features, NEtags[i]))

        lineNum += 1
    return corpus


def readFile(filepath):

	"""
	Reads a text file and returns a list of paragraphs

	Input
	------
	filepath: the file path

	Output
	-------
	text: a list of the sentences in the text file
	"""

	with open(filepath, "r", encoding = "utf-8") as fp:
		text = fp.readlines()
	fp.close()
	return text


def extractTags(exTo, exFrom = "Project2_fall2018/train.txt"):
    """
    Extracts the NE tags from train.txt to a new file.
    """
    lines = readFile(exFrom)
    lineNum = 1
    with open(exTo, "w", encoding = "utf-8") as f:
        for line in lines:
            if (lineNum % 3) == 0:
                f.write(line)
            lineNum += 1
    f.close()


def main():
    print('Hi user, Glad at your service!')
    #extractTags('temptrainNE.txt', 'tempTrain.txt')


if __name__ == "__main__":
	main()
