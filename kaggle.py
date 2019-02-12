"""
This python script generates the csv files for the tagging models.
"""
import preprocess
import baseline
import HMM
import MEMM
import csv

TRAIN = "train.txt"
TEST = "test.txt"


def BLdebug(tags):
    """
    A debug function for mistakenly assigned 'I' tags:
        1) 'I' tag assigned without a leading 'B' tag
        2) Inconsistant tag class
    For such cases, replace 'I' with 'B'.
    """
    preClass = None; NEcontinues = False
    for i in range(len(tags)):
        tag = tags[i]
        bioTag = tag[:1]
        if bioTag == 'B':
            preClass = tag[2:]
            NEcontinues = True
        elif bioTag == 'I':
            curClass = tag[2:]
            if (not NEcontinues) or (curClass != preClass):
                #Error 1) or 2)
                tags[i] = 'B' + tag[1:]
                preClass = curClass
                NEcontinues = True
        elif bioTag == 'O':
            preClass = None
            NEcontinues = False
    return tags


def baselineClassify(train, test):
    """
    The function returns the tagging prediction by the baseline system as
    a dictionary.
    """
    model = baseline.Baseline(train)
    lines = preprocess.readFile(test)
    prediction = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    lineNum = 1

    for line in lines:
        if (lineNum % 3) == 1:
            #Line with Tokens
            tokens = line.strip().split()
            tags = BLdebug(model.assignTags(tokens))
        elif (lineNum % 3) == 0:
            #Line with indexes
            indexes = line.strip().split()
            preClass = None; firstIdx = None; lastIdx = None
            NEcontinues = False

            for i in range(len(tags)):
                bioTag = tags[i][:1]

                if bioTag == 'B':
                    if NEcontinues:
                        #Previous tag ends
                        prediction[preClass].append(firstIdx + '-' + lastIdx)
                    preClass = tags[i][2:]
                    firstIdx = indexes[i]; lastIdx = indexes[i]
                    NEcontinues = True

                elif bioTag == 'I':
                    curClass = tags[i][2:]
                    assert NEcontinues and curClass == preClass
                    lastIdx = indexes[i]

                else:   # bioTag == 'O'
                    if NEcontinues:
                        prediction[preClass].append(firstIdx + '-' + lastIdx)
                    preClass = None;
                    firstIdx = None; lastIdx = None
                    NEcontinues = False

        lineNum += 1

    return prediction


def HMMClassify(train, trainNE, test):
    """
    The function returns the tagging prediction by the HMM system as
    a dictionary.
    """
    model = HMM.HMM(train, trainNE)
    lines = preprocess.readFile(test)
    prediction = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    lineNum = 1

    for line in lines:
        if (lineNum % 3) == 1:
            #Line with Tokens
            tags = model.assignTags(line)
        elif (lineNum % 3) == 0:
            #Line with indexes
            indexes = line.strip().split()
            preClass = None; firstIdx = None; lastIdx = None
            NEcontinues = False

            for i in range(len(tags)):
                tag = tags[i]
                if tag == 'O':
                    if NEcontinues:
                        #Previous tag ends
                        prediction[preClass].append(firstIdx + '-' + lastIdx)
                    preClass = None
                    firstIdx = None; lastIdx = None
                    NEcontinues = False

                else:
                    if NEcontinues:
                        if tag != preClass:
                            #Previous tag ends, new Tag begins
                            prediction[preClass].append(firstIdx + '-' + lastIdx)
                            preClass = tag
                            firstIdx = indexes[i]; lastIdx = indexes[i]
                        else:
                            #Previous tag continues
                            lastIdx = indexes[i]
                    else:
                        #New tag begins
                        preClass = tag
                        firstIdx = indexes[i]; lastIdx = indexes[i]
                        NEcontinues = True
        lineNum += 1
    return prediction


def MEMMClassify(train, test):
    """
    The function returns the tagging prediction by the MEMM system as
    a dictionary.
    """
    model = MEMM.MEMM(train)
    lines = preprocess.readFile(test)
    prediction = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    lineNum = 1

    for line in lines:
        if (lineNum % 3) == 1:
            #Line with Tokens
            sentence = line

        elif (lineNum % 3) == 2:
            tags = model.assignTags(sentence, line)

        else:
            #Line with indexes
            indexes = line.strip().split()
            preClass = None; firstIdx = None; lastIdx = None
            NEcontinues = False

            for i in range(len(tags)):
                tag = tags[i]
                if tag == 'O':
                    if NEcontinues:
                        #Previous tag ends
                        prediction[preClass].append(firstIdx + '-' + lastIdx)
                    preClass = None
                    firstIdx = None; lastIdx = None
                    NEcontinues = False

                else:
                    if NEcontinues:
                        if tag != preClass:
                            #Previous tag ends, new Tag begins
                            prediction[preClass].append(firstIdx + '-' + lastIdx)
                            preClass = tag
                            firstIdx = indexes[i]; lastIdx = indexes[i]
                        else:
                            #Previous tag continues
                            lastIdx = indexes[i]
                    else:
                        #New tag begins
                        preClass = tag
                        firstIdx = indexes[i]; lastIdx = indexes[i]
                        NEcontinues = True
        lineNum += 1
    return prediction


def writeKaggle(prediction, filepath):
    """
    Function that writes the prediction (dictionary) of a tagging LM
    into a csv file follwing the Kaggle format.
    """
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Prediction'])
        for type, indexes in prediction.items():
            str = ""
            for index in indexes:
                str += ' ' + index
            writer.writerow([type, str.strip()])
    f.close()


def main():
    print("Generating csv file...")
    #prediction = baselineClassify(TRAIN, TEST)
    #writeKaggle(prediction, "BLpredict.csv")

    #prediction = HMMClassify(TRAIN, 'trainNE.txt', TEST)
    #writeKaggle(prediction, "HMMpredict.csv")

    prediction = MEMMClassify(TRAIN, TEST)
    writeKaggle(prediction, "MEMMpredict.csv")


if __name__ == "__main__":
	main()
