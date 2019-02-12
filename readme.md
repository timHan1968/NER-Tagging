Note*: This is a previous group project imported from the Cornell-hosted github. 

Original link: https://github.coecis.cornell.edu/xh87/NLP-a2


# Named Entity Tagging
 
An NER tagging system that recognizes named entities, such as "Bostn" and "Google", in a given text file and lables them as one of the NER classes including "*location*", "*person*", "*organization*", and "*miscmiscellaneous*".

Includes three versions for comparison:

1. A simple rule-based Baseline model
2. A ML model using HMM (*Hidden Markov Model*)
3. A ML model using MEMM (*Maximum Entropy Markov Model*)

**Python Files:**

[Preprocess.py]:
Implements all preprocessing functions used for the three
tagging systems: Baseline, HMM, and MEMM.

[kaggle.py]:
Implements the functions that generate the csv output for each
of the three tagging systems. Run the [main] function would automatically
generate the csv file by a certain tagging model.

[baseline.py], [HMM.py], [MEMM.py]:
Implement the core functions for the three tagging models as the name indicate.
Each includes a [main] function that would evaluate the model with the file named
"tempTest.txt" as the testing set and "tempTrain.txt" as the training corpus.

[bigram.py]:
The bigram model imported from our previous project.


**Text Files:**

[tempTrain.txt]:
The txt file that serves as the training set for evaluating the three models.
The default one includes the first 36,000 lines of the original "train.txt".

[tempTest.txt]:
The txt file that serves as the testing set for evaluating the three models.
The default one includes the last 6,000 lines of the original "train.txt".

[test.txt] and [train.txt]:
The original "train.txt" and "test.txt" provided during the release of a2.

[trainNE.txt]:
The txt file with only the Named Entity lines of the original "train.txt".
Used for training the bigram (transition) model in [HMM.py].

[tempTrainNE.txt]:
Same as above except this file has NE lines only from teh "tempTrain.txt".