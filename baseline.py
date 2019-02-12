"""
This python script constructs a baseline NER tagging model.
"""
import preprocess

class Baseline:

	"""
	The baseline tagging model class......
	"""

	def __init__(self, filepath):

		"""
		The constructor initializes the corpus to be stored in the model.
		It creates a dictionary of tokens and taggs. Also a default tag used
        for unseen tokens.

		Input
		-----
		- filepath: The path of the file to be read in
		"""
		self._dict = preprocess.baselineDict(filepath)
		self._defaultTag = 'O'


	def assignTags(self, sentence):
		"""
		A function that return taggings to an observed sentence.

		Input
		-------
		sentence: a given list of tokens

		output
		-------
		returns the tags as a list with same length as [sentence]

		"""
		tags = sentence.copy()
		for i in range(len(tags)):
			word = tags[i]
			if word not in self._dict:
				tags[i] = self._defaultTag
			else:
				tags[i] = self._dict[word]

		return tags


	def scoreBaseLine(self, filepath):
		"""
		A function that grades the models accuracy using a given test
		file specified by [filepath].
		"""
		lines = preprocess.readFile(filepath)
		lineNum = 1
		correct = 0
		total = 0
		for line in lines:
			if (lineNum % 3) == 1:
				tokens = line.strip().split()
				tags = self.assignTags(tokens)
			elif (lineNum % 3) == 0:
				answers = line.strip().split()

                #Following section just for testing
				if len(tags) != len(answers):
					print("Lengths don't match!\n")
					print(tags)
					print(answers)
					break

				for i in range(len(tags)):
					if tags[i] == answers[i]:
						correct += 1
					total += 1
			lineNum += 1

		return correct, total


def main():
    train_file = "tempTrain.txt"
    test_file = "tempTest.txt"
    xx = Baseline(train_file)
    correct, total = xx.scoreBaseLine(test_file)
    print("Hello!\n")
    print("The Baseline Model assigns " + str(correct) + " correct tags")
    print("out of " + str(total) + " !\n")
    print("The accuracy is " + str(correct/total))


if __name__ == "__main__":
	main()
