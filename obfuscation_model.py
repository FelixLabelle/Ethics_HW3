import pandas
import sklearn
import argparse
from nltk import tokenize
import pickle

# TODO 1: Get the setup working from end-to-end, without any replacement
# TODO 2: Random replacement
# TODO 3: Selected replacement

# Load data
# Tokenize and prep data

# Replace "female" and "male" words
# Write a random replacement
# Write a selected replacement
	
# Write replaced data
# Save format as the training data

# Adapt the other script to this end

# Train, test, and cache the basic classifier for Obfuscation HW

def replacement_func(replacement_type, male_words, female_words):
	def replacer(text):
		pass
	return text

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", default="./dataset.csv")
	parser.add_argument("--output_file", default="./test.csv")
	parser.add_argument("--replacement", default="none", choices=['none', 'random', 'semantic'])
	args = parser.parse_args()

	input_data = pandas.read_csv(args.input_file)
	input_text = list(input_data["post_text"])
	input_tokens = [tokenize.word_tokenize(t) for t in input_text]
	# TODO: Add replacement here
	replaced_tokens = obfuscate()
	output_strings = [" ".join(tokens) for tokens in input_tokens]
	input_data["post_text"] = output_strings
	input_data.to_csv(args.output_file)

if __name__ == "__main__":
	main()