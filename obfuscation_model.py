import pandas
import sklearn
import argparse
from nltk import tokenize
import pickle
from random import choice

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
def load_gendered_words(female_load_path = "female.txt", male_load_path = "male.txt"):
	female_words = []
	male_words = []
	with open(female_load_path) as female_fh:
		female_words = [word.strip() for word in female_fh.readlines()]
	with open(male_load_path) as male_fh:
		male_words = [word.strip() for word in male_fh.readlines()]
	return set(male_words), set(female_words)


def replacement_selection(word_rep_func):
	def replacement_function(row):
		input_tokens = tokenize.word_tokenize(row['post_text'])
		return " ".join([word_rep_func(token, row['op_gender']) for token in input_tokens])
	return replacement_function

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", default="./dataset.csv")
	parser.add_argument("--output_file", default="./test.csv")
	parser.add_argument("--replacement", default="none", choices=['none', 'random', 'semantic'])
	args = parser.parse_args()
	input_data = pandas.read_csv(args.input_file)
	gendered_word_lut = {'M': {}, 'W':{}}
	male_vocab, female_vocab = load_gendered_words()
	#import pdb;pdb.set_trace()
	if args.replacement == "random":
		for word in male_vocab:
			gendered_word_lut['M'][word] = list(female_vocab)
		for word in female_vocab:
			gendered_word_lut['W'][word] = list(male_vocab)
	elif args.replacement == "semantic":
		pass
	else:
		pass
	def replacement(token, gender):
		return choice(gendered_word_lut[gender].get(token, [token]))
	
	processed_text = input_data.apply(replacement_selection(replacement), axis = 1) # place
	input_data["post_text"] = processed_text
	input_data.to_csv(args.output_file)

if __name__ == "__main__":
	main()