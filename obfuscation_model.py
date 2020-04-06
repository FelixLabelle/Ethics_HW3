import pandas
import sklearn
import argparse
from nltk import tokenize
import pickle
from random import choice, random
from utils import load_gendered_words
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


def replacement_selection(word_rep_func):
	def replacement_function(row):
		input_tokens = tokenize.word_tokenize(row['post_text'])
		return " ".join([word_rep_func(token, row['op_gender']) for token in input_tokens])
	return replacement_function

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", default="./dataset.csv")
	parser.add_argument("--output_file", default="test.csv")
	parser.add_argument("--replacement", default="none", choices=['none', 'random', 'semantic'])
	parser.add_argument("--replacement_prob", default=1, type=float)
	parser.add_argument("--semantic_threshold", default=-1, type=float)
	args = parser.parse_args()
	input_data = pandas.read_csv(args.input_file)
	gendered_word_lut = {'M': {}, 'W':{}}
	male_vocab, female_vocab = load_gendered_words()
	if args.replacement == "random":
		for word in male_vocab:
			gendered_word_lut['M'][word] = list(female_vocab)
		for word in female_vocab:
			gendered_word_lut['W'][word] = list(male_vocab)
	elif args.replacement == "semantic":
		with open('semantic_map.pkl', 'rb') as sem_fh:
			gendered_word_lut = pickle.load(sem_fh)
		for gender in gendered_word_lut:
			for word_to_replace in gendered_word_lut[gender]:
				scores, words = [lst for lst in zip(*gendered_word_lut[gender][word_to_replace])]
				if args.semantic_threshold:
					words = [word for word,score in zip(words,scores) if score >= args.semantic_threshold]
					if len(words) == 0:
						words = [word_to_replace]
				gendered_word_lut[gender][word_to_replace] = [words[0]]
	else:
		pass
	def replacement(token, gender):
		replacement_token = token
		if random() < args.replacement_prob:
			replacement_token = choice(gendered_word_lut[gender].get(token, [token]))
		return replacement_token
	
	processed_text = input_data.apply(replacement_selection(replacement), axis = 1) # place
	input_data["post_text"] = processed_text
	output_loc = "{}_{}_{}_{}".format(args.replacement, args.replacement_prob, args.semantic_threshold, args.output_file)
	print("Writting to {}".format(output_loc))
	input_data.to_csv(output_loc)

if __name__ == "__main__":
	main()