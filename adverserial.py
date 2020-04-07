import pandas
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
from tqdm import tqdm
from torch import optim
import torch
from copy import deepcopy

class Vocab:
	def __init__(self, min_freq = 0, reserved_symbols = {"<unk>" : 0, "<pad>" : 1}):
		self.itos = []
		self.stoi = {}
		self.min_freq = min_freq
		self.reserved_symbols = reserved_symbols
		self.stoi = reserved_symbols
		self.itos = [''] * len(reserved_symbols)
		for item, pos in self.reserved_symbols.items():
			self.itos[pos] = item
		
	def createVocab(self, training_tokens):
		token_counts = Counter(training_tokens)
		self.itos = self.itos + [word for word, count in token_counts.items() if count >= self.min_freq]
		self.stoi = {word:idx for idx,word in enumerate(self.itos)}

	def __len__(self):
		return len(self.itos)

	def __getitem__(self,item):
		# Todo: Accept tensors and the likes
		if type(item) == int:
			if item >= len(self.itos):
				item = reserved_symbols['<unk>']
			return self.itos[item]
		elif type(item) == str:
			return self.stoi.get(item, self.stoi["<unk>"])
		else:
			print(item, type(item))
			raise "Invalid type {} passed to vocab".format(type(item))
		
class Preprocessor:
	def __init__(self, lower = True, split = "space"):
		self.lower = lower
		self.split = split
	
	def __call__(self,text):
		if self.lower:
			text = text.lower()

		if self.split == 'space':
			text = text.split(' ')
		elif self.split == "regex" and self.regex:
			text = self.regex.findall(text)
		else:
			raise 'Invalid split selected'
		return text

class PostData(Dataset):
	def __init__(self, filename, preprocessor, vocab,
			gender_LUT = {"M" : 0,
						"W" : 1},
			board_LUT = {"funny" : 0,
						"relationships" : 1}
				):
		self.filename = filename
		all_data = pandas.read_csv(filename)
		posts = all_data['post_text'].to_list()
		self.tokens = [torch.tensor([vocab[token] for token in preprocessor(post)]).long() for post in posts]
		self.gender = torch.tensor([gender_LUT[gender] for gender in all_data['op_gender'].to_list()]).long()
		self.board = torch.tensor([board_LUT[board] for board in all_data['subreddit'].to_list()]).long()
	
	def __len__(self):
		return len(self.board)

	def __getitem__(self, idx):
		return self.tokens[idx], (self.gender[idx], self.board[idx])

def collateFunction(batch):
	# batch contains a list of tuples of structure (sequence, target)
	data = [item[0] for item in batch]
	data = rnn_utils.pack_sequence(data, enforce_sorted=False)
	targets = torch.LongTensor([item[1] for item in batch])
	return [data, targets]	

class PredictionModel(nn.Module):
	''' Generic model that takes in a hidden layer
	and outputs some class (e.g. gender, board) '''
	def __init__(self, input_dim, output_dim):
		super(PredictionModel, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.output_predictor = nn.Linear(self.input_dim,self.output_dim)
		
	def __call__(self, hid_layer):
		return self.output_predictor(hid_layer)

class TextPredictionModel(nn.Module):
	''' Takes text and generates a hidden layer '''
	def __init__(self, embedding_dim, hidden_dim, vocab_size, aggregation_method):
		super(TextPredictionModel, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
		self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim)
		self.aggregation_method = aggregation_method #Start with the max over the layers

	def _element_wise_embedding(self, packed_sequence):
		"""applies a pointwise function fn to each element in packed_sequence"""
		return torch.nn.utils.rnn.PackedSequence(self.embeddings(packed_sequence.data), packed_sequence.batch_sizes)
	
	def __call__(self, text):
		embeddings = self._element_wise_embedding(text)
		output, hidden_state = self.rnn(embeddings)
		hidden_space, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
		if self.aggregation_method == "max":
			hidden_space = hidden_space.max(0)[0]
		elif self.aggregation_method == "attention":
			pass
		elif self.aggregation_method == "last":
			row_indices = torch.arange(0, unpacked_len.size(0)).long()
			col_indices = unpacked_len - 1
			hidden_space = hidden_space[col_indices ,row_indices, :]

		return hidden_space

class AdverserialModel(nn.Module):
	def __init__(self, **params):
		super(AdverserialModel, self).__init__()
		self.text_pred_model = TextPredictionModel(params['embedding_dim'],params['hidden_dim'],params['vocab_size'],params['agg_method'])
		self.gender_predictor = PredictionModel(params['hidden_dim'],params['num_genders'])
		self.board_predictor = PredictionModel(params['hidden_dim'],params['num_boards'])
	
	def __call__(self, text):
		hid_layer = self.text_pred_model(text)
		return self.gender_predictor(hid_layer), self.board_predictor(hid_layer)

def splitDataset(filename, split_size = 0.8):
	# Load entire data
	all_data = pandas.read_csv(filename)
	dataset_size = len(all_data)
	split_idx = int(0.8 * dataset_size)

	# Split
	training_data = all_data.iloc[:split_idx, :]
	validation_data = all_data.iloc[split_idx:, :]
	training_data.to_csv("training_data.csv")
	validation_data.to_csv("validation_data.csv")

def train(**params):
	preprocessor = Preprocessor()
	all_data = pandas.read_csv("training_data.csv")
	posts = all_data['post_text'].to_list()
	training_tokens = [token for post in posts for token in preprocessor(post)]
	vocab = Vocab()
	vocab.createVocab(training_tokens)
	params['vocab_size'] = len(vocab)
	training_set = PostData("training_data.csv", preprocessor, vocab)
	val_set = PostData("validation_data.csv", preprocessor, vocab)
	train_loader = DataLoader(training_set, batch_size=params['batch_size'], shuffle=True, num_workers=0, collate_fn=collateFunction, pin_memory=True)
	test_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=0, collate_fn=collateFunction, pin_memory=True)
	model = AdverserialModel(**params).cuda()
	# Setup model
	best_model = None
	best_test_acc = 0
	# Setup training loop
	num_epochs = 25
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	for epoch in tqdm(range(num_epochs)):
		train_loss = 0.0
		train_gender_correct = 0
		train_board_correct = 0
		test_gender_correct = 0
		test_board_correct = 0
		test_loss = 0.0
		test_total = 0
		train_total = 0
		for inputs, labels in train_loader:
			# zero the parameter gradients
			#import pdb;pdb.set_trace()
			optimizer.zero_grad()
			inputs = inputs.cuda()
			labels = labels.cuda()
			# forward + backward + optimize
			gender_logits, board_logits = model(inputs)
			loss = criterion(board_logits, labels[:, 1]) - criterion(outputs[0], labels[:, 0])
			_, gender_pred = torch.max(gender_logits, 1)
			_, board_pred = torch.max(board_logits, 1)
			loss.backward()
			optimizer.step()
			train_total += labels.size(0)
			# print statistics
			train_gender_correct += (gender_pred == labels[:, 0]).sum().item()
			train_board_correct += (board_pred == labels[:, 1]).sum().item()
			train_loss += loss.item()
		print("Train accuracy is {} with a loss of {}".format(train_gender_correct/train_total,train_board_correct/train_total))
		with torch.no_grad():
			for inputs, labels in test_loader:
				# zero the parameter gradients
				inputs = inputs.cuda()
				labels = labels.cuda()
				optimizer.zero_grad()

				# forward + backward + optimize
				gender_logits, board_logits = model(inputs)
				loss = criterion(board_logits, labels[:, 1]) - criterion(gender_logits, labels[:, 0])
				_, gender_pred = torch.max(gender_logits, 1)
				_, board_pred = torch.max(board_logits, 1)
				# print statistics
				test_gender_correct += (gender_pred == labels[:, 0]).sum().item()
				test_board_correct += (board_pred == labels[:, 1]).sum().item()
				test_loss += loss.item()
				test_total += labels.size(0)
		print("Test accuracy is {} with a loss of {}".format(test_gender_correct/test_total,test_board_correct/test_total))
		if best_test_acc <= (test_board_correct/test_total):
			best_model = deepcopy(model)
			best_test_acc = (test_board_correct/test_total)
	return best_model

def eval():
	pass
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--eval_file", default="./dataset.csv")
	parser.add_argument("--mode", default="train", choices=["train","split","eval"])
	args = parser.parse_args()
	
	if args.mode == "split":
		splitDataset(args.eval_file)
	elif args.mode == "train":
		params = {
			'batch_size' : 4,
			'hidden_dim' : 64,
			'embedding_dim' : 100,
			'agg_method' : 'last',
			'num_genders' : 2,
			'num_boards' : 2
		}
		train(**params)
	elif args.mode == "eval":
		eval()
		
if __name__ == "__main__":
	main()