import gensim
from utils import load_gendered_words


male_vocab, female_vocab = load_gendered_words()
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

word_distances = {}
def cachedSimilarityScore(word1,word2):
	word_pair = frozenset((word1,word2))
	if word_pair in word_distances:
		return word_distances[word_pair]
	else:
		distance = model.similarity(word1, word2)
		word_distances[word_pair] = distance
		return distance

genderLUT = {'M':{}, 'W':{}}
for word in male_vocab:
	if word in model:
		scores = [[cachedSimilarityScore(word, replacement_word),replacement_word] for replacement_word in female_vocab if replacement_word in model]
		ranked_words = sorted(scores, key=lambda x: x[0], reverse=True)
		genderLUT['M'][word] = ranked_words
	else :
		genderLUT['M'][word] = [(1, word)]
print("Finished male words")
# TODO: Consider adding caching using permutation tolerant datastructures (frozen set), it should cut the time about in half
for word in female_vocab:
	if word in model:
		scores = [[cachedSimilarityScore(word, replacement_word),replacement_word] for replacement_word in male_vocab if replacement_word in model]
		ranked_words = sorted(scores, key=lambda x: x[0], reverse=True)
		genderLUT['W'][word] = ranked_words
	else :
		genderLUT['W'][word] = [(1, word)]
#import pdb;pdb.set_trace()
import pickle
with open('semantic_map.pkl','wb+') as sem_fh:
	pickle.dump(genderLUT, sem_fh)