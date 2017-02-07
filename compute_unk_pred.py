from collections import defaultdict
import pickle


test_data = "../../data/multi/id/test.txt"
word_list = "../../data/multi/id/analysis/redup_words.txt"
output_vocab = "wiki_models/id/word/words_vocab.pkl"

vocab_dict = defaultdict(int)

with open(output_vocab, "rb") as f:
    word_to_id, _ = pickle.load(f)

with open(word_list) as f:
    for line in f:
        line = line.strip()
        word, freq = line.split()
        vocab_dict[word] = int(freq)


unk_freq = 0
unk_rare = 0
total_unk = 0
total_case = 0
total_tokens = 0

with open(test_data) as f:
    for line in f:
        line = line.strip().lower()
        for token in line.split():
            total_tokens += 1
            if token in vocab_dict:
                total_case += 1
                if token in word_to_id:
                    if word_to_id[token] > 5002:
                        total_unk += 1
                        if vocab_dict[token] < 10:
                            unk_rare += 1
                        else:
                            unk_freq += 1
                else:
                    total_unk += 1
                    unk_rare += 1


print("Total case: %.3f" % (total_case / total_tokens * 100))
print("Total UNK prediction: %.3f" % (total_unk / total_case * 100))
print("Freq UNK prediction: %.3f" % (unk_freq / total_unk * 100))
print("Rare UNK prediction: %.3f" % (unk_rare / total_unk * 100))





