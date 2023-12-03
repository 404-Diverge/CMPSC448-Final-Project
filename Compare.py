from nltk.corpus import ConllCorpusReader


def load_data(file_path):
    corp = ConllCorpusReader('.', file_path, ('words', 'pos'))
    sents = corp.tagged_sents()
    sentences = []
    pos_tags = []
    for sent in sents:
        words = []
        tags = []
        for word, tag in sent:
            words.append(word)
            tags.append(tag)
        sentences.append(words)
        pos_tags.append(tags)
    return sentences, pos_tags


def compare_files(file1, file2, type='rnn'):
    sentences1, pos_tags1 = load_data(file1)
    sentences2, pos_tags2 = load_data(file2)

    total_count = 0
    difference_count = 0
    for sent1, sent2, pos1, pos2 in zip(sentences1, sentences2, pos_tags1, pos_tags2):
        for word1, word2, tag1, tag2 in zip(sent1, sent2, pos1, pos2):
            total_count += 1
            if word1 != word2 or tag1 != tag2:
                difference_count += 1
                # print(f'Difference found: {word1}/{tag1} vs {word2}/{tag2}')
    print(f'{type} model comparison:')
    print(f'Total lines: {total_count}')
    print(f'Different lines: {difference_count}')
    print(f'Percentage of differences: {(difference_count / total_count) * 100}%')
    print(f'Correctness: {(1 - (difference_count / total_count)) * 100}%')
    print('=' * 80)


Answer_file1_path = 'Answer/in_domain_test_with_label.txt'
Answer_file2_path = 'Answer/LLaMA.test.txt'

cnn_result_file1_path = 'Predict/cnn_predict1.txt'
rnn_result_file1_path = 'Predict/rnn_predict1.txt'

cnn_result_file2_path = 'Predict/cnn_predict2.txt'
rnn_result_file2_path = 'Predict/rnn_predict2.txt'

nltk_result_file1_path = 'Predict/nltk_predict1.txt'
nltk_result_file2_path = 'Predict/nltk_predict2.txt'

compare_files(Answer_file1_path, cnn_result_file1_path, 'cnn')
compare_files(Answer_file1_path, rnn_result_file1_path, 'rnn')

compare_files(Answer_file2_path, cnn_result_file2_path, 'cnn')
compare_files(Answer_file2_path, rnn_result_file2_path, 'rnn')

