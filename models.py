# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """

        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class NNSupportingClass(nn.Module):

    def __init__(self, embeddings: nn.Embedding, dimension_hid: int):
        super().__init__()

        self.embeddings = embeddings
        self.hidden_layer_dimension = dimension_hid
        self.in_features = embeddings.weight.size()[1]
        self.linear_transformation = nn.Linear(self.in_features, self.hidden_layer_dimension)
        self.relu = nn.Tanh()
        self.classifier = nn.Linear(self.hidden_layer_dimension, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, index) -> torch.Tensor:

        #Get the embedded words
        embedded_word = self.embeddings(index)
        embed_word_mean = embedded_word.mean(axis=0)

        #linearly tranform the dimensions
        linearly_transformed = self.linear_transformation(embed_word_mean)

        #apply non-linear transformation
        non_linear_transformation = self.relu(linearly_transformed)
        non_linear_output = self.classifier(non_linear_transformation)

        #apply softmax
        softmax_applied = self.softmax(non_linear_output)

        return softmax_applied


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, data, embeddings):
        self.embedded_data = embeddings
        self.model_data = data

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        def get_index(word):
            index = self.embedded_data.word_indexer.index_of(word)
            if index == -1:
                return 1
            else:
                return index

        index_input = torch.tensor([get_index(word) for word in ex_words])
        result = self.model_data(index_input)
        return torch.argmax(result, ).item()


def train_deep_averaging_network(args, train_examples: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    total_epochs = args.num_epochs
    hidden_dim = args.hidden_size
    if torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device('cpu')

    if not args.use_typo_setting:
        embeddings = word_embeddings.get_initialized_embedding_layer()
    else:
        #prefix_word_embeddings = Prefix_WordEmbeddings()
        word_embeddings = read_prefix_word_embeddings(args.word_vecs_path)
        embeddings = word_embeddings.get_initialized_embedding_layer()

    learning_rate = args.lr

    def get_word_index(word):
        index = word_embeddings.word_indexer.index_of(word)
        return 1 if index == -1 else index

    random.seed(2)
    random.shuffle(train_examples)

    nn_supporting_model = NNSupportingClass(embeddings, hidden_dim)
    nn_supporting_model.to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_supporting_model.parameters(), lr=learning_rate)

    for epoch_count in range(total_epochs):
        nn_supporting_model.train()
        for example in train_examples:
            if not args.use_typo_setting:
                index = torch.tensor([get_word_index(word) for word in example.words])
            else:
                index = torch.tensor([get_word_index(word[:3]) for word in example.words])
            input_label = torch.tensor(example.label)
            preds = nn_supporting_model(index)
            loss = criterion(preds, input_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return NeuralSentimentClassifier(nn_supporting_model, word_embeddings)

class Prefix_WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True):
        """
        :param frozen: True if you want the embedding layer to stay frozen, false to fine-tune embeddings
        :return: torch.nn.Embedding layer you can use in your network
        """
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_prefix_word_embeddings(embeddings_file: str) -> Prefix_WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    records = {}
    counter = {}
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word = line[:space_idx][:3]
            if word_indexer.contains(word) is True:
                word_index = word_indexer.index_of(word)
                records[word_index] += vector
                counter[word_index] += 1

            elif word_indexer.contains(word) is False:
                word_indexer.add_and_get_index(word)
                word_index = word_indexer.index_of(word)
                # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
                # of the file to see what the embedding dim is
                if len(vectors) == 0:
                    vectors.append(np.zeros(vector.shape[0]))
                    vectors.append(np.zeros(vector.shape[0]))
                records[word_index] = vector
                counter[word_index] = 1

    f.close()

    for key, value in records.items():
        vector = value/counter[key]
        vectors.append(vector)

    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return Prefix_WordEmbeddings(word_indexer, np.array(vectors))
