import re
import string
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout


def Pretreatment(text):
    STEMMER = PorterStemmer()
    PUNCT_TO_REMOVE = string.punctuation
    STOPWORDS = set(stopwords.words("english"))
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([STEMMER.stem(word) for word in text.split()])
    return text


def Labeltreat(label: str):
    if label == 'ham':
        return 0
    else:
        return 1


#  读取数据及预处理
Entire_Dataset = pd.read_csv("data/train.csv", encoding='utf-8')
Entire_Dataset['Email'] = Entire_Dataset['Email'].apply(Pretreatment)
Entire_Dataset['Label'] = Entire_Dataset['Label'].apply(Labeltreat)

#  语料库
df = Entire_Dataset
corpus = []
for i in tqdm(df['Email']):
    words = [word.lower() for word in word_tokenize(i)]
    corpus.append(words)

#  加载Glove词向量
embedding_dict = {}
with open("glove/glove.6B.100d.txt", 'r', encoding='utf-8')as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors

#  配置模型
MAX_LEN = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)
tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
word_index = tokenizer_obj.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

#  编码矩阵
for word, i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

#  构建模型
model = Sequential()
embedding = Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix),
                      input_length=MAX_LEN, trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#  划分
train_data = tweet_pad[:3000]
test_data = tweet_pad[3000:]
X_train, X_test, y_train, y_test = train_test_split(train_data, Entire_Dataset[:3000]['Label'].values, test_size=0.2)

#  训练评估
glove_model = model.fit(X_train, y_train, batch_size=4, epochs=2, validation_data=(X_test, y_test), verbose=2)
loss, accuracy = model.evaluate(test_data, Entire_Dataset[3000:]['Label'], batch_size=4, verbose=2)
print("Deep learning LSTM:")
print("loss:", loss)
print("accuracy: ", accuracy)
