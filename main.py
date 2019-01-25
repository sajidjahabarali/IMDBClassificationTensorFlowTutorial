#import tensorflow library
import tensorflow as tf
from tensorflow import keras

#import numpy library
import numpy

#import imdb dataset which is included with keras
imdbDataset = keras.datasets.imdb

#download data from keras. num_word=10000 keeps the top 10000 most frequently occurring words in the training data.
(traindata, trainlabels), (testdata, testlabels) = imdbDataset.load_data(num_words=10000)

#printing the text of the padded training and testing data
#print(len(traindata[0]), len(testdata[1]))

print("Training entries: {}, labels: {}".format(len(traindata), len(trainlabels)))

wordIndex = imdbDataset.get_word_index()
wordIndex = {k:(v + 3) for k, v in wordIndex.items()}
wordIndex["<PAD>"] = 0 #padding
wordIndex["<START>"] = 1 #start of review
wordIndex["<UNK>"] = 2 #unknown
wordIndex["<UNUSED>"] = 3 #unused



#creating a reverse word index to return integer values back into words.
reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

#creating method which can be called to return the corresponding word for a specific integer.
def decode_review(review):
    return ' '.join([reverseWordIndex.get(i, '?') for i in review])

#printing the text for the first review in the dataset
#print(decode_review(traindata[0]))

#padding the train and test data so that they can be used as input for the neural network.
traindata = keras.preprocessing.sequence.pad_sequences(traindata, value=wordIndex["<PAD>"], padding = 'post', maxlen=256)
testdata = keras.preprocessing.sequence.pad_sequences(testdata, value=wordIndex["<PAD>"], padding = 'post', maxlen=256)

#printing the text of the padded training and testing data
#print(len(traindata[0]), len(testdata[1]))

#size of data is stored in a variable ie 10000 words.
vocabSize = 10000

model = keras.Sequential()
#first layer is an embedded layer which takes the 1000 words and decides their vectors in 16 dimensions.
model.add(keras.layers.Embedding(vocabSize, 16))
#flatten the 16 dimension vectors into a 1-dimensional vector
model.add(keras.layers.GlobalAveragePooling1D())
#the one dimensional vector is then fed into a dense layer with 16 nodes, since there were 16 dimensions
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#the dense layer then outputs to a 1 node layer which has an activation of a sigmoid which compresses the result to a value between 0 and 1, where 1 is a good review and 0 is a bad review.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#print(model.summary())

#create optimizer and loss function in order to allow our model to learn from the training data.
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#create validation set to check the accuracy of the model on data it has not had access to before.
xVal = traindata[:10000]
partialXTrain = traindata[10000:]
yVal = trainlabels[:10000]
partialYTrain = trainlabels[10000:]

#train model for 40 epochs in batches of 512 samples.
history = model.fit(partialXTrain, partialYTrain, epochs=40, batch_size=512, validation_data=(xVal, yVal), verbose=1)

#print model results
results = model.evaluate(testdata, testlabels)
print(results)