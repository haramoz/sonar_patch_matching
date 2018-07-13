# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',', skiprows=1)

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=200, batch_size=5, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
