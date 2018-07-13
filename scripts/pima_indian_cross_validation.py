# organize imports
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# seed for reproducing same results
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split the data into training (67%) and testing (33%)
#(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=8, activation='relu'))
	model.add(Dense(6, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 

created_model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=10, verbose=1)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(created_model, X, Y, cv=kfold)
print(results.mean())
