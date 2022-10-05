#import needed modules
from sklearn.model_selection import train_test_split
import tensorflow
import pandas

#define the dataset used
dataset = pandas.read_excel('cancer.xlsx')

#split the data into the data and the diagnosis
data = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
diagnosis = dataset["diagnosis(1=m, 0=b)"]

#set 20% of data into the testing sample
data_train, data_test, diagnosis_train, diagnosis_test = train_test_split(data, diagnosis, test_size=0.2)

#start building neural network
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(256, input_shape=data_train.shape[1:], activation='sigmoid'))
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid'))
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

#comple the model and fit the data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data_train, diagnosis_train, epochs=1000)

print("\n")
model.evaluate(data_test, diagnosis_test) #evaluate with testing set
