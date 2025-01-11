import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input , Dense 
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping 

data = pd.read_csv('breast-cancer.csv')
data = data.drop('id',axis=1)
lable_encoder = LabelEncoder()
data['diagnosis'] = lable_encoder.fit_transform(data['diagnosis'])
with open('diagnosis_encoder.pkl' , 'wb') as file:
    pickle.dump(lable_encoder,file)

x =  data.drop('diagnosis', axis=1)
y = data['diagnosis']
x_train, x_test , y_train ,y_test = train_test_split(x,y , test_size=.2 ,random_state=42)
scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)
x_test = scaller.transform(x_test)
with open('scaller.pkl' , 'wb') as file:
    pickle.dump(scaller ,file)

model = Sequential([
    Input(shape=[x_train.shape[1],]),
    Dense(16 ,activation = 'relu'),
    Dense(8 ,activation = 'relu'),
    Dense(1 ,activation = 'sigmoid'),
])
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer =opt , loss = 'binary_crossentropy' , metrics=['accuracy'])
early_stopping_callbck = EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
history = model.fit(x_train,y_train , validation_data = (x_test,y_test) , epochs =100, callbacks = [early_stopping_callbck])
model.save('brestcancerdetection.keras')