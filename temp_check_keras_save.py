import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(4, input_dim=2, init='normal', activation='relu'))
model.add(Dense(2, init='normal', activation='softmax'))
rmsprop = keras.optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=rmsprop)
model.save_weights('temp/temp_keras_model_weights.keras')
print '--- keras model weights saved. ---'
