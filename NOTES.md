# Installing dependencies: based on python 3.6 - macbook pro

    conda create -n py36-keras python=3.6
    conda activate py36-keras
    pip install numpy scipy scikit-learn pillow h5py
    pip install theano
    pip install tensorflow
    pip install keras
    pip install matplotlib 

#Hello Theano

>>> import theano
WARNING (theano.configdefaults): install mkl with `conda install mkl-service`: No module named 'mkl'
>>> import theano.tensor as T
>>> x = T.dmatrix('x')
>>> s = 1 / (1 + T.exp(-x))
>>> logistic = theano.function([x], s)
>>> logistic([[0, 1],[-1, -2]])
array([[0.5       , 0.73105858],
       [0.26894142, 0.11920292]])
>>> 

#Hello MNIST

>>> import tensorflow as tf
>>> mnist = tf.keras.datasets.mnist
>>> (x_train, y_train), (x_test, y_test) = mnist.load_data()
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 2s 0us/step
>>> x_train, x_test = x_train / 255.0, x_test / 255.0
>>> 


vim $HOME/.keras/keras.json 

{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}

#useful commmands: saving models

# save as JSON 
json_string = model.to_json()

# save as YAML 
yaml_string = model.to_yaml() 

# model reconstruction from JSON: 

from keras.models import model_from_json 
model = model_from_json(json_string) 

# model reconstruction from YAML 
model = model_from_yaml(yaml_string)

from keras.models import load_model 
model.save('my_model.h5') # creates a HDF5 file 'my_model.h5' 
del model # deletes the existing model

# returns a compiled model
# identical to the previous one 

model = load_model('my_model.h5')

# useful commands: stop when model stops improving
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,  patience=0, verbose=0, mode='auto')

# useful commands: save loss history
class LossHistory(keras.callbacks.Callback):     

	def on_train_begin(self, logs={}):         
		self.losses = []     
	
	def on_batch_end(self, batch, logs={}):        
		self.losses.append(logs.get('loss')) 
	
model = Sequential() 
model.add(Dense(10, input_dim=784, init='uniform')) 
model.add(Activation('softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop') history = LossHistory() 
model.fit(X_train,Y_train, batch_size=128, nb_epoch=20,  verbose=0, callbacks=[history]) 

print history.losses

# Checkpointing: save weights periodically as backup, save the best
