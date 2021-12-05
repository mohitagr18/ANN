import tensorflow as tf
import time
import os

def create_model(loss_function, optimizer, metric, no_classes):
	# Define layers
	LAYERS = [
		tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
		tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
		tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
		tf.keras.layers.Dense(no_classes, activation="softmax", name="outputLayer")
	]
	# Build model
	model_clf = tf.keras.models.Sequential(LAYERS)
	# Print Summary
	model_clf.summary()
	# Compile model --> untrained
	model_clf.compile(loss=loss_function, optimizer=optimizer, metrics=metric)
	return model_clf

def get_unique_path(model_name):
	filename = time.strftime(f"%Y%m%d_%H%M%S_{model_name}")
	return filename

def save_model(model, model_name, model_dir):
	filename = get_unique_path(model_name)
	path = os.path.join(model_dir, filename)
	model.save(path)