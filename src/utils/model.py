import tensorflow as tf

def create_model(loss_function, optimizer, metric, no_classes):
	LAYERS = [
		tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
		tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
		tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
		tf.keras.layers.Dense(no_classes, activation="softmax", name="outputLayer")
	]

	model_clf = tf.keras.models.Sequential(LAYERS)
	model_clf.summary()
	print(model_clf.layers[1])
	print(type(model_clf))
	model_clf.compile(loss=loss_function, optimizer=optimizer, metrics=metric)
	return model_clf