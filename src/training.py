from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse

def training(config_path):
	config = read_config(config_path)

	# Get Data
	validation_datasize = config['params']['validation_datasize']
	(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_datasize)

	# Compile Model
	loss_function = config['params']['loss_functions']
	optimizer = config['params']['optimizer']
	metric = config['params']['metrics']
	no_classes = config['params']['no_classes']

	model = create_model(loss_function, optimizer, metric, no_classes)

	# Train model
	epochs = config['params']['epochs']
	validation_set = (x_valid, y_valid)
	history = model.fit(x_train, y_train,
	                    epochs=epochs, validation_data = validation_set)

if __name__ == '__main__':
	args = argparse.ArgumentParser()
	args.add_argument("--config", "-c", default="config.yaml")
	parsed_args = args.parse_args()
	training(parsed_args.config)