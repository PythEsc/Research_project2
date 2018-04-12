from neural_networks.nn_implementations.keras_TextRNNCNN import TextRNNCNN_Keras
from neural_networks.util import configuration_loader

if __name__ == '__main__':
    config = configuration_loader.load_config(path="../config/neural_nets/parameters.json")

    # Start mongo with "sudo service mongod start"
    cnn = TextRNNCNN_Keras(settings=config)
    cnn.train()
