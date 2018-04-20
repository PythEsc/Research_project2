import neural_networks.experiments.run_experiments as exp
from neural_networks.nn_implementations.keras_TextCNN import TextCNN_Keras
from neural_networks.nn_implementations.keras_TextRNNCNN import TextRNNCNN_Keras
from neural_networks.util import configuration_loader

if __name__ == '__main__':
    start_experiments = True

    if start_experiments:
        exp.process_experiments(TextCNN_Keras, exp.create_parameters_cnn)
    else:
        config = configuration_loader.load_config(path="../config/neural_nets/parameters.json")

        # Start mongo with "sudo service mongod start"
        combined = TextRNNCNN_Keras(settings=config)
        combined.train()

        content = ["I really love your delicious orange juice",
                   "This cookies were so old and dry they are disgusting :(",
                   "The employees in this supermarket are so rude."]

        predicted_reactions = combined.predict(content=content)

        for i, r in enumerate(content):
            print('{}:\n{}'.format(r, predicted_reactions[i]))
