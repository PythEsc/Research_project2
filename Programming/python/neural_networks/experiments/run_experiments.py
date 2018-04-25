import json
import logging
import os
import sys
from logging import DEBUG

from custom_logging import Logger
from neural_networks.util.configuration_loader import DEFAULT_DICT


def create_parameters_rnncnn(learning_rate, num_layer, dropout, file_handler):
    import datetime
    from dateutil.tz import tzlocal

    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/RNNCNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_numlayer{}_dropout{}.json".format(learning_rate, num_layer,
                                                                                         dropout)

    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["convolution_layers"] = num_layer
        parameters["lstm_layers"] = num_layer
        parameters["use_bn"] = False
        parameters["use_max_pooling"] = True
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate
        parameters["allow_augmented_data"] = True
        parameters["dropout_keep_prob"] = dropout

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "rnncnn")
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
        else:
            logging.getLogger().setLevel(DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            sys.stdout = Logger(logging.info)
            sys.stderr = Logger(logging.warning)

        file_handler = logging.FileHandler(logging_path)
        file_handler.setLevel(DEBUG)
        logging.getLogger().addHandler(file_handler)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters, file_handler


def create_parameters_cnn(learning_rate, num_layer, dropout, file_handler):
    import datetime
    from dateutil.tz import tzlocal

    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/CNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_numlayer{}_dropout{}.json".format(learning_rate, num_layer,
                                                                                         dropout)

    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["convolution_layers"] = num_layer
        parameters["use_bn"] = False
        parameters["use_max_pooling"] = True
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate
        parameters["allow_augmented_data"] = True
        parameters["dropout_keep_prob"] = dropout

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "cnn")
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
        else:
            logging.getLogger().setLevel(DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            sys.stdout = Logger(logging.info)
            sys.stderr = Logger(logging.warning)

        file_handler = logging.FileHandler(logging_path)
        file_handler.setLevel(DEBUG)
        logging.getLogger().addHandler(file_handler)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters, file_handler


def create_parameters_rnn(learning_rate, num_layer, dropout, file_handler):
    import datetime
    from dateutil.tz import tzlocal
    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/RNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_numlayer{}_dropout{}.json".format(learning_rate, num_layer,
                                                                                         dropout)
    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["lstm_layers"] = num_layer
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate
        parameters["allow_augmented_data"] = True
        parameters["dropout_keep_prob"] = dropout

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "rnn")
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
        else:
            logging.getLogger().setLevel(DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            sys.stdout = Logger(logging.info)
            sys.stderr = Logger(logging.warning)

        file_handler = logging.FileHandler(logging_path)
        file_handler.setLevel(DEBUG)
        logging.getLogger().addHandler(file_handler)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters, file_handler


def process_experiments(neural_network, create_parameters):
    f_handler = None
    for learning_rate in [0.001, 0.0001]:
        for num_layer in [1, 2]:
            for dropout in [0.3, 0.5]:
                print("Start combination learning_rate: {}, layer: {}, dropout: {}".format(learning_rate, num_layer,
                                                                                           dropout))
                print("Setup network")
                # Parameter creation/loading
                parameters, f_handler = create_parameters(learning_rate=learning_rate, num_layer=num_layer,
                                                          dropout=dropout, file_handler=f_handler)

                # Start the network
                nn = neural_network(parameters)

                print("Training started!")
                nn.train()

                print("Training finished!")

                nn.validate()

                # Reset the Keras session, otherwise the last session will be used
                from keras import backend as K
                K.clear_session()
