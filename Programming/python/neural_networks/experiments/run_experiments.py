import json
import os
import sys
import traceback

from custom_logging import Logger
from neural_networks.util.configuration_loader import DEFAULT_DICT


def create_parameters_rnncnn(learning_rate, num_layer):
    import datetime
    from dateutil.tz import tzlocal

    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/RNNCNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_numlayer{}.json".format(learning_rate, num_layer)

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

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "rnncnn")

        std_out = Logger.get(file_path=logging_path, classname="STDOUT", console_out=sys.stdout)
        err_out = Logger.get(file_path=logging_path, classname="STDERR", console_out=sys.stderr)

        def log_exception(type, value, tb):
            err_out.exception(
                "Uncaught exception {0}: {1}\n{2}".format(str(type.__name__), str(value),
                                                          ''.join(traceback.format_tb(tb))))

        sys.excepthook = log_exception
        sys.stdout = Logger(std_out.info)
        sys.stderr = Logger(err_out.warning)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters


def create_parameters_cnn(learning_rate, num_layer):
    import datetime
    from dateutil.tz import tzlocal

    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/CNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_convlayer{}.json".format(learning_rate, num_layer)

    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["convolution_layers"] = num_layer
        parameters["use_bn"] = False
        parameters["use_max_pooling"] = True
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "cnn")

        std_out = Logger.get(file_path=logging_path, classname="STDOUT", console_out=sys.stdout)
        err_out = Logger.get(file_path=logging_path, classname="STDERR", console_out=sys.stderr)

        def log_exception(type, value, tb):
            err_out.exception(
                "Uncaught exception {0}: {1}\n{2}".format(str(type.__name__), str(value),
                                                          ''.join(traceback.format_tb(tb))))

        sys.excepthook = log_exception
        sys.stdout = Logger(std_out.info)
        sys.stderr = Logger(err_out.warning)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters


def create_parameters_rnn(learning_rate, num_layer):
    import datetime
    from dateutil.tz import tzlocal
    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/RNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_lstmlayers{}.json".format(learning_rate, num_layer)
    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["lstm_layers"] = num_layer
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate

        # Redirect the console prints to the log file for better evaluation later
        logging_path = "{}/{}.log".format(time_string, "rnn")

        std_out = Logger.get(file_path=logging_path, classname="STDOUT", console_out=sys.stdout)
        err_out = Logger.get(file_path=logging_path, classname="STDERR", console_out=sys.stderr)

        def log_exception(type, value, tb):
            err_out.exception(
                "Uncaught exception {0}: {1}\n{2}".format(str(type.__name__), str(value),
                                                          ''.join(traceback.format_tb(tb))))

        sys.excepthook = log_exception
        sys.stdout = Logger(std_out.info)
        sys.stderr = Logger(err_out.warning)

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
    return parameters


def process_experiments(neural_network, create_parameters):
    console_std_out = sys.stdout
    for learning_rate in [0.001, 0.0005, 0.0001]:
        for num_layer in [1, 2, 3]:
            sys.stdout = console_std_out
            print("Start combination learning_rate: {}, layer: {}".format(learning_rate, num_layer, num_layer))
            print("Setup network")
            # Parameter creation/loading
            parameters = create_parameters(learning_rate=learning_rate, num_layer=num_layer)

            # Start the network
            nn = neural_network(parameters)

            print("Training started!")
            nn.train()

            sys.stdout = console_std_out
            print("Training finished!")
            # Reset the Keras session, otherwise the last session will be used
            from keras import backend as K
            K.clear_session()
