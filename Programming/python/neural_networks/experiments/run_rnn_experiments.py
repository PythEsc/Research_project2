import os
import json
import sys


from neural_networks.nn_implementations.keras_TextRNN import TextRNN_Keras
from neural_networks.util.configuration_loader import DEFAULT_DICT


def create_parameters(learning_rate, lstm_layers):
    import datetime
    from dateutil.tz import tzlocal
    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/RNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_lstmlayers{}.json".format(learning_rate, lstm_layers)
    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["lstm_layers"] = lstm_layers
        parameters["checkpoint_path"] = time_string + "/checkpoints"
        parameters["learning_rate"] = learning_rate

        # Redirect the console prints to the log file for better evaluation later
        open_file = open(time_string + "/logs.txt", 'w')
        sys.stdout = open_file

        # Save this dict to the json file in the results folder to trace the results later
        with open(config_file_name, 'x') as outfile:
            json.dump(parameters, outfile)
    else:
        with open(config_file_name, 'r') as outfile:
            parameters = json.load(outfile)
        open_file = None
    return parameters, open_file


def process_rnn_experiments():
    """
    This method is executing an experiment that tests different lengths of the signal dimension
    (length of the trajectory).
    """
    console_std_out = sys.stdout
    for learning_rate in [0.005, 0.001, 0.0005, 0.0001]:
        for lstm_layers in [1, 2, 3, 4]:
            sys.stdout = console_std_out
            print("Start combination learning_rate: {}, lstm_layers: {}".format(learning_rate, lstm_layers))
            print("Training started!")
            # Parameter creation/loading
            parameters, open_file = create_parameters(learning_rate=learning_rate, lstm_layers=lstm_layers)

            # Start the network
            cnn = TextRNN_Keras(parameters)
            cnn.train()
            if open_file is not None:
                open_file.close()

            sys.stdout = console_std_out
            print("Training finished!")
            # Reset the Keras session, otherwise the last session will be used
            from keras import backend as K
            K.clear_session()
