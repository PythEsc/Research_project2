import json
import os
import sys

from neural_networks.nn_implementations.keras_TextCNN import TextCNN_Keras
from neural_networks.util.configuration_loader import DEFAULT_DICT


def create_parameters(learning_rate, conv_layer, use_bn, use_maxpooling):
    import datetime
    from dateutil.tz import tzlocal

    now = datetime.datetime.now(tzlocal())
    timestamp = now.strftime('%y%m%d_%H_%M_%S')
    time_string = "../../results/CNN/" + timestamp

    config_file_name = time_string + "/learning_rate{}_convlayer{}_usebn{}_usemaxpool{}.json".format(learning_rate,
                                                                                                     conv_layer, use_bn,
                                                                                                     use_maxpooling)
    open_file = None
    if not os.path.exists(config_file_name):
        if not os.path.exists(time_string):
            os.makedirs(time_string, mode=0o744)
        parameters = DEFAULT_DICT
        parameters["convolution_layers"] = conv_layer
        parameters["use_bn"] = use_bn
        parameters["use_max_pooling"] = use_maxpooling
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
    return parameters, open_file


def process_cnn_experiments():
    console_std_out = sys.stdout
    for learning_rate in [0.005, 0.001, 0.0005, 0.0001]:
        for conv_layer in [1, 2, 3]:
            for use_bn in [True]:
                for use_maxpooling in [True]:
                    sys.stdout = console_std_out
                    print(
                        "Start combination learning_rate: {}, conv_layer: {}, use_bn: {} and use_maxpooling: {}".format(
                            learning_rate, conv_layer, use_bn, use_maxpooling))
                    print("Training started!")
                    # Parameter creation/loading
                    parameters, open_file = create_parameters(learning_rate=learning_rate, conv_layer=conv_layer,
                                                              use_bn=use_bn, use_maxpooling=use_maxpooling)

                    # Start the network
                    cnn = TextCNN_Keras(parameters)
                    cnn.train()
                    open_file.close()

                    sys.stdout = console_std_out
                    print("Training finished!")
                    # Reset the Keras session, otherwise the last session will be used
                    from keras import backend as K
                    K.clear_session()
