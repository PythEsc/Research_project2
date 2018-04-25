import json
import os
from collections import OrderedDict

DEFAULT_DICT = OrderedDict()

DEFAULT_DICT["checkpoint_path"] = "../results"  # Percentage of the training data to use for validation
# DEFAULT_DICT["dev_sample_percentage"] = .1  # Percentage of the training data to use for validation
# DEFAULT_DICT["positive_data_file"] = "./data/rt-polaritydata/rt-polarity.pos"  # Data source for the positive data.
# DEFAULT_DICT["negative_data_file"] = "./data/rt-polaritydata/rt-polarity.neg"  # Data source for the negative data.

# Model Hyperparameters
DEFAULT_DICT["embedding_dim"] = 50  # Dimensionality of character embedding (default: 128)
# DEFAULT_DICT["filter_sizes"] = "3,4,5"  # Comma-separated filter sizes (default: '3,4,5')
DEFAULT_DICT["num_filters"] = 40  # Number of filters per filter size (default: 128)
DEFAULT_DICT["kernel_size"] = 4  # The size of the kernel used in 2D Convolution (default: Unknown)
DEFAULT_DICT["dropout_keep_prob"] = 0.5  # Dropout keep probability (default: 0.5)
# DEFAULT_DICT["l2_reg_lambda"] = 0.0  # L2 regularization lambda (default: 0.0)
DEFAULT_DICT["lstm_layers"] = 2  # LSTM Layer (default: 2)
DEFAULT_DICT["convolution_layers"] = 1  # LSTM Layer (default: 2)
DEFAULT_DICT["use_max_pooling"] = True  # Add MaxPool Layer (default: True)
DEFAULT_DICT["use_bn"] = False  # Add BN Layer (default: False)

# Training parameters
DEFAULT_DICT["batch_size"] = 64  # Batch Size (default: 64)
DEFAULT_DICT["num_epochs"] = 30  # Number of training epochs (default: 200)
DEFAULT_DICT["allow_augmented_data"] = True  # Do also use data that was created with data_augmentation (default: true)


# Misc Parameters
# DEFAULT_DICT["allow_soft_placement"] = True  # Allow device soft device placement
# DEFAULT_DICT["log_device_placement"] = False  # Log placement of ops on devices


def load_config(path: str) -> dict:
    if not os.path.isfile(path=path):
        raise ValueError("The given path '{}' does not exist.".format(os.path.abspath(path)))

    with open(path) as config_file:
        config_dict = json.load(config_file)

        for key, value in DEFAULT_DICT.items():
            if key not in config_dict:
                config_dict[key] = value

        return config_dict


def create_default_config(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as config_file:
        json.dump(DEFAULT_DICT, config_file)


def create_config_from_dict(path: str, dictionary: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as config_file:
        json.dump(dictionary, config_file)


if __name__ == '__main__':
    create_default_config(path="../config/neural_nets/parameters.json-dist")
