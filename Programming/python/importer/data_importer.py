import errno
import glob
import logging
import math
import os
import re
import zipfile

import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("DataImporter")


class DataImporter:
    """
    A helper class that loads the facebook data sets into memory and provides an iterator yielding one comment after another.
    """

    CONST_FULLSTATS = "fullstats"
    CONST_COMMENTS = "comments"
    CONST_INDEX_POST = 4
    CONST_INDEX_LIKE = 22
    CONST_INDEX_LOVE = 23
    CONST_INDEX_WOW = 24
    CONST_INDEX_HAHA = 25
    CONST_INDEX_SAD = 26
    CONST_INDEX_ANGRY = 27
    CONST_INDEX_REACT = [CONST_INDEX_LIKE, CONST_INDEX_LOVE, CONST_INDEX_WOW, CONST_INDEX_HAHA, CONST_INDEX_SAD, CONST_INDEX_ANGRY]

    def __init__(self, file_location, unzip_location):
        """

        :param file_location: The location of the zip-file containing the data
        :param unzip_location: The directory in which the zip shall be extracted
        """
        self.file_location = file_location
        self.unzip_location = unzip_location
        self.data_storage = {}

    def iterate(self, filter):
        for _, dataMap in self.data_storage.items():
            for comment_structure in dataMap["comments"]:
                comment = comment_structure[8]
                if not isinstance(comment, str):
                    continue

                yield filter(comment)

    def __unzip_file(self):
        # Check the file_location
        if not os.path.exists(self.file_location):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.file_location)
        if not os.path.isfile(self.file_location):
            raise ValueError(
                "The specified path does not point to a file. Path to file: " + os.path.abspath(self.file_location))
        if not zipfile.is_zipfile(self.file_location):
            raise ValueError("The specified file is no .zip file")

        # Check the unzip_location
        if not os.path.exists(self.unzip_location):
            os.makedirs(self.unzip_location)
        if not os.path.isdir(self.unzip_location):
            raise ValueError(
                "The specified unzip path is no directory. Unzip path: " + os.path.abspath(self.unzip_location))

        logger.info("Extracting zip file into " + os.path.abspath(self.unzip_location))
        with zipfile.ZipFile(self.file_location) as input_data:
            input_data.extractall(self.unzip_location)

    def __load_data(self):
        filenames = glob.glob(self.unzip_location + "/**/*.csv", recursive=True)
        logger.debug("Parsing files: " + str(filenames))

        for file in filenames:
            try:
                logger.info("Parsing file: " + file)
                data = pd.read_csv(file, delimiter=';').as_matrix()

                filename_formatted = file.replace("\\", "/")

                pattern = "[A-Za-z]+/[A-Za-z]+\.csv"
                matches = re.search(pattern, filename_formatted)

                if matches:
                    keymatch = matches.group(0)
                    keymatch = keymatch[:-4]
                    keys = keymatch.split("/")
                    if keys[0] not in self.data_storage:
                        self.data_storage[keys[0]] = {}

                    if keys[1] in self.data_storage[keys[0]]:
                        raise ValueError("An entry " + str(
                            keys) + " already existed. There seem to be two datasets for the same month and company")
                    else:
                        self.data_storage[keys[0]][keys[1]] = data
                else:
                    raise ValueError("File does not match the needed pattern: " + filename_formatted)
            except:
                logger.error("Error parsing file: " + file)

                raise
        logger.info("Finished parsing files")

    def load(self):
        """
        Loads the data into memory
        """
        self.__unzip_file()
        self.__load_data()

    def get_data_and_labels(self):

        y = []
        x_text = []
        for _, data in self.data_storage.items():
            fullstats = data[self.CONST_FULLSTATS]
            former_post = ""
            for post_struct in fullstats:
                post = post_struct[self.CONST_INDEX_POST]
                if post is former_post:
                    continue
                former_post = post
                x_text.append(post)
                y.append([post_struct[react] for react in self.CONST_INDEX_REACT])

        return [x_text, y]

    def prepare_data_for_CNN(self):
        [posts, reactions_matrix] = self.get_data_and_labels()

        [cleaned_posts, new_reactions_matrix] = [[], []]

        for i, post in enumerate(posts):
            reactions = reactions_matrix[i]
            reactions = reactions[1:]  # remove likes
            reaction_sum = sum(reactions)
            if math.isnan(reaction_sum) or reaction_sum < 15:
                continue

            new_reactions = []
            for reaction in reactions:
                new_reactions.append(reaction / reaction_sum)

            cleaned_posts.append(post)
            new_reactions_matrix.append(new_reactions)
        return [cleaned_posts, new_reactions_matrix]
