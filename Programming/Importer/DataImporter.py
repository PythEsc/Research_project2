import zipfile
import glob
import pandas as pd
import errno
import os
import re


class DataImporter:

    def __init__(self, file_location, unzip_location):
        self.file_location = file_location
        self.unzip_location = unzip_location
        self.data_storage = {}

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

        print("Trying to extract zip file (" + os.path.abspath(self.file_location) + ")")

        input_data = zipfile.ZipFile(self.file_location)
        input_data.extractall(self.unzip_location)
        input_data.close()

        print("Data successfully unzipped into " + os.path.abspath(self.unzip_location))

    def __load_data(self, criteria):
        filenames = glob.glob(self.unzip_location + "/**/*.csv", recursive=True)
        print("Found the following *.csv files: " + str(filenames))
        print("Start parsing...")

        for file in filenames:
            try:
                print("\t" + file)
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
                print("Error parsing file: " + file)

                raise

    def load(self, criteria):
        self.__unzip_file()
        self.__load_data(criteria)


# ----------- Some small test -------------

importer_sainsbury = DataImporter("../Filtered/Sainsbury.zip",
                                  "../Unzipped/Sainsbury")
importer_tesco = DataImporter("../Filtered/Tesco.zip",
                              "../Unzipped/Tesco")

# TODO: Add the function to filter for some criteria (e.g. only specific columns)
importer_sainsbury.load("put the criteria in here (not yet implemented)")
importer_tesco.load("put the criteria in here (not yet implemented)")

# Do anything with the data
