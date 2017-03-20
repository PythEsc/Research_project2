import os


class LineIterator:
    """
    A class that iterates over each file in a directory row per row
    """
    def __init__(self, dirname="../training/processed"):
        """
        :param dirname: The directory name that contains the input files
        """
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
