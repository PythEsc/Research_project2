from wikipedia import page
import Programming.importer.word2vec_utitlity as util
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("TrainingDataImporter")


class WikipediaPageProcessor:
    """

    """
    file_prefix = "wiki_"

    def __init__(self, titles, training_data_directory="../training/"):
        self.titles = titles
        self.source_files = training_data_directory+"sources/"
        self.preprocessed_files = training_data_directory+"processed/"

    def download(self):
        logger.info("Start downloading pages from wikipedia: " + str(self.titles))
        for title in self.titles:
            filepath = self.source_files + self.file_prefix + title + ".txt"
            if os.path.exists(filepath):
                logger.info("File " + title + " already exists in target directory. Skipping file")
                continue

            logger.info("Loading wikipage: " + title)
            wikipage = page(title, auto_suggest=False)
            content = wikipage.content
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)

        logger.info("Finished downloading wikipedia articles")

    def process(self):
        stop = util.get_stopwords()

        logger.info("Loading source files for pre-processing")
        for title in self.titles:

            target_filepath = self.preprocessed_files + self.file_prefix + title + ".txt"
            source_filepath = self.source_files + self.file_prefix + title + ".txt"

            if os.path.exists(target_filepath):
                logger.info("File " + title + " already exists in target directory. Skipping file")
                continue

            logger.info("Pre-processing file: " + title)

            if not os.path.exists(source_filepath):
                logger.warning("Sourcefile " + title + " does not exist. Skipping...")
                continue

            with open(source_filepath, "r") as f_in:

                os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
                with open(target_filepath, "w") as f_out:
                    for line in f_in:
                        cleaned_content = util.clean_str(line)
                        cleaned_words = cleaned_content.split()

                        tokens = [word for word in cleaned_words if word not in stop]
                        f_out.write(" ".join(tokens) + os.linesep)

        logger.info("Pre-processing finished.")
