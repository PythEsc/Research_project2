import math
import Programming.importer.word2vec_utitlity as util
from Programming.importer.DataImporter import DataImporter
from Programming.importer.word2vec_new import preprocessing_filter
from numpy import sum


def main():
    # Create our DataImporter for the Sainsbury data set
    importer_sainsbury = DataImporter("../Filtered/Sainsbury.zip", "../Unzipped/Sainsbury")
    importer_sainsbury.load()
    [posts, reactions_matrix] = importer_sainsbury.get_data_and_labels()
    stopwords = util.get_stopwords()

    [cleaned_posts, new_reactions_matrix] = [[], []]

    for i, post in enumerate(posts):
        reactions = reactions_matrix[i]
        cleaned_post = preprocessing_filter(post, stopwords)
        reaction_sum = sum(reactions)
        if math.isnan(reaction_sum) or reaction_sum < 1:
            continue

        new_reactions = []
        for reaction in reactions:
            new_reactions.append(reaction / reaction_sum)

        cleaned_posts.append(cleaned_post)
        new_reactions_matrix.append(new_reactions)

    print()

if __name__ == '__main__':
    main()
