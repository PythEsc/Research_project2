import csv

from importer.database.data_types import Post, Comment
from importer.database.mongodb import MongodbStorage


def read_data(database: MongodbStorage, supermarket: str, prefix: str):
    base_string = "../../../../data/Filtered/" + supermarket + "/"
    months = ["March/", "April/", "May/", "June/", "July/", "August/", "September/", "October/", "November/",
              "December/"]
    count = 0
    for month in months:
        # COMMENTS
        f = open(base_string + prefix + "_" + month + "comments.csv", "r")
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                user_id = row[1].split("_")[0]
                comment_object = Comment.create_from_single_values(comment_id=row[5], parent_id=row[1], content=row[8],
                                                                   date=row[9], user_id=user_id)
                database.insert_comment(comment_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1
        # FULLSTATS
        f = open(base_string + prefix + "_" + month + "fullstats.csv", "r")
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                user_id = row[1].replace("post_user_", "")
                post_object = Post.create_from_single_values(message=row[4], date=row[9], link=row[3], post_id=row[2],
                                                             reactions={'LIKE': row[22], 'LOVE': row[23],
                                                                        'WOW': row[24], 'HAHA': row[25], 'SAD': row[26],
                                                                        'ANGRY': row[27], 'THANKFUL': row[28]},
                                                             user_id=user_id, off_topic=False)
                database.insert_post(post_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1

    print("Could not integrate this many posts and comments: " + str(count) + " for " + supermarket)


if __name__ == '__main__':
    db = MongodbStorage()
    read_data(db, "Tesco", "TE")
