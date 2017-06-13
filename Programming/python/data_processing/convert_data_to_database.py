from python.importer.database.data_types import Emotion, Post, Comment
from python.importer.database.mongodb import MongodbStorage
import csv

def read_data(database : MongodbStorage, supermarket : str, prefix : str):
    base_string = "../../../data/Filtered/" + supermarket + "/";
    months = ["March/", "April/", "May/", "June/", "July/", "August/", "September/", "October/", "November/", "December/"]
    count = 0
    for month in months:
        # COMMENTS
        f = open(base_string + prefix + "_" + month + "comments.csv", "r")
        reader = csv.reader(f,  delimiter=';')
        for row in reader:
            try:
                user_id = row[1].split("_")[0]
                comment_object = Comment.create_from_single_values(comment_id=row[5], parent_id=row[1], content=row[8], date=row[9], user_id=user_id)
                database.insert_comment(comment_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1
        # FULLSTATS
        f = open(base_string + prefix + "_" + month + "fullstats.csv", "r")
        reader = csv.reader(f,  delimiter=';')
        for row in reader:
            try:
                user_id = row[1].replace("post_user_", "")
                post_object = Post.create_from_single_values(message=row[4], date=row[9], link=row[3], post_id=row[2],
                                                             reactions=[row[22], row[23], row[24], row[25], row[26],
                                                                        row[27]], user_id=user_id)
                database.insert_post(post_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1

    print("Could not integrate this many posts and comments: " + str(count) + " for " + supermarket)

if __name__ == '__main__':
    db = MongodbStorage()
    read_data(db, "Tesco", "TE")
