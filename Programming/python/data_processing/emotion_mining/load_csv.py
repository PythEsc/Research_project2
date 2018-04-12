from importer.database.data_types import Post
from importer.database.mongodb import MongodbStorage

db = MongodbStorage()
#
# filter = {Post.COLL_POST_ID: '165836530143824_1281633928564073'}
# db.select_single_post()

file = 'C:/Users/blu/Dropbox/research_project/Emotions/safeway.csv'
with open(file, 'r') as file:
    for line in file.readlines():
        line = line.rstrip()
        if len(line) > 1:
            l = line.split(',')
            print(l[0])

            filter = {Post.COLL_POST_ID: l[0]}
            post = db.select_single_post(filter)
            post.comment_emotion = l[1:]

            print(post.comment_emotion)

            db.update_post(post)
