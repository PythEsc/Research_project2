import pymongo

from python.importer.database.data_types import Post, Comment
from python.importer.database.database_access import DataStorage


class MongodbStorage(DataStorage):
    # Tables
    TABLE_POSTS = "posts"
    TABLE_COMMENTS = "comments"

    def __init__(self, host="localhost", port=27017, database="research_project"):
        self.client = pymongo.MongoClient(host=host, port=port)
        self.db = self.client[database]

    ###########################################################################
    # Post-methods
    ###########################################################################

    def count_posts(self, filter: dict) -> int:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        count = post_collection.count(filter)
        return count

    def insert_post(self, post: Post):
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        post_collection.insert_one(post.data)

    def iterate_batch_post(self, filter: dict, batch_size: int) -> list:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        cursor = post_collection.find(filter=filter, no_cursor_timeout=True)

        batch = []
        counter = 0
        size = cursor.count()
        for entry in cursor:
            batch.append(Post(entry))
            counter += 1
            print("\r%.2f%%" % (counter / size * 100), end='')
            if len(batch) is batch_size:
                yield batch
                batch = []

        cursor.close()
        print("\n")
        yield batch
        return

    def iterate_single_post(self, filter: dict) -> list:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        cursor = post_collection.find(filter=filter, no_cursor_timeout=True).batch_size(100)

        counter = 0
        size = cursor.count()
        for entry in cursor:
            counter += 1
            print("\r%.2f%%" % (counter / size * 100), end='')
            yield Post(entry)
        cursor.close()
        print("\n")

    def select_multiple_posts(self, filter: dict) -> list:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        cursor = post_collection.find(filter)
        posts = []
        for entry in cursor:
            posts.append(Post(entry))
        cursor.close()
        return posts

    def select_single_post(self, filter: dict) -> Post:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        result = post_collection.find_one(filter)
        return Post(result) if result is not None else None

    def select_newest_post(self) -> Post:
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        cursor = post_collection.find({}).sort(Post.COLL_DATE, pymongo.DESCENDING).limit(1)
        post = None
        for entry in cursor:
            post = Post(entry)
            break
        return post

    def update_post(self, post: Post):
        post_collection = self.db[MongodbStorage.TABLE_POSTS]
        post_collection.update_one({'_id': post.post_id}, {'$set': post.data})

    ###########################################################################
    # Comment-methods
    ###########################################################################

    def insert_comment(self, comment: Comment):
        comment_collection = self.db[MongodbStorage.TABLE_COMMENTS]
        comment_collection.insert_one(comment.data)

    def count_comments(self, filter: dict) -> int:
        comment_collection = self.db[MongodbStorage.TABLE_COMMENTS]
        count = comment_collection.count(filter)
        return count
