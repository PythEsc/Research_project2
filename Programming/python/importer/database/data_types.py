class Post:
    """
    Holder class that contains the data of one (processed) facebook post
    """
    COLL_POST_ID = "_id"
    COLL_USER_ID = "user_id"
    COLL_MESSAGE = "message"
    COLL_DATE = "date"
    COLL_LINK = "link"
    COLL_REACTIONS = "reactions"

    VALID_COLUMNS = [COLL_POST_ID, COLL_USER_ID, COLL_MESSAGE, COLL_DATE, COLL_LINK, COLL_REACTIONS]
    MANDATORY_COLUMNS = [COLL_POST_ID, COLL_USER_ID, COLL_MESSAGE, COLL_DATE, COLL_LINK, COLL_REACTIONS]

    def __init__(self, structure: dict):
        self.__check_post(structure)
        self.data = structure

    def __check_post(self, post: dict):
        """
        Check that the dictionary can be parsed to a valid post (that it contains all the necessary data and no data that is invalid)
        
        :param post: The dictionary that contains the post's data
        """
        for key, value in post.items():
            assert key in self.VALID_COLUMNS, "Post contains invalid key: '{key}'".format(key=key)
            assert isinstance(value, (str, dict)), "Post invalid. The entry for the key '{key}' is invalid: '{value}'" \
                .format(key=key,
                        value=value)

        for key in self.MANDATORY_COLUMNS:
            assert key in post, "Mandatory key missing in post: '{key}'".format(key=key)

    @staticmethod
    def create_from_single_values(post_id: str, user_id: str, message: str, date: str, link: str, reactions: dict):
        """
        Creates a post object from single post values
        
        :param post_id: The id of the post
        :param user_id: The id of the user that created the post
        :param message: The message/content of the post 
        :param date: The date of the post
        :param link: The facebook link of the post
        :param reactions: The facebook user reactions
        :return: A Post object
        """

        data = {Post.COLL_POST_ID: post_id,
                Post.COLL_USER_ID: user_id,
                Post.COLL_MESSAGE: message,
                Post.COLL_DATE: date,
                Post.COLL_LINK: link,
                Post.COLL_REACTIONS: reactions}

        return Post(data)

    @property
    def post_id(self) -> str:
        return self.data[Post.COLL_POST_ID]

    @property
    def user_id(self) -> str:
        return self.data[Post.COLL_USER_ID]

    @property
    def message(self) -> str:
        return self.data[Post.COLL_MESSAGE]

    @property
    def date(self) -> str:
        return self.data[Post.COLL_DATE]

    @property
    def link(self) -> str:
        return self.data[Post.COLL_LINK]

    @property
    def reactions(self) -> dict:
        return self.data[Post.COLL_REACTIONS]


class Comment:
    """
    Holder class that contains the data of one (processed) facebook post
    """
    COLL_ID = "_id"
    COLL_PARENT_ID = "parent_id"
    COLL_USER_ID = "user_id"
    COLL_CONT = "content"
    COLL_DATE = "date"

    VALID_COLUMNS = [COLL_ID, COLL_PARENT_ID, COLL_USER_ID, COLL_CONT, COLL_DATE]
    MANDATORY_COLUMNS = [COLL_ID, COLL_PARENT_ID, COLL_USER_ID, COLL_CONT, COLL_DATE]

    def __init__(self, structure: dict):
        self.__check_comment(structure)
        self.data = structure

    def __check_comment(self, comment: dict):
        """
        Check that the dictionary can be parsed to a valid comment (that it contains all the necessary data and no data that is invalid)
        
        :param comment: The dictionary that contains the comment's data
        """
        for key, value in comment.items():
            assert key in self.VALID_COLUMNS, "Comment contains invalid key: '{key}'".format(key=key)
            assert isinstance(value, str), "Comment invalid. The entry for the key '{key}' is invalid: '{value}'".format(key=key,
                                                                                                                         value=value)

        for key in self.MANDATORY_COLUMNS:
            assert key in comment, "Mandatory key missing in comment: '{key}'".format(key=key)

    @staticmethod
    def create_from_single_values(comment_id: str, parent_id: str, user_id: str, content: str, date: str):
        """
        Creates a comment object from single comment values
        
        :param comment_id: The id of the comment
        :param parent_id: The id of the parent-comment (if there is no parent the id is -1)
        :param user_id: The id of the user that created this comment
        :param content: The content of the comment
        :param date: The date of the comment
        :return: A Comment object
        """

        data = {Comment.COLL_ID: comment_id,
                Comment.COLL_PARENT_ID: parent_id,
                Comment.COLL_USER_ID: user_id,
                Comment.COLL_CONT: content,
                Comment.COLL_DATE: date}

        return Comment(data)

    @property
    def id(self) -> str:
        return self.data[Comment.COLL_ID]

    @property
    def parent_id(self) -> str:
        return self.data[Comment.COLL_PARENT_ID]

    @property
    def fb_link(self) -> str:
        return self.data[Comment.COLL_CONT]

    @property
    def date(self) -> str:
        return self.data[Comment.COLL_DATE]
