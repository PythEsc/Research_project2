from abc import ABC, abstractmethod

from python.importer.database.data_types import Post, Comment, Emotion


class DataStorage(ABC):
    """
    Abstract class / Interface that contains the method signatures needed for data access
    """

    @abstractmethod
    def update_post(self, post: Post):
        """
        Updates an already existing single Post
        
        :param post: The new Post
        """
        pass

    @abstractmethod
    def insert_post(self, post: Post):
        """
        Inserts a new Post into the data
        
        :param post: The new Post
        """
        pass

    @abstractmethod
    def insert_comment(self, comment: Comment):
        """
        Inserts a new Comment into the data
        
        :param comment: The new Comment
        """
        pass

    @abstractmethod
    def select_single_post(self, filter: dict) -> Post:
        """
        Selects a single post. This method is limited to one post, if the filter is not unique it will still return only the first 
        matching entry
        
        :param filter: Filter to search for
        :return: A single Post object
        """
        pass

    @abstractmethod
    def select_multiple_posts(self, filter: dict) -> list:
        """
        Selects all posts matching the given <filter>
        
        :param filter: The filter to search for
        :return: A list containing Post objects matching the required filter
        """
        pass

    @abstractmethod
    def iterate_batch_post(self, filter: dict, batch_size: int) -> list:
        """
        Iterator that returns lists of Post objects matching the filter with a size of <batch_size>
        
        :param filter: The filter to search for
        :param batch_size: The size of the returned lists
        :return: A list with <batch_size> entries of Post objects with each iteration
        """
        pass

    @abstractmethod
    def iterate_single_post(self, filter: dict) -> list:
        """
        Iterator that returns a single Post object with each iteration
        
        :param filter: The filter to search for
        :return: A Post object with each iteration
        """
        pass

    @abstractmethod
    def count_posts(self, filter: dict) -> int:
        """
        Returns the amount of Posts matching the given <filter>
        
        :param filter: The filter to search for
        :return: The amount (int) of Posts matching the filter
        """
        pass

    @abstractmethod
    def count_comments(self, filter: dict) -> int:
        """
        Returns the amount of Comments matching the given <filter>

        :param filter: The filter to search for
        :return: The amount (int) of Comments matching the filter
        """
        pass

    @abstractmethod
    def insert_emotion(self, emotion: Emotion):
        """
        Inserts a new emotion into the data

        :param emotion: The new emotion
        """
        pass

    @abstractmethod
    def iterate_single_emotion(self, filter: dict, print_progress: bool = True) -> Emotion:
        """
        Iterator that returns a single Emotion object with each iteration

        :param print_progress: If true the progress of iterating will be printed
        :param filter: The filter to search for
        :return: A Emotion object with each iteration
        """
        pass

    def select_single_emotion(self, filter: dict) -> Emotion:
        """
        Returns a single emotion matching the given filter

        :param filter: The filter to search for 
        :return: A single Emotion object
        """
        pass