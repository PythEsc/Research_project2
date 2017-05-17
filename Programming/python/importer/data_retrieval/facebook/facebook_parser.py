import datetime
import logging

from facepy.exceptions import FacebookError
from facepy.graph_api import GraphAPI


class FacebookParser:
    """
    Class for parsing Facebook data
    """
    CONST_ID = "id"
    CONST_FROM = "from"
    CONST_MESSAGE = "message"
    CONST_DATE = "created_time"
    CONST_PERMALINK = "permalink_url"
    CONST_COMMENTS = "comments"
    CONST_REACTIONS = "reactions"
    CONST_ATTACHED_OBJ_ID = "object_id"
    CONST_REACTIONS_TYPES = ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY', 'THANKFUL']

    LOGGER = logging.getLogger('FacebookParser')

    def __init__(self, auth_token: str):
        self.graph = GraphAPI(version='2.8', oauth_token=auth_token)

    def __get_page_id(self, name: str) -> str:
        """
        Returns the pageid of a Facebook site for a given page name
        
        :param name: The name of the site
        :return: The id of the site
        """
        url = '/{pagename}'.format(pagename=name)
        result = self.graph.get(path=url)
        return result[FacebookParser.CONST_ID]

    def __iterate_all_posts_from_page(self, pageid: str, since: int, until: int) -> list:
        """
        Iterates over all posts of a site with given <pageid> that were posted between <since> and <until>
        
        :param pageid: The id of the Facebook page
        :param since: The starting date of the interval in which the posts are crawled
        :param until: The end date of the interval in which the posts are crawled
        :return: A single post (dictionary) per iteration
        """
        fields = [FacebookParser.CONST_ID,
                  FacebookParser.CONST_MESSAGE,
                  FacebookParser.CONST_DATE,
                  FacebookParser.CONST_PERMALINK,
                  FacebookParser.CONST_COMMENTS,
                  FacebookParser.CONST_FROM,
                  FacebookParser.CONST_ATTACHED_OBJ_ID]

        field_query = ','.join(fields)
        try:
            url = "/{pageid}/feed?since={since}&until={until}&fields={fields}".format(since=since,
                                                                                      until=until,
                                                                                      pageid=pageid,
                                                                                      fields=field_query)
            response = self.graph.get(url)
            for data in response['data']:
                if FacebookParser.CONST_COMMENTS not in data:
                    continue
                data[FacebookParser.CONST_COMMENTS] = data[FacebookParser.CONST_COMMENTS]["data"]
                yield data
        except FacebookError as e:
            if e.code is 1 and "amount of data" in e.message:
                mid_of_window = int((since + until) / 2)
                mid_of_window_date = datetime.datetime.fromtimestamp(mid_of_window)
                FacebookParser.LOGGER.warning("Too many posts, trying with smaller window: [%s - %s] and [%s - %s]",
                                              datetime.datetime.fromtimestamp(since),
                                              mid_of_window_date, mid_of_window_date,
                                              datetime.datetime.fromtimestamp(until))
                for post in self.__iterate_all_posts_from_page(pageid=pageid, since=since, until=mid_of_window):
                    yield post
                for post in self.__iterate_all_posts_from_page(pageid=pageid, since=mid_of_window, until=until):
                    yield post
            else:
                raise e

    def __get_reactions_for_post(self, postid: str) -> dict:
        """
        Retrieves the reactions for a given post
        
        :param postid: The id of the post
        :return: A dictionary containing the reactions
        """
        query_elements = []
        for reaction in FacebookParser.CONST_REACTIONS_TYPES:
            query_elements.append(
                'reactions.type({reaction}).summary(total_count).limit(0).as({reaction})'.format(reaction=reaction))

        query = ','.join(query_elements)
        url = '/{postid}?fields={fields}'.format(postid=postid, fields=query)
        response = self.graph.get(url)
        response.pop(FacebookParser.CONST_ID, None)
        for reaction in FacebookParser.CONST_REACTIONS_TYPES:
            response[reaction] = response[reaction]["summary"]["total_count"]
        return response

    def iterate_all_user_posts_for_page(self, pagename: str, since: int, until: int, skip_posts_with_image: bool) -> list:
        """
        Iterates over all posts of a site with given <pagename> that were posted between <since> and <until> and that has not been created 
        by page-owner
        
        :param pagename: The name of the site that shall be crawled
        :param since: The starting date of the interval in which the posts are crawled
        :param until: The end date of the interval in which the posts are crawled
        :param skip_posts_with_image: Boolean indicating whether posts with images shall be skipped
        :return: A single post with reactions (dictionary) per iteration
        """
        post = None
        try:
            supermarket_id = self.__get_page_id(pagename)
            for post in self.__iterate_all_posts_from_page(pageid=supermarket_id, since=since, until=until):
                if post[FacebookParser.CONST_FROM]['id'] == supermarket_id:
                    continue
                if skip_posts_with_image and FacebookParser.CONST_ATTACHED_OBJ_ID in post:
                    continue
                reactions = self.__get_reactions_for_post(post[FacebookParser.CONST_ID])
                post[FacebookParser.CONST_REACTIONS] = reactions
                yield post

        except Exception as e:
            import sys

            if post is not None:
                detail = post[FacebookParser.CONST_ID]
            else:
                detail = "'{pagename}' before receiving the reactions (so actually when receiving the posts)".format(
                    pagename=pagename)

            raise type(e)(str(e) +
                          '. Exception thrown here: %s' % detail).with_traceback(sys.exc_info()[2])
