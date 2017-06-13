import datetime
import logging
import time
import traceback

# configure logging
from importer.data_retrieval.facebook.facebook_parser import FacebookParser
from importer.database.data_types import Post, Comment
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage

FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

ONE_DAY = 60 * 60 * 24
AUTH_TOKEN = '1279401372135473|e1948b86d049d7d171db22b5dc0eb9a7'
SUPERMARKET = ['AldiUK', 'walmart', 'tesco', 'sainsburys']
OFF_TOPIC_SHOPS = ['target', 'Walgreens', 'Amazon', 'bestbuy', 'Safeway', 'Macys', 'publix', 'Costco']


def __store_post(db_storage: DataStorage, post: dict, off_topic: bool):
    db_storage.insert_post(Post.create_from_single_values(post[FacebookParser.CONST_ID],
                                                          post[FacebookParser.CONST_FROM]['id'],
                                                          post[FacebookParser.CONST_MESSAGE],
                                                          post[FacebookParser.CONST_DATE],
                                                          post[FacebookParser.CONST_PERMALINK],
                                                          post[FacebookParser.CONST_REACTIONS],
                                                          off_topic))

    for comment in post[FacebookParser.CONST_COMMENTS]:
        db_storage.insert_comment(Comment.create_from_single_values(comment[FacebookParser.CONST_ID],
                                                                    post[FacebookParser.CONST_ID],
                                                                    comment[FacebookParser.CONST_FROM]['id'],
                                                                    comment[FacebookParser.CONST_MESSAGE],
                                                                    comment[FacebookParser.CONST_DATE]))


def main():
    logger = logging.getLogger('Main')

    time_step = 0.5 * ONE_DAY
    days = 365 / 2
    total_interval_size = ONE_DAY * days

    # set the authentication token and create a new FacebookParser object
    parser = FacebookParser(auth_token=AUTH_TOKEN)

    # setup db connection
    db = MongodbStorage()

    # set the limit (maximum allowed value in API: 100, maximum value because of timeout: 38 (at least when I tested))
    timestamp = time.time()

    # iterate over all supermarkets
    for shop in SUPERMARKET + OFF_TOPIC_SHOPS:
        # get posts of the last half year
        since = int(timestamp - total_interval_size)
        until = int(since + time_step)
        batch_counter = 0
        processed_posts_counter = 0

        logger.info("Start processing posts of '%s'", shop)

        # iterate as long as the until flag is before the current date (So basically until we processed all the posts until now)
        while until < timestamp:

            logger.info("Processing posts of %s from %s until %s", shop, datetime.datetime.fromtimestamp(since),
                        datetime.datetime.fromtimestamp(until))

            # iterate over the posts of the given time interval and retrieve external content
            # noinspection PyBroadException
            try:
                for post in parser.iterate_all_user_posts_for_page(pagename=shop, since=since, until=until,
                                                                   skip_posts_with_image=True):
                    # ignore post if already stored
                    if db.select_single_post({Post.COLL_POST_ID: post[FacebookParser.CONST_ID]}):
                        continue

                    __store_post(db, post, shop in OFF_TOPIC_SHOPS)
                    processed_posts_counter += 1

            except Exception:
                traceback.print_exc()

            # Set the new time interval and increase processing counter
            since += time_step
            until += time_step
            batch_counter += time_step / ONE_DAY

            logger.info("Processed %.2f%% of %s.", batch_counter / days * 100, shop)
            logger.info("Found %s total posts for this shop yet", processed_posts_counter)
        logger.info("Finished processing posts of '%s'. Found %s posts whose content could be parsed", shop,
                    processed_posts_counter)


if __name__ == '__main__':
    main()
