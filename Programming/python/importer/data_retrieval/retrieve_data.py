import datetime
import logging
import time
import traceback

# configure logging
from importer.data_retrieval import FacebookParser

FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

ONE_DAY = 60 * 60 * 24
AUTH_TOKEN = '1279401372135473|e1948b86d049d7d171db22b5dc0eb9a7'
SUPERMARKET = ['walmart']


def main():
    logger = logging.getLogger('Main')

    time_step = 0.5 * ONE_DAY
    days = 365 / 2
    total_interval_size = ONE_DAY * days

    # set the authentication token and create a new FacebookParser object
    parser = FacebookParser(auth_token=AUTH_TOKEN)

    # set the limit (maximum allowed value in API: 100, maximum value because of timeout: 38 (at least when I tested))
    timestamp = time.time()

    # iterate over all newspapers
    for supermarket in SUPERMARKET:
        # get posts of the last half year
        since = int(timestamp - total_interval_size)
        until = int(since + time_step)
        batch_counter = 0
        processed_posts_counter = 0

        logger.info("Start processing posts of '%s'", supermarket)

        # iterate as long as the until flag is before the current date (So basically until we processed all the posts until now)
        while until < timestamp:

            logger.info("Processing posts of %s from %s until %s", supermarket, datetime.datetime.fromtimestamp(since),
                        datetime.datetime.fromtimestamp(until))

            # iterate over the posts of the given time interval and retrieve external content
            # noinspection PyBroadException
            try:
                for post in parser.iterate_all_posts_for_page(pagename=supermarket, since=since, until=until):
                    # ignore post if already processed
                    a = post["id"]

            except Exception:
                traceback.print_exc()

            # Set the new time interval and increase processing counter
            since += time_step
            until += time_step
            batch_counter += time_step / ONE_DAY

            logger.info("Processed %.2f%% of %s.", batch_counter / days * 100, supermarket)
            logger.info("Found %s total posts for this supermarket yet", processed_posts_counter)
        logger.info("Finished processing posts of '%s'. Found %s posts whose content could be parsed", supermarket,
                    processed_posts_counter)


if __name__ == '__main__':
    main()
