import traceback

from flask import Flask, jsonify, request, make_response, abort

from data_processing.emotion_mining import EmotionMiner
from data_processing.sentiment_miner import Sentimenter
from importer.database.data_types import Emotion
from importer.database.mongodb import MongodbStorage

db = MongodbStorage()

emotion = EmotionMiner(db)
sentiment = Sentimenter(db)

app = Flask(__name__)


@app.route('/predict/single', methods=['POST'])
def predict_single_post():
    try:
        if not request.json:
            abort(400, 'Mandatory data missing: json-body')
        if 'post' not in request.json or len(request.json["post"]) == 0:
            abort(400, 'Missing mandatory parameter "post"')

        post = request.json['post']

        if not isinstance(post, str):
            abort(400, 'The parameter "post" needs to be of type str')

        # Emotions
        emotion_list = emotion.get_post_emotion_value(post)
        emotions = {}
        for index, emotionname in enumerate(Emotion.EMOTION_TYPES):
            emotions[emotionname] = emotion_list[index]

        # Sentiment
        sentiments = sentiment.get_post_sentiment_value(post)

        # Reactions
        # TODO: Add the reaction_prediction

        # Create response
        response = dict(reactions={},
                        emotions=emotions,
                        sentiment=sentiments)

        return jsonify(response)
    except Exception as e:
        print("Error processing request: " + str(request.json))
        traceback.print_exc()
        abort(500, str(e))

@app.route('/predict/batch', methods=['POST'])
def predict_batch_post():
    try:
        if not request.json:
            abort(400, 'Mandatory data missing: json-body')
        if 'posts' not in request.json:
            abort(400, 'Missing mandatory parameter "posts"')

        posts = request.json['posts']

        if not isinstance(posts, list):
            abort(400, 'The parameter "posts" needs to be of type list')

        response_list = []
        for post in posts:
            if not isinstance(post, str):
                abort(400, 'The parameter "posts" may only contain values of type str')

            # Emotions
            emotion_list = emotion.get_post_emotion_value(post)
            emotions = {}
            for index, emotionname in enumerate(Emotion.EMOTION_TYPES):
                emotions[emotionname] = emotion_list[index]

            # Sentiment
            sentiments = sentiment.get_post_sentiment_value(post)

            # Reactions
            # TODO: Add the reaction_prediction

            # Create response
            response = dict(reactions={},
                            emotions=emotions,
                            sentiment=sentiments)
            response_list.append(response)

        return jsonify(response_list)
    except Exception as e:
        print("Error processing request: " + str(request.json))
        traceback.print_exc()
        abort(500, str(e))



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'message': error.description}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'message': error.description}), 400)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify({'message': error.description}), 500)


if __name__ == '__main__':
    app.run(debug=True)
