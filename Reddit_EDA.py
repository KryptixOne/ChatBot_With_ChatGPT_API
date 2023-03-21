import pandas as pd
import datetime as dt

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline


def read_in_data(post_df_path: str,
                 comments_df_path: str):
    posts_df = pd.read_csv(post_df_path)
    comments_df = pd.read_csv(comments_df_path)

    return posts_df, comments_df


def convert_utc_to_DateYear(posts_df):
    posts_df['created_date'] = posts_df['created_utc'].apply(lambda x: dt.datetime.fromtimestamp(x))
    posts_df['created_year'] = posts_df['created_date'].dt.year
    return posts_df


def merge_comments_posts(post_df, comments_df):
    comments_posts_df = post_df.merge(comments_df, on='post_id', how='left')
    comments_posts_df = comments_posts_df[~comments_posts_df['comment'].isnull()]
    return comments_posts_df


def aggregate_text(comments_posts_df):
    """
    Aggregate Text with posts
    Post     |    Comment    |
    --------------------------
    Post1    |    Ciaooo     |
    Post1    |    Olaaaa     |
    Post2    |    Helllo     |
    --------------------------

    to

    Post     |    Comment    |     Post_Comment       |
    ---------------------------------------------------
    Post1    |Ciaooo, Olaaaa |  Post1, Ciaooo, Olaaaa |
    Post2    |    Helllo     |    Post2, Helllo       |
    ---------------------------------------------------
    """

    comments_posts_dt_tmp = comments_posts_df[['post_title', 'selftext', 'comment']].astype(str)
    agg_comments = comments_posts_dt_tmp.groupby(['post_title', 'selftext'])['comment'].apply('. '.join).reset_index()
    agg_comments['combined_text'] = agg_comments.astype(str).agg('. '.join, axis=1)
    all_text = ''.join(agg_comments['combined_text'])

    return all_text

def visualize_wordCloud(posts_df):
    """
    Build/Visualize a WordCloud which helps visualize the frequency that certain words/phrases are being said/posted.
    i.e. How much a certain word is trending
    :param posts_df: dataframe of posts
    """
    post_title_text = ''.join([title for title in posts_df['post_title'].str.lower()])
    word_cloud = WordCloud(collocation_threshold=2, width=1000, height=500, background_color='white').generate(
        post_title_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()

    return None


def get_emotion_anaylsis_model():
    """
    Emotion models return a [dict] of emotions which include sadness, joy, love, anger, fear, and surprise. Along with
    the score each emotion receives based on the text input
    """
    emotion_classifier = pipeline('text-classification', model='bhadresh-savani/distilbert-base0uncased-emotion',
                                  return_all_scores=True)
    return emotion_classifier


def get_emotion(text: str, emotion_model):
    """

    :param text: input text str
    :param emotion_model: input emotion classifier model
    :return: emotion with the highest score
    """

    pred_scores = emotion_model(text)
    emotion = max(pred_scores[0], key=lambda x: x['score'])['label']

    return emotion


def get_sentiment_anaylsis_model():
    """
    Sentiment models return whether a text input is associated with negative, positive or neutral feeling. It also
    returns the probability associated with that classification
    """
    sentiment_classifier = pipeline(model='finiteautomata/bertweet-base-sentiment-analysis')
    return sentiment_classifier


def get_text_sentiment(text, sentiment_model):
    try:
        sentiment = sentiment_model(text)[0]['label']
    except:
        """
        Due to error that can occur. Possibly due to input text length. Need to debug later
        """
        sentiment = 'Not Classified'
    return sentiment


if __name__ == "__main__":
    path_posts = r'D:/ChatGPTAPI_Project/DS_ML_AI_posts.csv'
    path_comments = r'D:/ChatGPTAPI_Project/DS_ML_AI_comments.csv'

    posts_df, comments_df = read_in_data(path_posts, path_comments)
    posts_df = convert_utc_to_DateYear(posts_df)
    comments_posts_df = merge_comments_posts(posts_df, comments_df)


    comments_posts_df_sub = comments_posts_df[comments_posts_df['post_title'].str.contains('chatgpt')]

    # get sentiment of posts that contain a specific word
    comments_posts_df_sub['sentiment'] = comments_posts_df_sub['comment'].astype(str).astype(str).apply(
        lambda x: get_text_sentiment(x))
    # get the highest ranking emotion of posts that contain a specific word
    comments_posts_df_sub['emotion'] = comments_posts_df_sub['comment'].astype(str).astype(str).apply(
        lambda x: get_emotion(x))

    all_text = aggregate_text(comments_posts_df)

    with open(r'D:\ChatGPTAPI_Project/all_text_reddit.txt','w', encoding = 'utf-8') as f:
        f.write(all_text)


