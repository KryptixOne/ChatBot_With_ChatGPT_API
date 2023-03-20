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

def merge_comments_posts(post_df,comments_df):
    comments_posts_df = post_df.merge(comments_df, on='post_id', how ='left')
    comments_posts_df = comments_posts_df[~comments_posts_df['comment'].isnull()]
    return comments_posts_df

if __name__ == "__main__":
    path_posts = r'D:/ChatGPTAPI_Project/DS_ML_AI_posts.csv'
    path_comments = r'D:/ChatGPTAPI_Project/DS_ML_AI_comments.csv'

    posts_df, comments_df = read_in_data(path_posts, path_comments)
    posts_df = convert_utc_to_DateYear(posts_df)
    comments_posts_df = merge_comments_posts(posts_df,comments_df)

    pass