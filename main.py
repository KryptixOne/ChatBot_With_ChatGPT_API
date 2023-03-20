import praw
import pandas as pd


def get_top_posts(subreddit_list='MachineLearning', limit=1000, time_filter='all'):
    reddit = praw.Reddit(client_id='thGxXn50aXKMCqWwz3boSg',
                         client_secret='2333NbmJYn-ksIZ77baMpQ9juMWQvg',
                         redirect_uri='http://localhost:8080',
                         user_agent='Kyrptix')

    # pulls top submissions from all time in redditdev and learn python subreddits
    """
    for submission in reddit.subreddit('redditdev+learnpython').top(time_filter='all',limit = 1000):
        print([submission])
    """
    post_df = []
    posts = reddit.subreddit(subreddit_list).top(time_filter=time_filter, limit=limit)
    for post in posts:
        post_df.append({
            'post_id': post.id,
            'subreddit': post.subreddit,
            'created_utc': post.created_utc,
            'selftext': post.selftext,
            'post_url': post.url,
            'post_title': post.title,
            'link_flair_text': post.link_flair_text,
            'score': post.score,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio
        })
    return pd.DataFrame(post_df), reddit


create_csv = True
posts_df, reddit = get_top_posts(subreddit_list='MachineLearning+datascience+artificial', limit=10, time_filter='all')
if create_csv == True:
    posts_df.to_csv('DS_ML_AI_posts.csv', header=True, index=False)

comments_list = []
for post_id in posts_df['post_id']:
    submission = reddit.submission(post_id)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments_list.append({'post_id': post_id,
                              'comment': comment.body})

comments_df = pd.DataFrame(comments_list)
comments_df.to_csv('DS_ML_AI_comments.csv', header=True, index=False)

