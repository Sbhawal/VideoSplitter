import os, random
import requests
import praw
from urllib.parse import urlparse
from tqdm import tqdm
import re
import asyncio

# ---- CONFIGURATION ----

REDDIT_CLIENT_ID = 'ImxGilc6S3KjTvAvgnp3Gg'
REDDIT_CLIENT_SECRET = 'sMmVLnu-qJAJ-6hA8OcvmSycqUY7hQ'
REDDIT_USER_AGENT = 'media scraper by u/annonymous1216'

# ---- AUTHENTICATION ----

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username='YOUR_USERNAME',         # <-- Replace with your Reddit username
    password='YOUR_PASSWORD'          # <-- Replace with your Reddit password
)

def get_followed_subreddits():
    """Returns a list of subreddit names the authenticated user is subscribed to."""
    return [sub.display_name for sub in reddit.user.subreddits(limit=None)]

if __name__ == "__main__":
    subreddits = get_followed_subreddits()
    print("Subreddits followed by the user:")
    for sub in subreddits:
        print(sub)