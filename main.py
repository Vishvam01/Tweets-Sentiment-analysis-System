import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob


# ----------------------------
# Clean Tweet
# ----------------------------
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"[^A-Za-z ]+", "", tweet)
    return tweet.lower()


# ----------------------------
# Get Sentiment
# ----------------------------
def get_sentiment(text):
    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# ----------------------------
# Main Program
# ----------------------------
print("=== Twitter Sentiment Analysis (Offline Dataset) ===")


# Load dataset
print("Loading dataset...")

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

# Column names
df.columns = [
    "Target",
    "ID",
    "Date",
    "Flag",
    "User",
    "Tweet"
]


# Take sample (for speed)
df = df.sample(5000)


# Clean tweets
print("Cleaning tweets...")

df["Clean_Tweet"] = df["Tweet"].apply(clean_tweet)


# Analyze sentiment
print("Analyzing sentiment...")

df["Sentiment"] = df["Clean_Tweet"].apply(get_sentiment)


# Convert to labels
def label_sentiment(polarity):
    if polarity == "Positive":
        return "Positive"
    elif polarity == "Negative":
        return "Negative"
    else:
        return "Neutral"


df["Sentiment"] = df["Sentiment"].apply(label_sentiment)


# Count
pos = len(df[df["Sentiment"] == "Positive"])
neg = len(df[df["Sentiment"] == "Negative"])
neu = len(df[df["Sentiment"] == "Neutral"])


print("\n===== RESULT =====")
print("Positive:", pos)
print("Negative:", neg)
print("Neutral :", neu)


# Plot
plt.figure(figsize=(7, 7))

plt.pie(
    [pos, neg, neu],
    labels=["Positive", "Negative", "Neutral"],
    autopct="%1.1f%%"
)

plt.title("Offline Twitter Sentiment Analysis")
plt.show()


# Save output
df.to_csv("offline_sentiment_result.csv", index=False)

print("\nData saved as offline_sentiment_result.csv")