# sentiment_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (run once)
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data downloaded successfully")
except:
    print("NLTK data download failed - using fallback stopwords")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                 "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
                 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
                 "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
                 "weren't", 'won', "won't", 'wouldn', "wouldn't"}

def clean_tweet(tweet):
    """Clean and preprocess tweet text"""
    if not isinstance(tweet, str):
        return ""
    
    # Remove URLs, user mentions, and hashtags
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    
    # Convert to lowercase and remove extra spaces
    tweet = tweet.lower().strip()
    
    # Tokenize and remove stopwords
    words = tweet.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def get_sentiment_vader(text):
    """Get sentiment using VADER (specifically for social media)"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def get_sentiment_textblob(text):
    """Get sentiment using TextBlob"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def categorize_sentiment(score):
    """Categorize sentiment score into positive, negative, or neutral"""
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def analyze_sentiment(df):
    """Perform sentiment analysis on tweets"""
    # Calculate sentiment scores using both methods
    df['vader_score'] = df['cleaned_text'].apply(get_sentiment_vader)
    df['textblob_score'] = df['cleaned_text'].apply(get_sentiment_textblob)
    
    # Use VADER for categorization (better for social media)
    df['sentiment'] = df['vader_score'].apply(categorize_sentiment)
    
    return df

def generate_visualizations(df, topic):
    """Generate all visualizations for the analysis"""
    
    # Set the style for plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    
    # 1. Sentiment distribution pie chart
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # red, blue, green
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Distribution')
    
    # 2. Time series of sentiment
    plt.subplot(2, 2, 2)
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    df_temp.set_index('date', inplace=True)
    daily_sentiment = df_temp['vader_score'].resample('D').mean()
    daily_sentiment.plot()
    plt.title('Daily Average Sentiment')
    plt.ylabel('Sentiment Score')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # 3. Sentiment by engagement (likes + retweets)
    plt.subplot(2, 2, 3)
    df['engagement'] = df['likes'] + df['retweets']
    engagement_by_sentiment = df.groupby('sentiment')['engagement'].mean()
    engagement_by_sentiment.plot(kind='bar', color=colors)
    plt.title('Average Engagement by Sentiment')
    plt.ylabel('Engagement (Likes + Retweets)')
    
    # 4. Word clouds for positive and negative tweets
    plt.subplot(2, 2, 4)
    positive_text = " ".join(tweet for tweet in df[df['sentiment']=='positive']['cleaned_text'])
    
    if positive_text.strip():
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(positive_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Words')
    else:
        plt.text(0.5, 0.5, 'No positive text data', ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    filename = f'sentiment_analysis_{topic.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate word cloud for negative sentiment if exists
    negative_text = " ".join(tweet for tweet in df[df['sentiment']=='negative']['cleaned_text'])
    if negative_text.strip():
        plt.figure(figsize=(8, 6))
        wordcloud = WordCloud(width=800, height=400, background_color='black', 
                             colormap='Reds').generate(negative_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Words')
        neg_filename = f'negative_words_{topic.replace(" ", "_")}.png'
        plt.savefig(neg_filename, dpi=300, bbox_inches='tight')
        plt.show()

def use_sample_data(topic="sample"):
    """Use sample data for demonstration"""
    print(f"Creating sample data for topic: {topic}")
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2023-10-01', periods=100, freq='D')
    sample_tweets = [
        f"Loving the new {topic} product! It's amazing!",
        f"Having issues with {topic} customer service. Very disappointed.",
        f"{topic} is okay, nothing special really.",
        f"The {topic} event was a huge success!",
        f"Why is {topic}'s website always down? #frustrated",
        f"Just bought {topic} and I'm really impressed with the quality!",
        f"{topic} needs to improve their delivery times. It's taking forever!",
        f"Great experience with {topic} support team. Very helpful!",
        f"Not sure how I feel about {topic}'s new update. It's confusing.",
        f"{topic} has the worst user interface I've ever seen.",
        f"I recommend {topic} to all my friends. Best product ever!",
        f"{topic} customer service is terrible. Avoid at all costs.",
        f"Pretty happy with my {topic} purchase. Good value for money.",
        f"{topic} keeps crashing. Need to fix these bugs ASAP.",
        f"Amazing features in the new {topic} update. Well done!",
        f"Disappointed with {topic}'s quality. Expected better.",
        f"{topic} is decent but could be improved.",
        f"Best decision I made was buying {topic}. Love it!",
        f"{topic} stopped working after 2 days. Poor quality.",
        f"Very satisfied with {topic} performance. Exceeded expectations."
    ]
    
    # Create DataFrame with sample data
    df = pd.DataFrame({
        'date': np.random.choice(dates, 100),
        'text': np.random.choice(sample_tweets, 100),
        'user': [f'user_{i}' for i in range(100)],
        'location': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney', 'Unknown'], 100),
        'retweets': np.random.randint(0, 50, 100),
        'likes': np.random.randint(0, 100, 100)
    })
    
    # Clean text and analyze sentiment
    print("Cleaning and analyzing sample data...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    df = analyze_sentiment(df)
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, topic)
    
    # Save results
    filename = f'sentiment_analysis_{topic.replace(" ", "_")}.csv'
    df.to_csv(filename, index=False)
    print(f"Sample analysis complete. Results saved to {filename}")
    
    # Print summary
    print("\n=== SAMPLE DATA SUMMARY ===")
    total_tweets = len(df)
    positive_count = len(df[df['sentiment'] == 'positive'])
    neutral_count = len(df[df['sentiment'] == 'neutral'])
    negative_count = len(df[df['sentiment'] == 'negative'])
    
    print(f"Total tweets analyzed: {total_tweets}")
    print(f"Positive: {positive_count} ({positive_count/total_tweets*100:.1f}%)")
    print(f"Neutral: {neutral_count} ({neutral_count/total_tweets*100:.1f}%)")
    print(f"Negative: {negative_count} ({negative_count/total_tweets*100:.1f}%)")
    print(f"Average sentiment score: {df['vader_score'].mean():.3f}")
    
    # Show sample of the data
    print("\n=== SAMPLE TWEETS ===")
    for i, (text, sentiment) in enumerate(zip(df['text'].head(5), df['sentiment'].head(5))):
        print(f"{i+1}. [{sentiment.upper()}] {text}")
    
    return df

def main():
    """Main function to run the sentiment analysis"""
    print("=== SOCIAL MEDIA SENTIMENT ANALYSIS ===")
    print("This tool analyzes public opinion on topics or brands")
    print("=" * 50)
    
    # Get user input for analysis
    topic = input("Enter a topic or brand to analyze: ").strip()
    if not topic:
        topic = "iPhone"
    
    # Use sample data directly (bypassing Twitter API issues)
    print(f"\nAnalyzing sentiment for: {topic}")
    df = use_sample_data(topic)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the generated files:")
    print(f"- sentiment_analysis_{topic.replace(' ', '_')}.png (visualizations)")
    print(f"- sentiment_analysis_{topic.replace(' ', '_')}.csv (raw data)")
    
    # Offer to show more details
    show_details = input("\nWould you like to see detailed results? (y/n): ").lower()
    if show_details == 'y':
        print(f"\nDetailed sentiment breakdown for '{topic}':")
        sentiment_breakdown = df['sentiment'].value_counts()
        for sentiment, count in sentiment_breakdown.items():
            percentage = count / len(df) * 100
            print(f"{sentiment.upper()}: {count} tweets ({percentage:.1f}%)")
        
        # Show most positive and negative tweets
        print(f"\nMost positive tweet:")
        most_positive = df.loc[df['vader_score'].idxmax()]
        print(f"Score: {most_positive['vader_score']:.3f} - {most_positive['text']}")
        
        print(f"\nMost negative tweet:")
        most_negative = df.loc[df['vader_score'].idxmin()]
        print(f"Score: {most_negative['vader_score']:.3f} - {most_negative['text']}")

if __name__ == "__main__":
    main()