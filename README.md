# HitWicket
//HitWicket Project


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google_play_scraper import reviews_all, Sort
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 1: Fetch Reviews
def fetch_reviews(app_id, lang='en', country='in', num_reviews=100000):
    all_reviews = []
    count = 0
    while count < num_reviews:
        reviews_data = reviews_all(
            app_id,
            sleep_milliseconds=0,
            lang=lang,
            country=country,
            sort=Sort.NEWEST  # Get the newest reviews
        )
        
        if not reviews_data:
            break
        
        all_reviews.extend(reviews_data)
        count += len(reviews_data)
        print(f"Fetched {count} reviews so far...")
        
        if len(reviews_data) < 2000:
            break
    
    return all_reviews

# Step 2: Filter Reviews by Keywords
def filter_reviews_by_keywords(reviews_df, keywords):
    keyword_filter = reviews_df['content'].apply(lambda x: any(keyword.lower() in x.lower() for keyword in keywords))
    filtered_reviews = reviews_df[keyword_filter]
    return filtered_reviews

# Step 3: Sentiment Analysis Function
def perform_sentiment_analysis(reviews_df):
    # Prepare data for training (using sentiment score as a label for now)
    reviews_df['label'] = reviews_df['score'].apply(lambda x: 1 if x > 3 else 0)  # Simplistic positive (1) / negative (0)
    
    # Train a simple model using the 'content' of reviews
    X_train, X_test, y_train, y_test = train_test_split(reviews_df['content'], reviews_df['label'], test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    predictions = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    # Predict sentiment on full dataset
    reviews_df['predicted_sentiment'] = model.predict(vectorizer.transform(reviews_df['content']))
    
    return reviews_df, model, vectorizer

# Step 4: Extract Best, Worst, and Average Sentiments
def extract_sentiment_extremes(reviews_df):
    positive_reviews = reviews_df[reviews_df['predicted_sentiment'] == 1]
    negative_reviews = reviews_df[reviews_df['predicted_sentiment'] == 0]
    
    best_review = positive_reviews.loc[positive_reviews['score'].idxmax()]['content']
    worst_review = negative_reviews.loc[negative_reviews['score'].idxmin()]['content']
    
    # Handle case where the mean score might not have an exact match
    mean_score = reviews_df['score'].mean()
    average_review = reviews_df.loc[reviews_df['score'].sub(mean_score).abs().idxmin()]['content']
    
    return best_review, worst_review, average_review

# Step 5: Generate Essence Review
def generate_essence_review(filtered_reviews_df):
    positive_ratio = filtered_reviews_df['predicted_sentiment'].mean()
    essence = f"The general sentiment of the reviews is {'positive' if positive_ratio > 0.5 else 'negative'}, with {positive_ratio*100:.2f}% positive reviews."
    return essence

# Function to plot sentiment distribution dynamically using Plotly with descriptive labels
def plot_sentiment_distribution_dynamic(reviews_df, positive_label="What People Like the Most", negative_label="What People Dislike"):
    sentiment_counts = reviews_df['predicted_sentiment'].value_counts()
    fig = px.bar(
        x=[positive_label, negative_label], 
        y=[sentiment_counts.get(1, 0), sentiment_counts.get(0, 0)],
        color=[positive_label, negative_label],
        color_discrete_map={positive_label: 'green', negative_label: 'red'},
        labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
        title='Sentiment Distribution'
    )
    fig.update_layout(bargap=0.2)
    fig.show()

# Function to visualize best, worst, and average review sentiment scores
def plot_best_worst_average_reviews(best_review, worst_review, average_review):
    reviews = [best_review, worst_review, average_review]
    labels = ['Best Review (What People Love)', 'Worst Review (What People Dislike)', 'Average Review']
    sentiment_scores = [1, 0, 0.5]  # Assuming best is positive, worst is negative, and average is neutral
    
    fig = go.Figure(data=[
        go.Bar(name='Review Sentiment', x=labels, y=sentiment_scores, text=reviews, textposition='auto')
    ])
    
    fig.update_layout(title='Sentiment Analysis of Best, Worst, and Average Reviews',
                      xaxis_title='Review Type',
                      yaxis_title='Sentiment Score',
                      yaxis=dict(tickvals=[0, 0.5, 1], ticktext=['Negative', 'Neutral', 'Positive']))
    fig.show()

# Function to plot the presence of each keyword in the reviews
def plot_keyword_presence(reviews_df, keywords):
    keyword_counts = {keyword: reviews_df['content'].str.contains(keyword, case=False).sum() for keyword in keywords}
    
    fig = plt.figure(figsize=(10, 6))
    plt.bar(keyword_counts.keys(), keyword_counts.values(), color='skyblue')
    plt.title('Keyword Presence in Reviews')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Function to visualize sentiment score distribution
def plot_sentiment_score_distribution(reviews_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(reviews_df['score'], kde=True, bins=5, color='purple')
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Fetch reviews for the app
    app_id = 'cricketgames.hitwicket.strategy'
    reviews_data = fetch_reviews(app_id, num_reviews=100000)
    
    # Convert to DataFrame
    df_reviews = pd.DataFrame(reviews_data)
    
    # Ensure necessary columns are present
    if 'content' not in df_reviews.columns or 'score' not in df_reviews.columns:
        raise ValueError("DataFrame must contain 'content' and 'score' columns.")
    
    # Keywords to filter reviews
    keywords = ('multiplayer','global alliance','offline mode','leagues','rewards')
    
    # Filter reviews by keywords
    filtered_reviews = filter_reviews_by_keywords(df_reviews, keywords)
    print(f"Filtered {len(filtered_reviews)} reviews with specified keywords.")
    
    # Perform sentiment analysis on filtered reviews
    filtered_reviews, model, vectorizer = perform_sentiment_analysis(filtered_reviews)
    
    # Extract best, worst, and average reviews
    best_review, worst_review, average_review = extract_sentiment_extremes(filtered_reviews)
    
    print("Best Review:", best_review)
    print("Worst Review:", worst_review)
    print("Average Review:", average_review)
    
    # Generate essence review
    essence = generate_essence_review(filtered_reviews)
    print("Essence Review:", essence)
    
    # Plot sentiment distribution dynamically with descriptive labels
    plot_sentiment_distribution_dynamic(filtered_reviews, positive_label="What People Like the Most", negative_label="What People Dislike")
    
    # Plot best, worst, and average review sentiments
    plot_best_worst_average_reviews(best_review, worst_review, average_review)
    
    # Plot keyword presence
    plot_keyword_presence(filtered_reviews, keywords)
    
    # Plot sentiment score distribution
    plot_sentiment_score_distribution(filtered_reviews)
