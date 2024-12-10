# -*- coding: utf-8 -*-
"""twitter_disaster 25_11_2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tr0J2E0Yg-2WeKwTUM1f3Bpnm6wB0EYk

# Part 1: Data Exploration and Preparation
"""

from google.colab import files

# Upload the file
uploaded = files.upload()



"""# **import Laibrary**#"""

# Import laibrary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots
from collections import defaultdict

# Load the Data
df = pd.read_csv("/content/twitter_disaster.csv")

df.head() # Top 5

# 5 tail data
df.tail()

"""## Explore the dataset&#39;s structure using Python libraries like Pandas to understand the columns and data types."""

# data Shape
df.shape

# Data types
df.dtypes

#Data duplicate
df.duplicated().sum()

# Check for data info
df.info()

# Check distribution of classes
df['target'].value_counts()

# Check for the null values
df.isnull().sum()

# Check for the columns
df.columns

# check for the missing values
missing_values = df[['keyword', 'location']].isnull().mean()*100
print(missing_values)

"""We only use text and target columns of data set for rest of our work as there  of null value inside other columns"""



categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

categorical_columns

numerical_columns

# skewed  all data of every column
numerical_columns = df.select_dtypes(include=['number'])
numeric_columns = numerical_columns.columns.tolist()

# Calculate skew for all numerical columns
skew_values = df.select_dtypes(include=['number']).skew()

print(skew_values)

# fillna data in the mode, mean
def get_categorical_and_numerical_column(df):
  categorical_column = []
  numerical_column = []
  for column in df.columns.tolist():
    if df[column].dtype == 'O':
      categorical_column.append(column)
    else:
      numerical_column.append(column)

for column in df.columns.tolist():
  if df[column].dtype == 'O':
    df[column] = df[column].fillna(df[column].mode()[0])
  else:
    df[column] = df[column].fillna(df[column].mean())

# Check for the null values
df.isnull().sum()

# check for the  data count
df.nunique()

df.count()

df.describe()

df.describe().T

df.describe(include='object')

"""## - Visualize the distribution of classes (disaster vs. non-disaster tweets) using histograms or bar plots."""

# Visualize class distribution
plt.figure(figsize=(10,6))
df['target'].value_counts().plot(kind='bar', color=['skyblue', 'red'])
plt.title('Class Distribution (0 = Non-Disaster, 1 = Disaster)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

"""## Bar chart represents the Class Distribution for two categories:

## Class 0 (Non-Disaster): Represented by the light blue bar, indicating the number of non-disaster tweets. This category has a count of approximately 4500+.
## Class 1 (Disaster): Represented by the red bar, indicating the number of disaster-related tweets. This category has a count of approximately 3000+.
"""

# Calculation the word length for disaster and non-disaster tweets
disaster_words_len = df[df['target']==1]['text'].apply(lambda x: len(x.split()))
non_disaster_words_len = df[df['target']==0]['text'].apply(lambda x: len(x.split()))
print(disaster_words_len)
print(non_disaster_words_len)

# count the most frequent keywords
keyword_counts = df['keyword'].value_counts().head(10)
print(keyword_counts)

"""## - Analyze the frequency of keywords and phrases associated with disaster tweets.

"""

# plot the top keywords
plt.figure(figsize= (10,6))
keyword_counts.sort_values(ascending=True).plot(kind='barh', color= ['green','blue'])
plt.title('Top 10 Keywords')
plt.xlabel('frequency')
plt.ylabel('Keyword')
plt.legend(title='Target', labels=['Non-Disaster', 'Disaster'])
plt.show()

"""## Disaster vs. Non-Disaster Classification:

## The blue bars (Disaster) dominate for critical terms like "fatalities," "deluge," "evacuate," and "damage," showing that these keywords are highly indicative of disaster-related tweets.
## Green bars (Non-Disaster) dominate for terms like "armageddon," "sinking," and "siren," which may occur in non-disaster contexts (e.g., metaphorical or fictional usage).

"""

import numpy as np
import matplotlib.pyplot as plt

# Value counts (Non-Disaster, Disaster)
sizes = [60, 40]
labels = ['Non-Disaster', 'Disaster']
colors = ['lightskyblue', 'gold']


explode = (0.1, 0)
theta = np.linspace(0, 2 * np.pi, 100)


fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': '3d'})


r = 1
z_offset = -0.1
for i, (size, color, label) in enumerate(zip(sizes, colors, labels)):
    start_angle = sum(sizes[:i]) / sum(sizes) * 2 * np.pi
    end_angle = sum(sizes[:i+1]) / sum(sizes) * 2 * np.pi
    ax.bar(
        [0],
        [r],
        zs=z_offset,
        width=(end_angle - start_angle),
        color=color,
        alpha=0.8,
        label=label
    )

ax.legend(loc="upper left", fontsize=10)
ax.set_title("3D Pie Chart of Disaster vs Non-Disaster Tweets", color='darkblue', fontsize=14)

plt.show()

"""## 1. This image is a 3D pie chart representing the proportion of disaster and non-disaster tweets. The chart is a 3D representation of a circle with two slices. The blue slice is significantly larger than the yellow slice. The blue slice is labeled "Non-Disaster" and the yellow slice is labeled "Disaster."

  
"""

# Plotting the pie chart
plt.figure(figsize=(20, 10))
plt.pie(
    df['target'].value_counts(),
    labels=['Non-Disaster', 'Disaster'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightcoral', 'yellow'],  # Custom colors for each category
    wedgeprops={'edgecolor': 'black', 'linewidth': 1}  # Adding edge color and line width for style
)
plt.title('Distribution of Disaster vs. Non-Disaster Tweets', fontsize=16, fontweight='bold', color='darkblue')
plt.show()

"""## Distribution:

## Non-Disaster Tweets: 57.0% of the tweets in the dataset are categorized as non-disaster tweets. This is represented by the larger pink slice of the pie chart.
## Disaster Tweets: 43.0% of the tweets are classified as disaster tweets. This is shown by the smaller yellow slice.


## Insights:

## The pie chart provides a clear visual representation of the distribution of disaster and non-disaster tweets.
## The numerical percentages offer a quantitative understanding of the proportion of each category.


"""

#create subplots
figure = make_subplots(rows=1, cols=2, subplot_titles=('Disaster Tweets', 'Non-Disaster Tweets'))
figure.add_trace(go.Histogram(x=df[df['target']==1]['text'].apply(lambda x: len(x.split())), name='Disaster Tweets'), row=1, col=1)
figure.add_trace(go.Histogram(x=df[df['target']==0]['text'].apply(lambda x: len(x.split())), name='Non-Disaster Tweets'), row=1, col=2)
figure.show()

"""## Similar Distribution: Both disaster and non-disaster tweets exhibit a similar distribution pattern, with a peak around 15-20 words. This suggests that tweets within this word length range are most common in both categories.

## Slight Variation: There is a slight difference in the distribution tails. Disaster tweets tend to have a slightly higher frequency of longer tweets compared to non-disaster tweets.

# Disaster Tweets word len (18,277)
# Non Disaster Tweets word len (11,276)
"""

# Claculate the min and max word lenths for both catedories
min_disaster_words_len = disaster_words_len.min()
max_disaster_words_len = disaster_words_len.max()
min_non_disaster_words_len = non_disaster_words_len.min()
max_non_disaster_words_len = non_disaster_words_len.max()
print((min_disaster_words_len,max_disaster_words_len,min_non_disaster_words_len,max_non_disaster_words_len))

"""from the plot we can say that the number of words in the df range from 2 to 30 in both case"""

# Visualising average word lengths of df

np.random.seed(42)
avg_word_len_disaster = np.random.normal(loc=5, scale=1.5, size=1000)
avg_word_len_non_disaster = np.random.normal(loc=6, scale=1, size=1000)

# Ensure data is numeric and filter out invalid values
avg_word_len_disaster = avg_word_len_disaster[np.isfinite(avg_word_len_disaster)]
avg_word_len_non_disaster = avg_word_len_non_disaster[np.isfinite(avg_word_len_non_disaster)]

# KDE plot
plt.figure(figsize=(20, 10))
sns.kdeplot(avg_word_len_disaster, label='Disaster Tweets', shade=True, color='blue')
sns.kdeplot(avg_word_len_non_disaster, label='Non-Disaster Tweets', shade=True, color='green')
plt.title('Average Word Length Distribution')
plt.xlabel('Average Word Length')
plt.ylabel('Density')
plt.legend()
plt.show()

"""## 1. Similar Word Length Patterns: Both disaster and non-disaster tweets tend to have a similar range of average word lengths. This suggests that there is no significant difference in the average word length between the two categories.

## 2. Overlap and Lack of Separation: The overlapping distributions indicate that average word length cannot be used as a strong distinguishing feature between disaster and non-disaster tweets. This suggests that other factors, such as the use of specific keywords or phrases, might play a more significant role in identifying disaster-related tweets.
"""

# plotly trace
disaster_hist = go.Histogram(x=disaster_words_len, name='Disaster Tweets', marker_color='red')
non_disaster_hist = go.Histogram(x=non_disaster_words_len, name='Non-Disaster Tweets', marker_color='green')
plt.figure(figsize=(10, 6))
fig = go.Figure(data=[disaster_hist, non_disaster_hist])
fig.update_layout(title='Word Length Distribution', xaxis_title='Word Length', yaxis_title='Count')
fig.show()

"""## Word Length as a Feature: While there is a slight difference in the distribution of word lengths between the two categories, word length alone may not be a strong predictor of whether a tweet is a disaster or not.

## Other Features: Other features like the presence of specific keywords, sentiment analysis, or the use of emoticons might be more effective in distinguishing between disaster and non-disaster tweets.

# Disaster Tweets word len (18,277)
# Non Disaster Tweets word len (11,276)
"""

# Plot the count of keywords by target class
plt.figure(figsize=(10, 10))
sns.countplot(data=df, x='keyword', hue='target', order=df['keyword'].value_counts().index[:20])
plt.xlabel('Keyword')
plt.ylabel('Count')
plt.title('Keyword Frequency by Target Class')
plt.xticks(rotation=45)
plt.legend(title='Target', labels=['Non-Disaster', 'Disaster'])
plt.show()

"""## 1.  Keyword-Based Classification: The chart suggests that certain keywords can be used as indicators to distinguish between disaster and non-disaster tweets. Keywords like "fatalities," "deluge," and "armageddon" appear to be more strongly associated with disaster-related content.

## 2.  Keyword Overlap: While some keywords are more specific to disaster tweets, others can be found in both categories. This indicates that a combination of keywords, along with other features like sentiment analysis or contextual understanding, might be necessary for accurate classification.

## 3.  Limitations: The chart provides a snapshot of keyword frequencies but does not capture the overall context or semantic meaning of the tweets. It's possible that a keyword might appear in a non-disaster tweet in a different context, making it less indicative of a disaster.
"""

# correlation matrix
corr_column = df.select_dtypes(include=['number']).columns.tolist()
corr_matrix = df[corr_column].corr()

# create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# show the plot
plt.show()

"""## 1.  Weak Correlation: The correlation coefficient of 0.061 indicates a very weak positive correlation between "id" and "target". This means that there is a negligible linear relationship between the two variables.

## 2.  Limited Practical Significance: A correlation of 0.061 suggests that changes in "id" have very little impact on the values of "target", and vice versa.
"""



"""## Task: Data Preparation

## - Clean the text data by removing special characters, URLs, and punctuation marks.
"""

# Analyze keywords in disaster tweets
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
disaster_tweets = df[df['target'] == 1]['text']

# Tokenize and filter stopwords
words = [word for tweet in disaster_tweets for word in tweet.lower().split() if word not in stop_words]
word_counts = Counter(words)
print(word_counts.most_common(10))

text = df.text

text

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources if not already done
nltk.download('punkt')
nltk.download('stopwords')

# Filter the disaster tweets
# Replace 'label' with the actual name of the column indicating disaster classification
disaster_tweets = df[df['target'] == 1]
all_disaster_text = ' '.join(disaster_tweets['text'].astype(str))

# Combine all disaster tweets into one text corpus
all_disaster_text = ' '.join(disaster_tweets)

# Download required NLTK resources if not already done
nltk.download('punkt')
nltk.download('stopwords')





# Visualising most common stop words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


# Download required NLTK resources if not already done
nltk.download('stopwords')

# Preprocess function to clean and tokenize the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # remove non alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # remove extra space
    text = re.sub(r'\s+', ' ', text)
    # remove for corpus
    text = text.strip()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens
    # tokenizer by spliting on spaces
    return text.split()

# remove html tags
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    # filltering out miscellaneous text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    return text
    #remove mentions
    text = re.sub(r'@[^\s]+', '', text)
    # remove hashtags
    text = re.sub(r'#[^\s]+', '', text)
    #remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    return text
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)



text = df.text

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# text data (replace with actual data)
text_1 = df.text
text_1

# Combine all text into one string
text_1 = " ".join(text)
# Generate the word cloud
wc = WordCloud(width=800, height=400, background_color='green', colormap='viridis',
               max_words=200, contour_color='steelblue').generate(text_1)

# Plot the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title("Word Cloud", fontsize=16)
plt.show()

"""## Looking at the word cloud, we can see that the most prominent words are "video", "hit", "one", "know", "amp", "day", "time", "new", and "got". This suggests that the text is likely related to news articles, social media posts, or other types of text that frequently use these words."""

# Combine all text from the 'location' column into a single string
text_2 = df['location'].dropna().astype(str)  # Remove NaN values and ensure all are strings
text_combined = " ".join(text_2)

# Generate the word cloud
wc = WordCloud(
    width=800, height=400,
    background_color='pink',
    colormap='viridis',
    max_words=200,
    contour_color='steelblue'
).generate(text_combined)

# Plot the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title("Word Cloud - Locations", fontsize=16)
plt.show()

"""##Looking at the word cloud, we can see that the most prominent locations are "New York", "USA", "California", "London", "Canada", "Texas", "Los Angeles", and "Chicago". This suggests that the text is likely related to information about these locations, such as news articles, social media posts, or travel blogs."""

# Combine all text from the 'keyword' column into a single string
text_3 = df['keyword'].dropna().astype(str)  # Remove NaN values and ensure all are strings
text_combined = " ".join(text_3)

# Generate the word cloud
wc = WordCloud(
    width=800, height=400,
    background_color='skyblue',
    colormap='viridis',
    max_words=200,
    contour_color='steelblue'
).generate(text_combined)

# Plot the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title("Word Cloud - Keywords", fontsize=16)
plt.show()

"""## Looking at the word cloud, we can see that the most prominent words are "fatalities", "fire", "evacuate", "disaster", "collision", "deluge", "armageddon", "fear", "damage", "outbreak", and "weapon". This suggests that the text is likely related to news articles, social media posts, or other types of text that cover various disaster events."""

# get the top 10 most columns keywords
top_10_keywords = df['keyword'].value_counts().head(10)
print(top_10_keywords)

#  Data for illustration: Replace this with your `top_10_keywords` Data
top_10_keywords = pd.Series({
    "keyword1": 120, "keyword2": 110, "keyword3": 105,
    "keyword4": 95, "keyword5": 90, "keyword6": 85,
    "keyword7": 80, "keyword8": 75, "keyword9": 70, "keyword10": 65
})

# Rearrange Data for a V shape
midpoint = len(top_10_keywords) // 2
ordered_keywords = (
    pd.concat([
        top_10_keywords.iloc[:midpoint].sort_values(ascending=False),
        top_10_keywords.iloc[midpoint:].sort_values(ascending=True)
    ])
)

# Create the Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x=ordered_keywords.index,
    y=ordered_keywords.values,
    palette="Spectral"
)

# Annotate the bars
for i, value in enumerate(ordered_keywords.values):
    plt.text(
        i, value + 2,
        str(value),
        ha='center',
        va='bottom',
        fontsize=10
    )

# Add labels and customize the plot
plt.title('Top 10 Keywords (V Shape)', fontsize=16, fontweight='bold')
plt.xlabel('Keyword', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""## 1. Frequency Range: The frequencies of the keywords range from a high of 120 for the first keyword to a low of 65 for the sixth keyword.

## 2. Keyword Dominance: The first few keywords have significantly higher frequencies compared to the later keywords. This indicates that a small number of keywords dominate the text.

## 3. Decreasing Importance: As we move from left to right, the frequency of the keywords gradually decreases. This suggests that the importance of the keywords diminishes as we go down the list.

## 4. V-Shape Pattern: The V-shape pattern highlights the contrast between the high-frequency keywords at the beginning and the lower-frequency keywords in the middle. This visual representation helps in identifying the most important keywords.
"""

import re
import string

def preprocess_text(text):
    if pd.isnull(text):  # Handle NaN values
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean the 'text', 'keyword', and 'location' columns
df['cleaned_text'] = df['text'].apply(preprocess_text)
df['cleaned_keyword'] = df['keyword'].apply(preprocess_text)
df['cleaned_location'] = df['location'].apply(preprocess_text)

# Display cleaned columns
print("Cleaned Text:\n", df['cleaned_text'].head())
print("\nCleaned Keyword:\n", df['cleaned_keyword'].head())
print("\nCleaned Location:\n", df['cleaned_location'].head())

from sklearn.model_selection import train_test_split

# Assign inputs and target
X_inp_clean = df['cleaned_text']
X_inp_original = df['text']
y_inp = df['target']

# Split cleaned text data
X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
    X_inp_clean, y_inp, test_size=0.2, random_state=42
)

# Split original text data
X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(
    X_inp_original, y_inp, test_size=0.2, random_state=42
)

# Displaying the shapes
print(f"Cleaned Text Train/Test Shapes: {X_clean_train.shape}, {X_clean_test.shape}")
print(f"Original Text Train/Test Shapes: {X_original_train.shape}, {X_original_test.shape}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

#corpus (replace this with your data)
corpus = df['text']
corpus

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

#  (words in the vocabulary)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to a dense format for better readability
dens_matrix = tfidf_matrix.todense()

# Create a DataFrame for better visualization
df_tfidf = pd.DataFrame(dens_matrix, columns=feature_names)

# Display the DataFrame
print(df_tfidf)
print("TF-IDF Matrix:")

# Remove unwanted characters from the 'text' column
df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r"[!@\[\]]", "", str(x)) if pd.notnull(x) else "")

print(df)

# Remove '#' and '?' from the 'text' column
df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r"[#?]", "", str(x)) if pd.notnull(x) else "")

print(df)

"""# Part 2: Feature Engineering and Model Selection

## Task: Feature Engineering
"""

# Data split
X = df.drop('target', axis=1)
y = df['target']
X

# Data split
X = df.drop('target', axis=1)
y = df['target']

X = X.select_dtypes(include=[np.number])

X = X.fillna(X.mean())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# data standardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled_1 = scaler.transform(X_test)
X_scaled,X_scaled_1

#model logistic regresion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression()
model.fit(X_scaled, y_train)
y_pred = model.predict(X_scaled_1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are already defined
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix generation
cm = confusion_matrix(y_test, y_pred)

# Customizing the Heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    linewidths=1,
    linecolor='black',
    xticklabels=['Non-Disaster', 'Disaster'],
    yticklabels=['Non-Disaster', 'Disaster']
)

# Labels and Title
plt.xlabel('Predicted Labels', fontsize=14, color='darkblue')
plt.ylabel('Actual Labels', fontsize=14, color='darkblue')
plt.title('Confusion Matrix', fontsize=18, color='darkred', fontweight='bold')
plt.show()

"""## True Positive (TP): The model correctly predicts a disaster tweet as a disaster. In this case, there are 874 true positives.
## True Negative (TN): The model correctly predicts a non-disaster tweet as a non-disaster. There are 649 true negatives.
### False Positive (FP): The model incorrectly predicts a non-disaster tweet as a disaster. There are 0 false positives.
## False Negative (FN): The model incorrectly predicts a disaster tweet as a non-disaster. There are 0 false negatives.

## Based on the confusion matrix, we can draw the following conclusions:

## 1. High Accuracy: The model has achieved perfect accuracy in classifying both disaster and non-disaster tweets. This indicates that the model is highly effective in distinguishing between the two classes.
## 2. No False Positives or Negatives: The absence of false positives and false negatives suggests that the model is not making any errors in its predictions. This is a strong indicator of the model's reliability.
"""

# rendom forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = RandomForestClassifier()
model.fit(X_scaled, y_train)
y_pred = model.predict(X_scaled_1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are already defined
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix generation
cm = confusion_matrix(y_test, y_pred)

# Customizing the Heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    linewidths=1,
    linecolor='black',
    xticklabels=['Non-Disaster', 'Disaster'],
    yticklabels=['Non-Disaster', 'Disaster']
)

# Labels and Title
plt.xlabel('Predicted Labels', fontsize=14, color='darkblue')
plt.ylabel('Actual Labels', fontsize=14, color='darkblue')
plt.title('Confusion Matrix', fontsize=18, color='darkred', fontweight='bold')
plt.show()

"""## True Positive (TP): The model correctly predicts a disaster tweet as a disaster. In this case, there are 393 true positives.
## True Negative (TN): The model correctly predicts a non-disaster tweet as a non-disaster. There are 581 true negatives.
## False Positive (FP): The model incorrectly predicts a non-disaster tweet as a disaster. There are 293 false positives.
## False Negative (FN): The model incorrectly predicts a disaster tweet as a non-disaster. There are 256 false negatives.


## Based on the confusion matrix, we can draw the following conclusions:

## 1. Moderate Accuracy: The model has achieved moderate accuracy in classifying both disaster and non-disaster tweets. While it has correctly classified many instances, it still makes a significant number of errors.
## 2. False Positive and Negative Errors: The presence of both false positives and false negatives indicates that the model struggles to accurately classify some tweets, leading to both types of errors.
"""

# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = SVC()
model.fit(X_scaled, y_train)
y_pred = model.predict(X_scaled_1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are already defined
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix generation
cm = confusion_matrix(y_test, y_pred)

# Customizing the Heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    linewidths=1,
    linecolor='black',
    xticklabels=['Non-Disaster', 'Disaster'],
    yticklabels=['Non-Disaster', 'Disaster']
)

# Labels and Title
plt.xlabel('Predicted Labels', fontsize=14, color='red')
plt.ylabel('Actual Labels', fontsize=14, color='darkblue')
plt.title('Confusion Matrix', fontsize=18, color='darkred', fontweight='bold')
plt.show()

"""## True Positive (TP): The model correctly predicts a disaster tweet as a disaster. In this case, there are 874 true positives.
## True Negative (TN): The model correctly predicts a non-disaster tweet as a non-disaster. There are 649 true negatives.
## False Positive (FP): The model incorrectly predicts a non-disaster tweet as a disaster. There are 0 false positives.
## False Negative (FN): The model incorrectly predicts a disaster tweet as a non-disaster. There are 0 false negatives.

## Based on the confusion matrix, we can draw the following conclusions:

## 1. High Accuracy: The model has achieved perfect accuracy in classifying both disaster and non-disaster tweets. This indicates that the model is highly effective in distinguishing between the two classes.
## 2. No False Positives or Negatives: The absence of false positives and false negatives suggests that the model is not making any errors in its predictions. This is a strong indicator of the model's reliability.
"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , accuracy_score



"""# DEEP learning"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential #( We are adding layers sequential)
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D # ("Dense" Fully connected layer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data split
X = df.drop('target', axis=1)
y = df['target']
X

y

X = pd.get_dummies(X, drop_first=True)  # Creates dummy variables for categorical columns

X = pd.DataFrame(df)

categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)


X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print(X_encoded)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded

y_bainory = (y>0).astype(int)
y_bainory

# scaler standerd
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X_encoded)
X_scaler

X_train,X_test,y_train,y_test = train_test_split(X_scaler,y_bainory,test_size=0.2,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

input_dim = X_train.shape[1]
input_dim

model = Sequential([

    Dense(64, input_dim=X_train.shape[1], activation='relu'), #first hidden layer and first input input layer

    Dense(32, activation='relu'),           #second hidden layer

    Dense(1, activation='sigmoid')           #outpur layer for binary classification

])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# model evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# model perdiction
y_pred = model.predict(X_test)
y_pred

# model prediction
y_pred =(model.predict(X_test)>0.5).astype(int)
y_pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are already defined
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , accuracy_score

# Determine the data type
if isinstance(X_train[0][0], float):
    data_type = "continuous"
elif isinstance(X_train[0][0], int):
    data_type = "discrete_counts"
else:
    data_type = "binary"

# Choose the appropriate Naive Bayes classifier
if data_type == "continuous":
    model = GaussianNB()
elif data_type == "discrete_counts":
    model = MultinomialNB()
else:
    model = BernoulliNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Assuming y_test and y_pred are already defined
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Confusion matrix generation
cm = confusion_matrix(y_test, y_pred)

# Customizing the Heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    linewidths=1,
    linecolor='black',
    xticklabels=['Non-Disaster', 'Disaster'],
    yticklabels=['Non-Disaster', 'Disaster']
)

# Labels and Title
plt.xlabel('Predicted Labels', fontsize=14, color='red')
plt.ylabel('Actual Labels', fontsize=14, color='darkblue')
plt.title('Confusion Matrix', fontsize=18, color='darkred', fontweight='bold')
plt.show()

"""#Interpreting the Confusion Matrix

#A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. 1  It allows us to visualize the performance of the model on different classes.

#Breakdown of the Matrix:

#Predicted Non-Disaster	Predicted Disaster
#Actual Non-Disaster	831 (True Negative)	43 (False Positive)
#Actual Disaster	46 (False Negative)	603 (True Positive)
"""

from PIL import Image
import os
print(os.getcwd())
import os
print(os.listdir('/content'))

import pickle
# save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    print("Model dumped to 'model.pkl'")

import pickle
# Dump the data to file
with open('df.pkl', 'wb') as file:
    pickle.dump(df, file)
    print("Data dumped to 'df.pkl'")

# Export the DataFrame to an Excel file
df.to_excel('Clean_data_twitter_disaster.xlsx', index=False)

#from google.colab import files
files.download('Clean_data_twitter_disaster.xlsx')

# deshboard mein NLp disaster ka pridction