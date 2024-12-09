# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import webbrowser
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import plotly.io as pio
import plotly.express as px 
import nltk

# %%


# %%
app_df = pd.read_csv("Play Store Data.csv")
reviews_df = pd.read_csv("User Reviews.csv")

# %%
app_df.head()

# %%
reviews_df.head()

# %%
app_df['Installs'].unique()

# %%

app_df['Installs'] = app_df['Installs'].replace({'Free': '0'})  
app_df['Installs'] = app_df['Installs'].str.replace(',', '')  
app_df['Installs'] = app_df['Installs'].str.replace('+', '')  

# Convert the cleaned values to integers
app_df['Installs'] = app_df['Installs'].astype(int)

# Verify the changes
print(app_df['Installs'].head())


# %%
app_df['Price'] = app_df['Price'].str.replace('$', '')  # Remove dollar sign

# %%
merge_df = pd.merge(app_df,reviews_df,on='App',how='inner')

# %%
merge_df.head()

# %%


# %%
app_df['Size'].unique()

# %%
app_df['Size'].unique()

# %%
app_df['Log_installs'] = np.log(app_df['Installs'])


# %%
# Convert 'Reviews' column to integers, handle errors by coercing non-numeric values to NaN
app_df['Reviews'] = pd.to_numeric(app_df['Reviews'], errors='coerce')

# Optionally, fill NaN values with 0
app_df['Reviews'] = app_df['Reviews'].fillna(0).astype(int)

# Display the updated DataFrame
print(app_df)


# %%
app_df.info()

# %%
app_df['log_Reviews'] = np.log(app_df['Reviews'])

# %%
def rating_group(rating):
    if rating >= 4:
        return 'Top rated App'
    elif rating >= 3:
        return "Above Average"
    elif rating >= 2:
        return "Average"
    else:
        return "Below Average"
app_df['Rating_Group'] = app_df['Rating'].apply(rating_group)

# %%
app_df['Revenue'] = app_df['Price']*app_df['Installs']

# %%
import nltk

# %%
nltk.download('vader_lexicon')

# %%
sia = SentimentIntensityAnalyzer()

# %%
review = "This app is amazing! I love this app"
sentiment_score = sia.polarity_scores(review)
print(sentiment_score)

# %%
review = "This app is very bad! I hate the features"
sentiment_score = sia.polarity_scores(review)
print(sentiment_score)

# %%
review = "This app is okay."
sentiment_score = sia.polarity_scores(review)
print(sentiment_score)

# %%
reviews_df['Sentiment_Scores'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# %%
app_df['Last Updated'] = pd.to_datetime(app_df['Last Updated'],errors='coerce')

# %%
app_df['Year'] = app_df['Last Updated'].dt.year

# %%
app_df.head()

# %%
html_files_path = "./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# filerting for paid apps only 
paid_apps = app_df[app_df['Type'] == 'Paid']


plt.figure(figsize=(10, 6))
sns.scatterplot(data=paid_apps, x='Installs', y='Revenue', hue='Category', palette='Set2', s=60)

X = paid_apps['Installs'].values.reshape(-1, 1)
y = paid_apps['Revenue'].values
model = LinearRegression().fit(X, y)
trendline = model.predict(X)
plt.plot(paid_apps['Installs'], trendline, color="blue", linestyle="--", label="Trendline")

plt.xlabel('Number of Installs')
plt.ylabel('Revenue')
plt.title('Relationship between Revenue and Installs for Paid Apps')
plt.legend(title="Category")
plt.show()


# %%
import pandas as pd
import plotly.express as px


top_5_categories = app_df['Category'].value_counts().nlargest(5).index

# Filter for only the top 5 categories
filtered_df = app_df[app_df['Category'].isin(top_5_categories)]
fig = px.bar(filtered_df, 
             x='Category', 
             y='Installs', 
             color='Category', 
             title='Top 5 Categories by Installs',
             labels={'Installs': 'Number of Installs', 'Category': 'App Category'},
             category_orders={'Category': top_5_categories})

fig.show()



# %%
fig = px.scatter(filtered_df, 
                 x='Category', 
                 y='Installs', 
                 size='Installs', 
                 color='Category', 
                 hover_name='App', 
                 title='Bubble Chart: Installs by Category', 
                 labels={'Installs': 'Number of Installs', 'Category': 'App Category'})
fig.show()


# %%
from datetime import datetime


current_time = datetime.now().time()
if not (current_time >= datetime.strptime("12:00", "%H:%M").time() and 
        current_time <= datetime.strptime("18:00", "%H:%M").time()):
    fig = px.bar(filtered_df, 
                 x='Category', 
                 y='Installs', 
                 color='Category', 
                 title='Top 5 Categories by Installs')
    fig.show()
else:
    print("This graph is unavailable between 12 PM and 6 PM.")


# %%
import plotly.express as px
from datetime import datetime
filtered_df = app_df[(app_df.groupby('Category')['Category'].transform('count') > 50) &
                 (app_df['App'].str.contains("C")) &
                 (app_df['Reviews'] >= 10) &
                 (app_df['Rating'] < 4.0)]

# Doesnt display between 6PM to 11PM
current_time = datetime.now().time()
if not (current_time >= datetime.strptime("18:00", "%H:%M").time() and 
        current_time <= datetime.strptime("23:00", "%H:%M").time()):
    fig = px.violin(filtered_df, y="Rating", x="Category", box=True, points="all",
                    title="Distribution of Ratings by App Category (Filtered)")
    fig.show()
else:
    print("This graph is unavailable between 6 PM and 11 PM.")


# %%



