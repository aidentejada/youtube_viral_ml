"""
YouTube Video Performance Prediction
Predicts view counts for trending US YouTube videos using XGBoostRegressor.
Features: TF-IDF text vectorization (title/desc), temporal features, metadata flags.
Model Performance: Score ≈ 0.89
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

'''
Originally attempted to use a combined dataset from ./data/combined_videos.csv
however it performed poorly,an R^2 score of 0.099. I am not sure of the root cause.
I tried debugging to see if the columns were different but theyre all identical just
pulled from different regions. but I believe the issue is with the data itself.
US videos performed better than other regions, having a mean view count of 2.36M, while
globally (combining the data from all regions in ./data) the mean view count was 185k, the model
completely failed to predict a drastically larger distribution, cuz globally the most popular
videos have a view count of 10M+ but theres a lot more less popular videos bringing that avg down.
Because of this i reverted to using the US dataset.
I believed bigger datasets would yield better results, but i was wrong. data quality > quantity.
'''

# loading the data.

path = "./data/USvideos.csv"
df = pd.read_csv(path)

# data cleaning.
drop = [
    'channel_title', # not needed
    'trending_date', # after the fact
    'likes', # after the fact
    'dislikes',
    'thumbnail_link',
    'comment_count',
    'video_error_or_removed'
]
df.drop(drop, axis=1, inplace=True)

# check for duplicates.
df.drop_duplicates(inplace=True)
check_dup = int(df.duplicated().sum())
if check_dup > 0:
    print(f"Found {check_dup} duplicates.")
    exit(1)

# convert publish_time to datetime.
df['publish_time'] = pd.to_datetime(df['publish_time'])

# Extract features
df['publish_hour'] = df['publish_time'].dt.hour
df['publish_day'] = df['publish_time'].dt.dayofweek
df['publish_month'] = df['publish_time'].dt.month

# Drop original
df.drop('publish_time', axis=1, inplace=True)

# train, test, split
X = df.drop('views', axis=1)
y = df['views']
print(xgb.__version__)
#splitting BEFORE manipualating to prevent leakage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}
'''
The following code was written with AI assistance
Initial model scored |0.778| with my engineering method, I used claude to enrich these features.
This was necessary because when I plotted the model, i found that the title, and description
were low on the feature importance chart, which is insane because for the youtube algorithm,
having a good name and description is vital. So claude separated the vectorizations for title
and description. Then I prompted it to add more numeric columns for features like tags, and descriptions etc.
new model scored |0.890|
'''
# Vectorize title
vectorizer_title = TfidfVectorizer(max_features=50)
title_train = vectorizer_title.fit_transform(X_train['title']).toarray()
title_test = vectorizer_title.transform(X_test['title']).toarray()

# Vectorize description
vectorizer_desc = TfidfVectorizer(max_features=50)
desc_train = vectorizer_desc.fit_transform(X_train['description'].fillna('')).toarray()
desc_test = vectorizer_desc.transform(X_test['description'].fillna('')).toarray()

# Numeric columns
numeric_cols = ['category_id', 'publish_hour', 'publish_day', 'publish_month']
numeric_train = X_train[numeric_cols].values
numeric_test = X_test[numeric_cols].values

# Additional features
tag_count_train = X_train['tags'].fillna('').apply(lambda x: len(str(x).split('|'))).values.reshape(-1, 1)
tag_count_test = X_test['tags'].fillna('').apply(lambda x: len(str(x).split('|'))).values.reshape(-1, 1)

desc_length_train = X_train['description'].fillna('').apply(len).values.reshape(-1, 1)
desc_length_test = X_test['description'].fillna('').apply(len).values.reshape(-1, 1)

title_length_train = X_train['title'].apply(len).values.reshape(-1, 1)
title_length_test = X_test['title'].apply(len).values.reshape(-1, 1)

comments_disabled_train = X_train['comments_disabled'].astype(int).values.reshape(-1, 1)
comments_disabled_test = X_test['comments_disabled'].astype(int).values.reshape(-1, 1)

ratings_disabled_train = X_train['ratings_disabled'].astype(int).values.reshape(-1, 1)
ratings_disabled_test = X_test['ratings_disabled'].astype(int).values.reshape(-1, 1)

# Combine all features
X_train = np.hstack([
    title_train,
    desc_train,
    numeric_train,
    tag_count_train,
    desc_length_train,
    title_length_train,
    comments_disabled_train,
    ratings_disabled_train
])

X_test = np.hstack([
    title_test,
    desc_test,
    numeric_test,
    tag_count_test,
    desc_length_test,
    title_length_test,
    comments_disabled_test,
    ratings_disabled_test
])

'''
The Code above was written with AI assistance.
'''
grid = GridSearchCV(xgb.XGBRegressor(), params, cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# selecting best hyperparameters.
best_lr = grid.best_params_['learning_rate']
best_depth = grid.best_params_['max_depth']

model = xgb.XGBRegressor(
    n_estimators=1000, # high n_estimators for better accuracy using early stopping.
    learning_rate=best_lr,
    max_depth=best_depth,
    random_state=67,
    early_stopping_rounds=10
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=10
          ) # early stopping after 10 rounds. print every 10 rounds.

predictions = model.predict(X_test)

score = model.score(X_test, y_test)
print(f"Model score: {score}")
# plot feature importance
xgb.plot_importance(model)
plt.show()



