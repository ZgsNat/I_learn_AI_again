import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2,SelectKBest, SelectPercentile
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from imblearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"final_project.ods")

data = pd.read_excel(data_path, engine="odf", dtype=str)

def filter_location(location: str) -> str:
    result = re.findall("\,\s[A-Z]{2}$",location)
    if len(result) > 0:
        return result[0][2:]
    else:
        return location
    
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1009,stratify=y)

ros = SMOTEN(random_state=42, k_neighbors=2,sampling_strategy={
    "bereichsleiter": 1000,
    "director_business_unit_leader": 500,
    "specialist": 500,
    "managing_director_small_medium_company": 100,
})
sampler = RandomOverSampler(random_state=42, sampling_strategy={
    "bereichsleiter": 1000,
    "director_business_unit_leader": 500,
    "specialist": 500,
    "managing_director_small_medium_company": 100,
})
print(y_train.value_counts())
print("---------------------")

# x_train, y_train = ros.fit_resample(x_train, y_train)
# print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english"), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"),["location"]),
    ("description", TfidfVectorizer(stop_words="english",ngram_range=(1,2)),"description"),
    ("function",OneHotEncoder(),["function"]),
    ("industry", TfidfVectorizer(stop_words="english"), "industry")
])

classifier = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("sampler",sampler),
    ("feature_selector",SelectPercentile(chi2)),
    ("classifier",RandomForestClassifier(random_state=1009,n_jobs=-1))
])

params = {
    "feature_selector__percentile": [10],
    "preprocessor__description__min_df": [0.05],
    "preprocessor__description__max_df": [0.95]
}

model = GridSearchCV(
    estimator=classifier,
    param_grid=params,
    cv=3,
    verbose=0,
    n_jobs=-1
)

model.fit(x_train,y_train)
y_predict = model.predict(x_test)

print(classification_report(y_test,y_predict))