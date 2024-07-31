import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
df = pd.read_csv('Hackathon_website\mental_health.csv',encoding='latin-1')
# df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
# remove duplicates
df = df.drop_duplicates(keep='first')
import nltk
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df[['num_characters','num_words','num_sentences']].describe()
df[df['label'] == 0][['num_characters','num_words','num_sentences']].describe()
df[df['label'] == 1][['num_characters','num_words','num_sentences']].describe()
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

df['text'][10]

transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

ps.stem('loving')

df['transformed_text'] = df['text'].apply(transform_text)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)

# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))

X.shape

y = df['label'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import BernoulliNB
m = BernoulliNB()
m.fit(X_train, y_train)
import pickle
pickle.dump(tfidf,open('vectorizer3.pkl','wb'))
pickle.dump(m,open('final_model2.pkl','wb'))

# tfidf = pickle.load(open('vectorizer3.pkl','rb'))
# model = pickle.load(open('final_model.pkl','rb'))