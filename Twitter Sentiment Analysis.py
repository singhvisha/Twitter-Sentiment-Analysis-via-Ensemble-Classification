#!/usr/bin/env python
# coding: utf-8

# # MAJOR PROJECT - TWITTER DATASET

# In[1]:


import pandas as pd, re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')


# In[2]:


nltk.download('punkt')
nltk.download()


# In[15]:


info_df=pd.read_csv('Information.csv',encoding='latin-1')


# # EXPLORATORY DATA ANALYSIS

# In[16]:


info_df.columns


# In[17]:


info_df=info_df[[ 'gender', 'gender:confidence', 'description',
        'link_color', 'name', 'sidebar_color', 'text', 'tweet_count']]


# In[18]:


info_df


# In[19]:


info_df.info()


# In[20]:


info_df = info_df[info_df['gender'].notna()]


# In[21]:


info_df.reset_index(drop=True,inplace=True)
info_df


# In[22]:


#Number of males and females
import matplotlib.pyplot as plt
info = info_df.groupby('gender').size()
print(type(info))
brand = info[0]
females = info[1]
males = info[2]
unknowns = info[3]

print(brand,females,males,unknowns)

genderData = {'brand':brand,
             'females':females,
             'males':males,
             'unknowns':unknowns}

types = list(genderData.keys())
count = list(genderData.values())

plt.bar(types,count,color='blue',width=0.4)
plt.xlabel("TYPES OF GENDERS") 
plt.ylabel("FREQUENCY") 
plt.title("GENDERS IN THE GIVEN DATASET") 
plt.show()


# In[23]:


genderConfidenceCount = info_df.groupby('gender:confidence').size()
gcData = genderConfidenceCount.to_dict()
gcData
x = list(gcData.keys())
y = list(gcData.values())
plt.plot(x, y)
plt.xlabel('GENDER:CONFIDENCE') 
plt.ylabel('COUNT')
plt.title('TYPES OF GENDER CONFIDENCE AND THEIR COUNTS') 
plt.show() 


# In[24]:


plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)


# In[26]:


info_df.info()


# In[27]:


info_df=info_df[info_df['gender:confidence']>0.8]
info_df


# In[28]:


info_df.reset_index(drop=True,inplace=True)
info_df


# In[29]:


info_df=info_df[info_df['gender']!='unknown']


# In[30]:


info_df.reset_index(drop=True, inplace=True)


# In[31]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('English')


# In[32]:


clean_texts = []
for i in range(info_df.shape[0]):
    current_message = info_df['text'].values[i]
    tokenized_words = word_tokenize(current_message)
    clean_message = ''
    for word in tokenized_words:
        if word not in stop_words and not word.startswith('@'):
            try:
                word = int(word)
            except:
                clean_message += word + ' '
                
    clean_texts.append(clean_message)        
    
len(clean_texts)


# In[33]:


info_df['clean_texts']=clean_texts
info_df


# Question 1:
# ----------------------
# Male and Female top used words.
# -------

# In[34]:


female_tweets = info_df[info_df['gender'] == 'female']
print('Female Tweets:')
print(female_tweets['text'].head())

print('\nMale Tweets:')
male_tweets = info_df[info_df['gender'] == 'male']
print(male_tweets['text'].head())


def get_clean_tweet(text):
    url_regex = r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
    twitter_handle_regex = r'(@[a-zA-Z_]+){1,15}'
    hashtag_regex = r'#[a-z-A-Z_\']+'
    clean_text = re.sub(url_regex, '', text)
    clean_text = re.sub(twitter_handle_regex, '', clean_text)
    clean_text = re.sub(hashtag_regex, '', clean_text)
    word_list = re.findall(r"[A-Za-z']+", clean_text)
    return ' '.join(word_list)

def get_word_count(tweets):
    words = dict()
    stop_words = stopwords.words("english")
    for tweet in enumerate(tweets):
        word_list = re.findall(r"[A-Za-z']+", tweet[1])
        for word in word_list:
            word = word.lower()
            if word in stop_words:
                continue
            words[word] = words.get(word, 0) + 1
    return words

female_tweets['clean_text'] = female_tweets['text'].apply(lambda tweet: get_clean_tweet(tweet))
male_tweets['clean_text'] = male_tweets['text'].apply(lambda tweet: get_clean_tweet(tweet))

female_words = get_word_count(female_tweets['clean_text'])
male_words = get_word_count(male_tweets['clean_text'])


sorted_female = {k: v for k, v in sorted(female_words.items(), key=lambda item: item[1], reverse=True)}
sorted_male = {k: v for k, v in sorted(male_words.items(), key=lambda item: item[1], reverse=True)}
print('Female top used words:')
for i in list(sorted_female.items())[:5]:
    print('{}: {}'.format(i[0], i[1]))
print('\nMale top used words:')
for i in list(sorted_male.items())[:5]:
    print('{}: {}'.format(i[0], i[1]))


# In[35]:


#TOP 5 WORDS MOST USED BY MALES AND FEMALES
#BY FEMALES
xF = list(sorted_female.keys())[:5]
yF = list(sorted_female.values())[:5]
xM = list(sorted_male.keys())[:5]
yM = list(sorted_male.values())[:5]
p1 = plt.bar(xF,yF,color='blue',width=0.6)
p2 = plt.bar(xM,yM,color='red',width=0.4)
plt.xlabel("WORDS") 
plt.ylabel("FREQUENCY") 
plt.title("COMPARISON OF TOP 5 WORDS MOST USED BY MALES AND FEMALES") 
plt.legend((p1[0], p2[0]), ('FEMALES', 'MALES'))
plt.show()


# Question 2:
# ------
# Is the gender male or female?
# -------------

# 
# Method 1: USING RANDOM FOREST CLASSIFICATION
# -----------

# In[71]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
X = info_df['clean_texts']
Y = info_df['gender']
X = cv.fit_transform(X)
X_train,X_test, y_train, y_test = train_test_split(X,info_df['gender'])


# In[72]:


from sklearn.ensemble import RandomForestClassifier


# In[73]:


rfClassifier = RandomForestClassifier(n_estimators=100)


# In[74]:


rfClassifier.fit(X_train,y_train)


# In[75]:


y_pred=rfClassifier.predict(X_test)


# In[76]:


from sklearn import metrics


# In[77]:


accuracy = metrics.accuracy_score(y_test, y_pred)


# In[78]:


accuracy


# Accuracy of RFC: 56.9%                                                                                                  
# 

# Method 2: Using Naive Bayes
# ----------

# Is the gender male or female?

# # NAIVE BAYES

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer


# In[46]:


cv = CountVectorizer()
sparse_data = cv.fit_transform(info_df['clean_texts'])


# In[60]:


X= sparse_data


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(sparse_data, info_df['gender'])


# In[22]:


from sklearn.naive_bayes import MultinomialNB


# Attempt-1:

#                   

# In[23]:


clf = MultinomialNB()


# In[24]:


clf.fit(X_train, y_train)


# In[25]:


predicted = clf.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test, predicted)


# Accuracy of Naive Bayes: 58.2%

#           

#                    

# In[28]:


## Concatenating description to clean_texts


# In[29]:


info_df['text+description']=info_df['clean_texts'].astype(str)+' '+ info_df['description'].astype(str)


# In[30]:


info_df


# Attempt-2:

#     

# In[31]:


sparse_data2 = cv.fit_transform(info_df['text+description'])


# In[32]:


X_train,X_test, y_train, y_test = train_test_split(sparse_data2, info_df['gender'])


# In[33]:


clf.fit(X_train, y_train)


# In[34]:


y_predicted = clf.predict(X_test)


# In[35]:


accuracy_score(y_test, y_predicted)


# Accuracy of Naive Bayes: 68.3% 

#  

#  

# In[36]:


## Concatenate name to text+ description


# Attempt-3:

#  

# In[37]:


info_df['text+description+name']=info_df['clean_texts'].astype(str)+' '+ info_df['description'].astype(str)+' '+info_df['name'].astype(str)


# In[38]:


sparse_data3 = cv.fit_transform(info_df['text+description+name'])


# In[39]:


X_train,X_test, y_train, y_test = train_test_split(sparse_data3, info_df['gender'])


# In[40]:


clf.fit(X_train, y_train)


# In[41]:


y_predicted = clf.predict(X_test)


# In[42]:


accuracy_score(y_test, y_predicted)


# Accuracy for Naive Bayes: 69.6%

#  

#  

# Method 3: Using KNN without Libraries
# -------------

# Is the gender male or female?

# # KNN from Scratch

# In[68]:


import math
def levenshtein(seq1, seq2):
    s1=seq1
    s2=seq2
#     if(!pd.isnull(seq1)):
#         s1 = seq1
#     if(!pd.isnull(seq2)):
#         s2 = seq2
        
#     print(s1,s2)
    size_x = len(s1) + 1
    size_y = len(s2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
#     print (matrix)
    return (matrix[size_x - 1, size_y - 1])

#Euclidean distance for numeric values
def get_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
#         print(row1[i],row2[i])
        if(isinstance(row1[i], int) or isinstance(row1[i], float)):
            distance += (row1[i] - row2[i])**2
        elif(isinstance(row1[i],str)):
            if(pd.isnull(row2[i])):
                distance += levenshtein(row1[i]," ")
            else:
                distance += levenshtein(row1[i],row2[i])
#     print(sqrt(distance))
    return distance

# Utility function for sorting
def getDistanceFromTuple(ele):
    return ele[1]


def KNNAlgorithmPredictor(testRow,trainDataSet):
#     Get input
#     clean_texts = input("Clean text")
#     description = input("Description")
#     gender_confidence = (int)(input("Gender confidence"))
#     link_color = input("Link colour")
#     name = input("Name")
#     sidebar_color = input("Sidebar color")
#     text = input("text")
#     textDescription = clean_texts+description
#     textDescriptionName = clean_texts+description+name
#     tweet_count = (int)(input("tweet count"))
    
#     Make the data in form that it can be compared
#     tweetCount = (float)(tweet_count)
#     genderConfidence = (float)(gender_confidence)
#     testingData = {
#         'gender':'',
#         'gender:confidence':genderConfidence,
#         'description':description,
#         'link_color':link_color,
#         'name':name,
#         'sidebar_color':sidebar_color,
#         'text':text,
#         'tweet_count':tweetCount,
#         'clean_texts':clean_texts,
#         'text+description':textDescription,
#         'text+description+name':textDescriptionName
#     }
# #     print(testingData)
#     toBePredictedRow = pd.Series(testingData)
    
#     Find the distances between the test row we formed and other rows in the dataset
    print("train set size : ",len(trainDataSet))
    distances = list()
    j = 0
    for i in range(1,len(trainDataSet)):
        distances.append((i-1,get_distance(testRow,trainDataSet.iloc[i])))
        print(distances[j])
        j+=1

#     Sort the distances to get the least at the top
    distances.sort(key = getDistanceFromTuple)
    
    # Now taking √n as number of neighbours
    n_neighbors = (int)(math.sqrt(len(info_df)))
    nearestNeighbors = []
    for i in range(0,n_neighbors):
        nearestNeighbors.append(distances[i])
    
    
#     And finally Count which gender comes most number of times and assign it to the prediction
    ans = "brand"
    male = 0
    female = 0
    brand = 0

    for i in nearestNeighbors:
        index = i[0]
        df = info_df.loc[index]['gender']
    #     print(df)
        if(df=='male'):
            male+=1
        elif(df=='female'):
            female+=1
        elif(df=='brand'):
            brand+=1
    
    gender = []
    gender.append(('male',male))
    gender.append(('female',female))
    gender.append(('brand',brand))
    gender.sort(reverse=True,key=getDistanceFromTuple)
    ans = gender[0][0]

#     if(male > female):
#         ans = ans.replace(ans,'male')
#     else:
#         ans = ans.replace(ans,'female')
        
#     And then return the answer
    return ans


# In[69]:


# KNNAlgorithmPredictor()


# In[64]:


# To find the accuracy of the self-made algorithm
testSet = pd.DataFrame()
trainSet = pd.DataFrame()
forChecking = []
# # Splitting the dataset into 80% and 20%
# # 80% of dataset - Training and 20% of remaining dataset - Testset

trainSize = (int)(0.4*len(info_df))
for i in range(0,trainSize):
    trainData = {
        'gender':info_df.iloc[i]['gender'],
        'gender:confidence':info_df.iloc[i]['gender:confidence'],
        'description':info_df.iloc[i]['description'],
        'link_color':info_df.iloc[i]['link_color'],
        'name':info_df.iloc[i]['name'],
        'sidebar_color':info_df.iloc[i]['sidebar_color'],
        'text':info_df.iloc[i]['text'],
        'tweet_count':info_df.iloc[i]['tweet_count'],
        'clean_texts':info_df.iloc[i]['clean_texts'],
        'text+description':info_df.iloc[i]['text+description'],
        'text+description+name':info_df.iloc[i]['text+description+name']
    }
    trainSet = trainSet.append(trainData,ignore_index=True)

for i in range(trainSize,(trainSize+50)+1):
    forChecking.append(info_df.iloc[i]['gender'])
    testData = {
        'gender':'',
        'gender:confidence':info_df.iloc[i]['gender:confidence'],
        'description':info_df.iloc[i]['description'],
        'link_color':info_df.iloc[i]['link_color'],
        'name':info_df.iloc[i]['name'],
        'sidebar_color':info_df.iloc[i]['sidebar_color'],
        'text':info_df.iloc[i]['text'],
        'tweet_count':info_df.iloc[i]['tweet_count'],
        'clean_texts':info_df.iloc[i]['clean_texts'],
        'text+description':info_df.iloc[i]['text+description'],
        'text+description+name':info_df.iloc[i]['text+description+name']
    }
    testSet = testSet.append(testData,ignore_index=True)


# In[76]:


# Starting the prediction
# print(testSet.iloc[0])
score = 0
# testSet.iloc[7]
for i in range(len(testSet)-10,len(testSet)):
    prediction = KNNAlgorithmPredictor(testSet.iloc[i],trainSet)
    print(prediction, forChecking[i])
    if(prediction==forChecking[i]):
        score+=1
    print(score)
        
print(score/len(testSet))


# In[77]:


print('Accuracy : ',(score/10)*100)


# Accuracy of KNN is: 40%

#  

#  
