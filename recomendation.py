##fetching data 
from sklearn.metrics import silhouette_score
import pandas as pd
from matplotlib import pyplot as plt
df_courses = pd.read_csv('courses.csv')

## preprossing data 

df_courses = df_courses.dropna(how='any')

df_courses['Description'] = df_courses['Description'].replace({"'ll": " "}, regex=True)
df_courses['CourseId'] = df_courses['CourseId'].replace({"-": " "}, regex=True)
comb_frame = df_courses.CourseId.str.cat(" "+df_courses.CourseTitle.str.cat(" "+df_courses.Description))
comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
#comb_frame = (comb_frame.iloc[:3000])
##converting data to vectors
#
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)
#
#from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_bow = count_vect.fit_transform(comb_frame)
#
#from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_bow)
#X_train_tf_temp = tf_transformer.transform(X_train_bow)
#
from sklearn.decomposition import LatentDirichletAllocation, PCA
#lda = LatentDirichletAllocation(n_components=5,random_state=0)
#lda.fit(X_train_tf_temp)
#X_train_tf=lda.transform(X_train_tf_temp)

#Elbow method to determine true_k value

# Continuing after vectorization step
# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 50
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
    comb_frame["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.savefig('elbow_method.png')

#K-means for fitting model

#true_k, derived from elbow method and confirmed from pluralsight's website
true_k = 40
# Running model with 15 different centroid initializations & maximum iterations are 500
model = KMeans(n_clusters=true_k,init='k-means++', max_iter=500, n_init=15)
model.fit(X)
#silhouette_score = silhouette_score(X_train_tf ,model.labels_, metric='euclidean')

X_train_tf_a=X.todense()
pca = PCA(n_components = 5).fit(X_train_tf_a)
coords = pca.transform(X_train_tf_a)
plt.scatter(coords[:,0],coords[:,1])
centroids = model.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:,0,],centroid_coords[:,1],marker = 'X', s = 200, linewidths=2, c = "#444d61")
plt.show()

#from sklearn.cluster import AgglomerativeClustering
#X_train_tf_a= X_train_tf.todense()
#Agm = AgglomerativeClustering().fit(X_train_tf_a)
#silhouette_score = silhouette_score(X_train_tf_a ,Agm.labels_, metric='euclidean')

from sklearn.mixture import GaussianMixture
X_train_tf_a= X_train_bow.todense()
gm = GaussianMixture(n_components=5, random_state=0).fit(X_train_tf_a)
label=gm.predict(X_train_tf_a)
silhouette_score = silhouette_score(X_train_tf_a ,label, metric='euclidean')
# Check top terms in clutsers
#Top terms in each clusters.
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
	
## Predict cluster numbers for each predictions
def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

# Create new column for storing predicted categories from our trained model.
df_courses['ClusterPrediction'] = ""


# load the complete data in a dataframe
df_courses = pd.read_csv('courses.csv')
# drop retired course from analysis. But, courses with no descriptions are kept.
df_courses = df_courses[df_courses.IsCourseRetired == 'no']
    
# create new column in dataframe which is combination of (CourseId, CourseTitle, Description) in existing data-frame
df_courses['InputString'] = df_courses.CourseId.str.cat(" "+df_courses.CourseTitle.str.cat(" "+df_courses.Description))
# Create new column for storing predicted categories from our trained model.
df_courses['ClusterPrediction'] = ""
# Cluster category for each live course
df_courses['ClusterPrediction']=df_courses.apply(lambda x: cluster_predict(df_courses['InputString']), axis=0)

## Function to recomeded similiar books

def recommend_util(str_input):
    
    # match on the basis course-id and form whole 'Description' entry out of it.
    temp_df = df_courses.loc[df_courses['CourseId'] == str_input]
    temp_df['InputString'] = temp_df.CourseId.str.cat(" "+temp_df.CourseTitle.str.cat(" "+temp_df['Description']))
    str_input = list(temp_df['InputString'])
    
    # Predict category of input string category
    prediction_inp = cluster_predict(str_input)
    prediction_inp = int(prediction_inp)
    # Based on the above prediction 10 random courses are recommended from the whole data-frame
    # Recommendation Logic is kept super-simple for current implementation.
    temp_df = df_courses.loc[df_courses['ClusterPrediction'] == prediction_inp]
    temp_df = temp_df.sample(10)
    
    return list(temp_df['CourseId'])


################Testing / To get recomendation for each book ##################33

queries = ['play-by-play-machine-learning-exposed']
#, 'microsoft-cognitive-services-machine-learning', 'python-scikit-learn-building-machine-learning-models', 'pandas-data-wrangling-machine-learning-engineers', 'xgboost-python-scikit-learn-machine-learning']
for query in queries:
    res = recommend_util(query)
    mid = [x.replace('-', ' ') for x in res]
    final = ','.join(mid)
