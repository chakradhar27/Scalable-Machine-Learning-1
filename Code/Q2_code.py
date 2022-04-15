"""
@author: User_200206552
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Assignment_1 Question 1") \
        .config("spark.local.dir","/fastdata/acp20cvs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir("/fastdata/acp20cvs")  # for handling stackoverflow exception while running CV tuning
sc.setLogLevel("WARN")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit,percent_rank
from pyspark.sql import Window

# load in ratings data
ratings = spark.read.load('../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
ratings.show(20,False)
#ratings.filter(($"UserId".isNull) or ($"MovieId".isNull) or ($"Rating".isNull) or ($"timestamp".isNull)).show()

# A:
# A.1
#sorting the ratings df by timestamp and assign percent_rank to the rows 
data = ratings.withColumn("rank", percent_rank().over(Window.partitionBy(lit("a")).orderBy("timestamp"))).cache()

myseed = 200206552

# splitting the data as per the given ratios
train_df_1 = data.where("rank <= .5").drop("rank").cache()
test_df_1 = data.where("rank > .5").drop("rank").cache()

train_df_2 = data.where("rank <= .65").drop("rank").cache()
test_df_2 = data.where("rank > .65").drop("rank").cache()

train_df_3 = data.where("rank <= .8").drop("rank").cache()
test_df_3 = data.where("rank > .8").drop("rank").cache()

train_df_1.show(5, False)
test_df_1.show(5, False)

#function to evaluate and get the metrics from the model
def get_metrics(model, test_data):
    predictions = model.transform(test_data)
    #evaluator for rmse
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    #evaluator for mse
    evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
    mse = evaluator.evaluate(predictions) 
    #evaluator for mae
    evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
    mae = evaluator.evaluate(predictions)  
    return rmse, mse, mae

#function to print the metrics
def print_metrics(rmse, mse, mae):
    print("Root-mean-square error = " + str(rmse))
    print("Mean square error = " + str(mse))  
    print("Mean absolute error = " + str(mae))
  
# A.2 | A.3
print("==================== A ====================")
#als model for 
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

# ALS_1
print("==================== ALS_1, 50 split ====================")

model_als_1_50 = als.fit(train_df_1)
rmse, mse, mae = get_metrics(model_als_1_50, test_df_1)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_1, 65 split ====================")

model_als_1_65 = als.fit(train_df_2)
rmse, mse, mae = get_metrics(model_als_1_65, test_df_2)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_1, 80 split ====================")

model_als_1_80 = als.fit(train_df_3)
rmse, mse, mae = get_metrics(model_als_1_80, test_df_3)
print_metrics(rmse,mse,mae)

print("====================================================")

#Hyperparameter tuning
# Import the requisite packages

#from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#param_grid = ParamGridBuilder() \
#            .addGrid(als.rank, [10, 30, 50]) \
#            .addGrid(als.regParam, [.05, .1, .2]) \
#            .addGrid(als.maxIter, [10, 30, 50]) \
#            .build()

#evaluator = RegressionEvaluator(
#           metricName="rmse", 
#           labelCol="rating", 
#           predictionCol="prediction") 
#print ("Num models to be tested: ", len(param_grid))

# Build cross validation using CrossValidator
#cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

#Fit cross validator to the 'train' dataset
#model = cv.fit(train_df_3)
#Extract best model from the cv model above
#best_model = model.bestModel

# Print "Rank"
#print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
#print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
#print("  RegParam:", best_model._java_obj.parent().getRegParam())

# ALS_2
#second als configuration with rank=50, maxIter=50 
#Increasing the rank (number of latent factors) will improve the performance till a optumum value and after that the model starts to overfit.
#maxIter indicates the number of times to alternate between (P & Q) lower rank user matrix and item matrix so by increasing it will lead to better optimization in finding the min error.
als_2 = ALS(userCol="userId", itemCol="movieId", rank=50, maxIter=50, seed=myseed, coldStartStrategy="drop")
print("==================== ALS_2, 50 split ====================")

model = als_2.fit(train_df_1)
rmse, mse, mae = get_metrics(model, test_df_1)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_2, 65 split ====================")

model = als_2.fit(train_df_2)
rmse, mse, mae = get_metrics(model, test_df_2)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_2, 80 split ====================")

model = als_2.fit(train_df_3)
rmse, mse, mae = get_metrics(model, test_df_3)
print_metrics(rmse,mse,mae)

print("====================================================")

# ALS_3
#third als configuration with rank=75, maxIter=50, regParam=0.05 
#as the rank is increased the number of features increases so there might be chance of overfit hence regParam is chosen a small value so that a small penalty is added over every iteration inorder for the model to fit the factors at min error.
als_3 = ALS(userCol="userId", itemCol="movieId", rank=75, maxIter=50, regParam=0.05, seed=myseed, coldStartStrategy="drop")

print("==================== ALS_3, 50 split ====================")

model = als_3.fit(train_df_1)
rmse, mse, mae = get_metrics(model, test_df_1)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_3, 65 split ====================")

model = als_3.fit(train_df_2)
rmse, mse, mae = get_metrics(model, test_df_2)
print_metrics(rmse,mse,mae)

print("====================================================")

print("==================== ALS_3, 80 split ====================")

model = als_3.fit(train_df_3)
rmse, mse, mae = get_metrics(model, test_df_3)
print_metrics(rmse,mse,mae)

print("====================================================")

print("====================================================")


# B:
print("==================== B ====================")
# B.1
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

#kmeans model with 20 clusters
kmeans = KMeans(k=20, seed=myseed)

print("==================== kmeans_ALS_1, 50 split ====================")
dfuserfactors_1 = model_als_1_50.userFactors
model_1 = kmeans.fit(dfuserfactors_1)
model_1.hasSummary
summary = model_1.summary
clusters = summary.clusterSizes
lg_cluster_indx_1 = clusters.index(max(clusters))
print(clusters)
clusters.sort(reverse=True)
print(f"Top 3 cluster sizes for split 1:")
print(clusters[:3])
predictions_1 = model_1.transform(dfuserfactors_1)
predictions_1.show(10, False)
print("====================================================")


print("==================== kmeans_ALS_1, 65 split ====================")
dfuserfactors_2 = model_als_1_65.userFactors
model_2 = kmeans.fit(dfuserfactors_2)
model_2.hasSummary
summary = model_2.summary
clusters = summary.clusterSizes
lg_cluster_indx_2 = clusters.index(max(clusters))
print(clusters)
clusters.sort(reverse=True)
print(f"Top 3 cluster sizes for split 2:")
print(clusters[:3])
predictions_2 = model_2.transform(dfuserfactors_2)
print("====================================================")

print("==================== kmeans_ALS_1, 80 split ====================")
dfuserfactors_3 = model_als_1_80.userFactors
model_3 = kmeans.fit(dfuserfactors_3)
model_3.hasSummary
summary = model_3.summary
clusters = summary.clusterSizes
lg_cluster_indx_3 = clusters.index(max(clusters))
print(clusters)
clusters.sort(reverse=True)
print(f"Top 3 cluster sizes for split 3:")
print(clusters[:3])
predictions_3 = model_3.transform(dfuserfactors_3)
print("====================================================")

# B.2
from pyspark.sql.functions import split, explode
#import movies data
movies = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
movies.show(10, False)
#to seperate out the genres and add as new rows
movies_df = movies.withColumn("genres", explode(split("genres", "[|]")))

#function to get top n genres for the users from the largest cluster from kmeans
def get_genres(data_set, predictions,largest_cluster_indx, num_genres=5):
  #combine movies data with the train or test set of different splits
  combined_df = movies_df.join(data_set, movies_df.movieId == data_set.movieId).drop(data_set.movieId)
  #filter users of the largest cluster from predictions
  user_lg_cl_1_df = predictions.select('*').where(predictions.prediction == largest_cluster_indx)
  #join or subset the combined data for the users from largest cluster
  combined_df_us_cl = combined_df.join(user_lg_cl_1_df, combined_df.userId == user_lg_cl_1_df.id).drop('features','prediction','id')
  #filter for the movies with user ratings greater than or equal to 4
  combined_df_us_cl = combined_df_us_cl.filter(combined_df_us_cl.rating >= 4.0)
  #most frequent genres
  top_genres = combined_df_us_cl.select('genres').groupBy('genres').count().orderBy('count', ascending=False)
  #list of top n genres from the rdd
  top_genres = top_genres.select('genres').rdd.flatMap(lambda x: x).collect()[:num_genres]

  return top_genres


#B.2.1
print("==================== B_2_1, 50:50 split ====================")
top_genres = get_genres(train_df_1, predictions_1, lg_cluster_indx_1, 5)
print("Top 5 genres from training set of split 1")
print(top_genres)
print("=========================================================")
top_genres = get_genres(test_df_1, predictions_1, lg_cluster_indx_1, 5)
print("Top 5 genres from test set of split 1")
print(top_genres)
print("=========================================================")

#B.2.2
print("==================== B_2_2, 65:35 split ====================")
top_genres = get_genres(train_df_2, predictions_2, lg_cluster_indx_2, 5)
print("Top 5 genres from training set of split 2")
print(top_genres)
print("=========================================================")
top_genres = get_genres(test_df_2, predictions_2, lg_cluster_indx_2, 5)
print("Top 5 genres from test set of split 2")
print(top_genres)
print("=========================================================")

#B.2.3
print("==================== B_2_3, 80:20 split ====================")
top_genres = get_genres(train_df_3, predictions_3, lg_cluster_indx_3, 5)
print("Top 5 genres from training set of split 3")
print(top_genres)
print("=========================================================")
top_genres = get_genres(test_df_3, predictions_3, lg_cluster_indx_3, 5)
print("Top 5 genres from test set of split 3")
print(top_genres)
print("===========================================================")

print("============================================================")
