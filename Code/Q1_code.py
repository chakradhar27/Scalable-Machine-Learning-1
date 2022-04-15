"""
@author: User_200206552
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Assignment_1 Question 1") \
        .config("spark.local.dir","/fastdata/acp20cvs") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
logFile.show(20, False)

#convert log data to a dataframe after formatting the columns
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

# A:
#look for hosts that ends with specific domain address which represents the university origin by filtering the dataframe
a1_out = data.filter(data.host.endswith("ac.jp")).count()
print("==================== A.1 ====================")
print(f"There are {a1_out} requests in total from Japanese Universities.")
print("====================================================")

a2_out = data.filter(data.host.endswith("ac.uk")).count()
print("==================== A.2 ====================")
print(f"There are {a2_out} requests in total from UK Universities.")
print("====================================================")

a3_out = data.filter(data.host.endswith(".edu")).count()
print("==================== A.3 ====================")
print(f"There are {a3_out} requests in total from US Universities.")
print("====================================================")

#plot bar graph and save 
fig, ax = plt.subplots()
University_Origin = ['Japanese', 'UK', 'US']
total_requests = [a1_out, a2_out, a3_out]
ax.bar(University_Origin,total_requests,align='center')
ax.set_xlabel('University Host Origin')
ax.set_ylabel('Total Requests')
ax.set_title('Total requests from Universities of different countries', fontsize=14, pad=20)
for i in range(len(total_requests)):
    ax.annotate(str(total_requests[i]), xy=(University_Origin[i],total_requests[i]), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("../Output/Q1_bar.png")

# B:
## B.1:

#find top 9 most frequent universities from three countries			
data.createTempView("logs")  # to directly use sql query on data, tried tempview to use SQL queries

b1_out = spark.sql("SELECT logs.host, COUNT(*) as requests FROM logs wHERE logs.host like '%.ac.jp' GROUP BY host ORDER BY COUNT(*) DESC").cache()
#convert to rdd to get the host names
b1_out_host = b1_out.select('host').rdd.flatMap(lambda x: x).collect()[:9]
#convert to rdd to get the number of requests
b1_out_req = b1_out.select('requests').rdd.flatMap(lambda x: x).collect()
#splitting the host names for the main university domain
b1_out_host_head = ['.'.join(host.split(".")[-3:]) for host in b1_out_host]  
print("==================== B.1.1 ====================")
print(f"Top 9 most frequent universities (hosts) from Japan.")
print(b1_out_host_head)
print("====================================================")

#repeat of the same steps for other countries
b2_out = spark.sql("SELECT logs.host, COUNT(*) as requests FROM logs wHERE logs.host like '%.ac.uk' GROUP BY host ORDER BY COUNT(*) DESC").cache()
b2_out_host = b2_out.select('host').rdd.flatMap(lambda x: x).collect()[:9]
b2_out_req = b2_out.select('requests').rdd.flatMap(lambda x: x).collect()
b2_out_host_head = ['.'.join(host.split(".")[-3:]) for host in b2_out_host]
print("==================== B.1.2 ====================")
print(f"Top 9 most frequent universities (hosts) from UK.")
print(b2_out_host_head)
print("====================================================")

b3_out = spark.sql("SELECT logs.host, COUNT(*) as requests FROM logs wHERE logs.host like '%.edu' GROUP BY host ORDER BY COUNT(*) DESC").cache()
b3_out_host = b3_out.select('host').rdd.flatMap(lambda x: x).collect()[:9]
b3_out_req = b3_out.select('requests').rdd.flatMap(lambda x: x).collect()
b3_out_host_head = ['.'.join(host.split(".")[-2:]) for host in b3_out_host]
print("==================== B.1.3 ====================")
print(f"Top 9 most frequent universities (hosts) from US.")
print(b3_out_host_head)
print("====================================================")

# B.2:
## B.2.1
university = list(b1_out_host_head)
university.append("Others")
requests = b1_out_req[:9] + [sum(b1_out_req[9:])]

#function to plot pie chart with university host names and the number of requests (top 9 and the rest)
def plot_pie(university, requests, org):
    fig, ax = plt.subplots(figsize = (8,5))
    wedges, texts = ax.pie(requests)
    percent = [100*(x/sum(requests)) for x in requests]
    ax.legend(wedges, labels=['%s - %1.2f%%' % (l, p) for l, p in zip(university, percent)], title="University", loc="center left", bbox_to_anchor=(1, 0, 0.5, 0.5))
    ax.axis('equal') 
    if org == 'jp':
      ax.set_title('Proportion of requests among Japanese Universities', fontsize=16, pad=20)
    elif org == 'uk':
      ax.set_title('Proportion of requests among UK Universities', fontsize=16, pad=20)
    else:
      ax.set_title('Proportion of requests among US Universities', fontsize=16, pad=20)

    plt.tight_layout()

plot_pie(university, requests, 'jp')
plt.savefig("../Output/Q1_pie_JP.png")

#repeat the same steps to generate pie chart for countries
## B.2.2
university = list(b2_out_host_head)
university.append("Others")
requests = b2_out_req[:9] + [sum(b2_out_req[9:])]

plot_pie(university, requests, 'uk')
plt.savefig("../Output/Q1_pie_UK.png")

## B.2.3
university = list(b3_out_host_head)
university.append("Others")
requests = b3_out_req[:9] + [sum(b3_out_req[9:])]

plot_pie(university, requests, 'us')
plt.savefig("../Output/Q1_pie_US.png")  

# C:
from pyspark.sql.functions import udf
#process to convert time format in the logs into a suitable timestamp for casting/sql operations
month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_time(text):
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )

udf_time = udf(parse_time)

data = data.select('*', udf_time(data['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
data.show(10, False)

#import sql functions to get days and hours from parsed timestamp
from pyspark.sql.functions import dayofmonth, hour

## C.1  
#filter the data for the Japan university with most frequent requests (can contain multiple domains)
jp_uni_df = data.select('host', 'time', dayofmonth(data.time).alias('day'), hour(data.time).alias('hour')).where(data.host.endswith(b1_out_host_head[0])).cache()
jp_uni_df.show(10, False)

#function to generate the 2D-array/Matrix of day and hourly requests
def get_heatmap_array(uni_df):
    #group by day and hour to get the distribution
    hm_df = uni_df.select('*').groupBy("day","hour").count().orderBy("day", "hour")
    #covert rdd to collect the triplets of day, hour and requests count
    request_map = hm_df.select('*').rdd.map(lambda x: [x[0],x[1],x[2]]).collect()
    #list of unique days of the visits
    unique_days = hm_df.select('day').distinct().rdd.flatMap(lambda x: x).collect()
    unique_days.sort()
    hm_array = np.zeros((31,24))
    #loop over the request_map to create the 2D array
    for i in range(0, len(request_map)):
      dl = request_map[i]
      hm_array[dl[0]-1][dl[1]] = dl[2]
    hm_array = hm_array[min(unique_days)-1:max(unique_days), :]
    return unique_days, hm_array

unique_days, hm_array = get_heatmap_array(jp_uni_df)

#function to plot heat map to visualise the distribution of the requets from the most visited university across the hours of the days in July month
def plot_hm(hm_array, org):
    fig, ax = plt.subplots(figsize = (10,9))
    im = ax.imshow(hm_array, cmap='YlGn')
    ax.set_xticks(np.arange((max(unique_days)-min(unique_days))+1))
    ax.set_yticks(np.arange(24))
    ax.set_xticklabels(range(min(unique_days),max(unique_days)+1))
    ax.set_yticklabels(range(24))
    ax.set_xlabel('Day of the Month', fontsize=14)
    ax.set_ylabel('Hour of visit', fontsize=14)
    # loop over the array to create requests annotations.
    for i in range(24):
      for j in range((max(unique_days)-min(unique_days))+1):
          text = ax.text(j, i, int(hm_array[i, j]),
                       ha="center", va="center", color="black")
  
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    if org == 'jp':
      ax.set_title("Requests from u-tokyo.ac.jp University", fontsize=18, pad=20)
    elif org=='uk':
      ax.set_title("Requests from hensa.ac.uk University", fontsize=18, pad=20)
    else:
      ax.set_title("Requests from msstate.edu University", fontsize=18, pad=20)
    fig.tight_layout()

plot_hm(hm_array.T, 'jp')  
plt.savefig("../Output/Q1_heatmap_JP.png")

## C.2 
#filter the data for the UK university with most frequent requests (can contain multiple domains)
uk_uni_df = data.select('host', 'time', dayofmonth(data.time).alias('day'), hour(data.time).alias('hour')).where(data.host.endswith(b2_out_host_head[0])).cache()
uk_uni_df.show(10, False)
unique_days, hm_array = get_heatmap_array(uk_uni_df)
plot_hm(hm_array.T, 'uk')
plt.savefig("../Output/Q1_heatmap_UK.png")

## C.3
#filter the data for the US university with most frequent requests (can contain multiple domains)
us_uni_df = data.select('host', 'time', dayofmonth(data.time).alias('day'), hour(data.time).alias('hour')).where(data.host.endswith(b3_out_host_head[0])).cache()
us_uni_df.show(10, False)
unique_days, hm_array = get_heatmap_array(us_uni_df)
plot_hm(hm_array.T, 'us')
plt.savefig("../Output/Q1_heatmap_US.png")

 