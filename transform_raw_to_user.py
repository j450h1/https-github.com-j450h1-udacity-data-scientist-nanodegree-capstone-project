# import libraries
from pyspark.sql import SparkSession
from pathlib import Path # better file paths
from pyspark.sql.functions import countDistinct, col, when, lit, count, expr
from pyspark.sql.functions import max as sparkMax #https://stackoverflow.com/questions/36924873/
import sys

####################################################################
# This script takes the raw data and transforms it into a dataset
# with the features we've selected at the user level - 1 row output per user
####################################################################

# Usage while in the ROOT of the project
#python transform_raw_to_user.py {RAW_DATA_FILENAME}
# python transform_raw_to_user.py mini_sparkify_event_data.json ##file must be in data folder

def get_churned_users(df):
    """
    Find out the users that cancelled so we can identify who churned.
    Return updated dataframe with additional column identifying as such
    """
    cancelled_ids = df.filter('page == "Cancellation Confirmation"').select("userId").distinct()
    # Convert to list to be used to filter later
    cancelled_ids = cancelled_ids.toPandas()['userId'].tolist()
    # 1 when a user churned and 0 when they did not
    df = df.withColumn("Churn", when((col("userId").isin(cancelled_ids)),lit('1')).otherwise(lit('0')))
    
    return df

def aggregate_to_user_level(df):
    """
    Aggregate the selected features to the user level
    """
    exprs = [\
    sparkMax(col('churn')).alias('churn')\
    ,sparkMax(col('Gender')).alias('gender')\
    ,sparkMax(col('level')).alias('subscription_level')\
    ,sparkMax(col('device_type')).alias('device_type')\
    ,sparkMax(when(col("page") == 'Upgrade', 1).otherwise(0)).alias('page_upgraded')
    ,sparkMax(when(col("page") == 'Downgrade', 1).otherwise(0)).alias('page_downgraded')
    ,count(when(col("auth") == 'Logged In', True)).alias('auth_logged_in_cnt')\
    ,count(when(col("auth") == 'Logged Out', True)).alias('auth_logged_out_cnt')\
    ,count(when(col("auth") == 'Guest', True)).alias('auth_guest_cnt')\
    ,count(when(col("status") == '404', True)).alias('status_404_cnt')\
    ,count(when(col("status") == '307', True)).alias('status_307_cnt')\
    ,count(when(col("page") == 'Next Song', True)).alias('page_next_song_cnt')
    ,count(when(col("page") == 'Thumbs Up', True)).alias('page_thumbs_up_cnt')
    ,count(when(col("page") == 'Thumbs Down', True)).alias('page_thumbs_down_cnt')
    ,count(when(col("page") == 'Add to Playlist', True)).alias('page_playlist_cnt')
    ,count(when(col("page") == 'Add Friend', True)).alias('page_friend_cnt')
    ,count(when(col("page") == 'Roll Advert', True)).alias('page_roll_ad_cnt')
    ,count(when(col("page") == 'Logout', True)).alias('page_logout_cnt')
    ,count(when(col("page") == 'Help', True)).alias('page_help_cnt')
    ,countDistinct('artist').alias('artist_cnt')\
    ,countDistinct('song').alias('song_cnt')\
    ,countDistinct('sessionId').alias('session_cnt')\
    ]
    # Additional feature engineering
    df = df.withColumn("device_type",\
    expr("CASE WHEN rlike(userAgent, '(Windows|Mac|Linux)') THEN 'desktop' \
    WHEN rlike(userAgent, 'iP')  THEN  'mobile' ELSE 'other' END AS device_type"))
    user_df = df.groupBy('userId')\
    .agg(*exprs)

    # Remove data with null values - needs to be added to pipeline
    user_df = user_df.where(col("gender").isNotNull()) #remove when gender is not specified
    
    return user_df

def main():
    """
    Load data, choose features, aggregate data to user level, and then export it to a csv file
    """
    
    spark = SparkSession \
        .builder \
        .appName('Sparkify') \
        .getOrCreate()

    # reading in data from local data folder which is gitignored due to large file size
    IN_FILENAME = sys.argv[1]
    print(f"\nStarting PySpark transform job to aggregate raw data to the user level: {IN_FILENAME}\n")
    event_data = Path.cwd() / "data" / f"{IN_FILENAME}" # assuming we are already in the data folder..
    #event_data = Path.cwd() / "data" / "mini_sparkify_event_data.json"
    raw = spark.read.json(str(event_data))

    # transform data
    df = get_churned_users(raw)
    user_df = aggregate_to_user_level(df)

    # export data
    OUT_FILENAME = f'TRANSFORMED_{IN_FILENAME.replace("json", "csv")}' 
    #https://stackoverflow.com/questions/36574617/how-to-write-csv-file-into-one-file-by-pyspark
    #user_df.coalesce(1).write.csv(str(Path.cwd() / "data" / OUT_FILENAME), mode="overwrite")
    user_df.toPandas().to_csv(str(Path.cwd() / "data" / OUT_FILENAME), index = False)

    print(f"\nData has been transformed and saved as: {OUT_FILENAME}\n")

if __name__ == '__main__':
    main()
