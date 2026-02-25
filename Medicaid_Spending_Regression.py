# Databricks notebook source
# DBTITLE 1,Cell 1
from pyspark.sql import functions as F

volume_path = "/Volumes/workspace/default/avi-coursework/medicaid-provider-spending.parquet"

df = spark.read.parquet(volume_path)

df.printSchema()
print("Total rows (approx):", df.select(F.approx_count_distinct(F.lit(1))).collect()[0][0])
df.show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Cell 2
# Feature engineering 
df = df.withColumn("year", F.substring(F.col("CLAIM_FROM_MONTH"), 1, 4).cast("int")) \
       .withColumn("month", F.substring(F.col("CLAIM_FROM_MONTH"), 6, 2).cast("int")) \
       .withColumn("paid_per_claim", F.when(F.col("TOTAL_CLAIMS") > 0, F.col("TOTAL_PAID") / F.col("TOTAL_CLAIMS")).otherwise(0.0)) \
       .withColumn("paid_per_beneficiary", F.when(F.col("TOTAL_UNIQUE_BENEFICIARIES") > 0, F.col("TOTAL_PAID") / F.col("TOTAL_UNIQUE_BENEFICIARIES")).otherwise(0.0)) \
       .withColumn("total_paid_target", F.when(F.col("TOTAL_PAID") < 0, 0.0).otherwise(F.col("TOTAL_PAID"))) \
       .withColumn("log_total_paid_target", F.log1p(F.col("total_paid_target")))

# COMMAND ----------

# DBTITLE 1,Cell 3
from pyspark.sql.functions import lit, approx_count_distinct

# 5–10% sample (~1.1–2.2M rows) – adjust fraction if needed
df_small = df.sample(fraction=0.05, seed=42)

print("Small sample approx rows:", df_small.select(approx_count_distinct(lit(1))).collect()[0][0])

# Split
train_s, test_s = df_small.randomSplit([0.8, 0.2], seed=42)

print("Train small approx:", train_s.select(approx_count_distinct(lit(1))).collect()[0][0])
print("Test small approx:", test_s.select(approx_count_distinct(lit(1))).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Cell 4
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import time

print("Preparing features...")

# Assemble numeric features into a vector
feature_cols = ["year", "month", "TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS", "paid_per_claim", "paid_per_beneficiary"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_s_prepared = assembler.transform(train_s)
test_s_prepared = assembler.transform(test_s)

print("Training LinearRegression on small sample...")

start = time.time()
lr = LinearRegression(
    featuresCol="features",
    labelCol="log_total_paid_target",
    maxIter=10
)
lr_model = lr.fit(train_s_prepared)

lr_time = time.time() - start
print(f"Trained in {lr_time:.2f} seconds")

# Evaluate
preds = lr_model.transform(test_s_prepared)
rmse = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="rmse"
).evaluate(preds)

r2 = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="r2"
).evaluate(preds)

print(f"LinearRegression → RMSE: {rmse:.4f}   R²: {r2:.4f}")

# COMMAND ----------

# DBTITLE 1,Cell 5
from pyspark.ml.feature import VectorAssembler

print("Preparing features with HCPCS...")
feature_cols = [
    "year", "month",
    "TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS",
    "paid_per_claim", "paid_per_beneficiary",
    "hcpcs_encoded"  # ← add this back
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

train_s_prepared = assembler.transform(train_s)
test_s_prepared = assembler.transform(test_s)

print("Features assembled – ready for training")

# COMMAND ----------

# DBTITLE 1,Cell 7
# 1. trying to add HCPCS encoding 
from pyspark.ml.feature import StringIndexer, OneHotEncoder

hcpcs_indexer = StringIndexer(inputCol="HCPCS_CODE", outputCol="hcpcs_index", handleInvalid="keep")
hcpcs_encoder = OneHotEncoder(inputCol="hcpcs_index", outputCol="hcpcs_encoded", handleInvalid="keep")

print("Fitting StringIndexer...")
indexer_model = hcpcs_indexer.fit(df)
df_indexed = indexer_model.transform(df)

print("Fitting OneHotEncoder...")
encoder_model = hcpcs_encoder.fit(df_indexed)
df_encoded = encoder_model.transform(df_indexed)

df_encoded.printSchema()

# COMMAND ----------

# assembling
from pyspark.ml.feature import VectorAssembler, StandardScaler

feature_cols = [
    "year", "month",
    "TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS",
    "paid_per_claim", "paid_per_beneficiary",
    "hcpcs_encoded"  
]

print("Assembling features...")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
df_features = assembler.transform(df_encoded)

print("Scaling...")
scaler = StandardScaler(inputCol="raw_features", outputCol="features")
scaler_model = scaler.fit(df_features)
df_prepared = scaler_model.transform(df_features)

df_prepared.printSchema() 
df_prepared.show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Cell 9
# creating small small sample, spliting
df_small = df_prepared.sample(fraction=0.05, seed=42)

print("Small DF approx rows:", df_small.select(F.approx_count_distinct(F.lit(1))).collect()[0][0])

train_s, test_s = df_small.randomSplit([0.8, 0.2], seed=42)

print("Small train approx:", train_s.select(F.approx_count_distinct(F.lit(1))).collect()[0][0])
print("Small test approx:", test_s.select(F.approx_count_distinct(F.lit(1))).collect()[0][0])

# COMMAND ----------

# performing linear regression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import time

print("Training LinearRegression with full features...")
start = time.time()
lr = LinearRegression(featuresCol="features", labelCol="log_total_paid_target", maxIter=10)
lr_model = lr.fit(train_s)
lr_time = time.time() - start
print(f"Trained in {lr_time:.2f} seconds")

preds = lr_model.transform(test_s)
rmse = RegressionEvaluator(labelCol="log_total_paid_target", predictionCol="prediction", metricName="rmse").evaluate(preds)
r2   = RegressionEvaluator(labelCol="log_total_paid_target", predictionCol="prediction", metricName="r2").evaluate(preds)
print(f"LinearRegression → RMSE: {rmse:.4f}   R²: {r2:.4f}")

# COMMAND ----------

# adding second model (RandomForest) for comparation
from pyspark.ml.regression import RandomForestRegressor

print("Training tiny RandomForest...")
start = time.time()
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="log_total_paid_target",
    numTrees=10,
    maxDepth=5,
    subsamplingRate=0.3,
    seed=42
)
rf_model = rf.fit(train_s)
rf_time = time.time() - start
print(f"RandomForest trained in {rf_time:.2f} seconds")

# Evaluate
preds_rf = rf_model.transform(test_s)
rmse_rf = RegressionEvaluator(labelCol="log_total_paid_target", predictionCol="prediction", metricName="rmse").evaluate(preds_rf)
r2_rf   = RegressionEvaluator(labelCol="log_total_paid_target", predictionCol="prediction", metricName="r2").evaluate(preds_rf)
print(f"RandomForest → RMSE: {rmse_rf:.4f}   R²: {r2_rf:.4f}")

# COMMAND ----------

# trying to add GBT regressoor again
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

print("Training tiny GBT ...")

start = time.time()

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="log_total_paid_target",
    maxIter=5,             
    maxDepth=3,            
    subsamplingRate=0.1,   
    stepSize=0.1,           
    seed=42
)

gbt_model = gbt.fit(train_s)

gbt_time = time.time() - start
print(f"GBT trained in {gbt_time:.2f} seconds")

# Evaluate GBT
preds_gbt = gbt_model.transform(test_s)
rmse_gbt = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="rmse"
).evaluate(preds_gbt)

r2_gbt = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="r2"
).evaluate(preds_gbt)

print(f"GBT → RMSE: {rmse_gbt:.4f}   R²: {r2_gbt:.4f}")

# COMMAND ----------

# Final model, DecisionTreeRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

print("Training DecisionTreeRegressor ...")

start = time.time()

dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="log_total_paid_target",
    maxDepth=5,               
    maxBins=32,            
    minInstancesPerNode=10,   
    seed=42
)

dt_model = dt.fit(train_s)

dt_time = time.time() - start
print(f"Decision Tree trained in {dt_time:.2f} seconds")

# Evaluate
preds_dt = dt_model.transform(test_s)

rmse_dt = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="rmse"
).evaluate(preds_dt)

r2_dt = RegressionEvaluator(
    labelCol="log_total_paid_target",
    predictionCol="prediction",
    metricName="r2"
).evaluate(preds_dt)

print(f"DecisionTree → RMSE: {rmse_dt:.4f}   R²: {r2_dt:.4f}")

# COMMAND ----------

# Importance of features

# Linear Regression
print("LinearRegression Coefficients (top 10 by absolute value):")
coefficients = lr_model.coefficients.toArray()
feature_names = ["year", "month", "TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS", "paid_per_claim", "paid_per_beneficiary", "hcpcs_encoded"]

# absolute coefficient
ranked_lr = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)

for name, coef in ranked_lr[:10]:
    print(f"{name:35} : {coef:.6f} (abs: {abs(coef):.6f})")

# RandomForest
print("RandomForest Feature Importance (top 10):")
importance_rf = rf_model.featureImportances
feature_names = ["year", "month", "TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS", "paid_per_claim", "paid_per_beneficiary", "hcpcs_encoded"]

ranked_rf = sorted(zip(feature_names, importance_rf.toArray()), key=lambda x: x[1], reverse=True)

for name, imp in ranked_rf[:10]:
    print(f"{name:35} : {imp:.4f}")

# GBT
print("GBT Feature Importance (top 10):")
importance_gbt = gbt_model.featureImportances
ranked_gbt = sorted(zip(feature_names, importance_gbt.toArray()), key=lambda x: x[1], reverse=True)

for name, imp in ranked_gbt[:10]:
    print(f"{name:35} : {imp:.4f}")

# Decision Tree
print("\nDecisionTree Feature Importance (top 10):")
importance_dt = dt_model.featureImportances
ranked_dt = sorted(zip(feature_cols, importance_dt.toArray()), key=lambda x: x[1], reverse=True)

for name, imp in ranked_dt[:10]:
    print(f"{name:35} : {imp:.4f}")

# COMMAND ----------

# final comparison print
print("Final Model Comparison Summary (small sample):")
print(f"LinearRegression → RMSE: 2.0123   R²: 0.5363")
print(f"RandomForest     → RMSE: 0.5278   R²: 0.9681")
print(f"GBT              → RMSE: 0.6206   R²: 0.9559")
print(f"DecisionTree     → RMSE: {rmse_dt:.4f}   R²: {r2_dt:.4f}")

# COMMAND ----------

# DBTITLE 1,Cell 13
# MAGIC %skip
# MAGIC # Using RF predictions to get aggregatted results for best quality
# MAGIC preds_rf = rf_model.transform(test_s)
# MAGIC
# MAGIC agg_df = preds_rf.groupBy("year", "month").agg(
# MAGIC     F.avg("TOTAL_PAID").alias("avg_actual"),
# MAGIC     F.avg("prediction").alias("avg_predicted"),
# MAGIC     F.count("*").alias("num_records")
# MAGIC ).orderBy("year", "month")
# MAGIC
# MAGIC agg_df.coalesce(1).write.mode("overwrite").csv("/Volumes/workspace/default/avi-coursework/tableau_export.csv", header=True)
# MAGIC print("Exported to /Volumes/workspace/default/avi-coursework/tableau_export.csv – import into Tableau")

# COMMAND ----------

# DBTITLE 1,Cell 14
pdf = df_small.limit(100000).toPandas()  # to convert small sample to Pandas

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt

X = pdf[feature_cols[:-1]]  # exclude hcpcs_encoded (sparse – skip for sklearn simplicity)
y = pdf["log_total_paid_target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sk_rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
sk_rf.fit(X_train, y_train)

y_pred = sk_rf.predict(X_test)
rmse_sk = sqrt(mean_squared_error(y_test, y_pred))
r2_sk = r2_score(y_test, y_pred)

print(f"Scikit-learn RF baseline → RMSE: {rmse_sk:.4f}   R²: {r2_sk:.4f}")

# COMMAND ----------

# DBTITLE 1,Cell 17
# CSV Files for Tableau Visualization (Main)

# Data Quality and Model Pipeline Monitoring CSV
df_quality = df.groupBy("year", "month").agg(
    F.count("*").alias("num_records")
).orderBy("year", "month")

df_quality.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/dashboard1_data_quality.csv", 
    header=True
)
print("Exported: dashboard1_data_quality.csv")

# Model Performance (actual vs predicted)
preds_gbt = gbt_model.transform(test_s)

df_perf = preds_gbt.groupBy("year", "month").agg(
    F.avg("log_total_paid_target").alias("avg_actual"),
    F.avg("prediction").alias("avg_predicted")
).orderBy("year", "month")

df_perf.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/dashboard2_performance.csv", 
    header=True
)
print("Exported: dashboard2_performance.csv")

# Feature Importance Table CSV
import pandas as pd

# Get the actual number of features from the models
num_features = len(lr_model.coefficients.toArray())

# Create feature names: first 6 are named, rest are hcpcs_encoded_X
feature_names = feature_cols[:6] + [f"hcpcs_encoded_{i}" for i in range(num_features - 6)]

importance_data = {
    "Feature": feature_names,
    "LinearRegression_abs_coef": [abs(c) for c in lr_model.coefficients.toArray()],
    "DecisionTree": dt_model.featureImportances.toArray(),
    "RandomForest": rf_model.featureImportances.toArray(),
    "GBT": gbt_model.featureImportances.toArray()
}

df_importance = pd.DataFrame(importance_data)
df_importance.to_csv("/Volumes/workspace/default/avi-coursework/feature_importance.csv", index=False)
print("Exported: feature_importance.csv")

# Business Insights (with error %)
df_insights = preds_gbt.groupBy("year", "month").agg(
    F.avg("log_total_paid_target").alias("avg_actual"),
    F.avg("prediction").alias("avg_predicted")
).withColumn("error_pct", 
    (F.col("avg_predicted") - F.col("avg_actual")) / F.col("avg_actual") * 100
).orderBy("year", "month")

df_insights.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/dashboard3_insights.csv", 
    header=True
)
print("Exported: dashboard3_insights.csv")

# 4. Scalability and Cost Analysis CSV
scalability_data = [
    ("LinearRegression", 98.0, 2.0123, 0.5363, "Fast baseline"),
    ("DecisionTree", 571.35, 0.4736, 0.9743, "Simple tree, high R²"),
    ("RandomForest", 973.99, 0.5278, 0.9681, "Ensemble balance"),
    ("GBT", 2042.36, 0.6206, 0.9559, "Best performer, slowest")
]

df_scalability = spark.createDataFrame(scalability_data, 
    ["model", "training_time_seconds", "rmse", "r2", "notes"])

df_scalability.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/dashboard4_scalability.csv", 
    header=True
)
print("Exported: dashboard4_scalability.csv")


# COMMAND ----------

# Other Insights for Visualization

# Ingestion Throughput and Data Density
df_throughput = df.groupBy("year", "month").agg(
    F.count("*").alias("num_records"),
    F.countDistinct("BILLING_PROVIDER_NPI_NUM").alias("unique_providers"),
    F.avg("TOTAL_PAID").alias("avg_paid")
).orderBy("year", "month")

df_throughput.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/throughput_monthly.csv", 
    header=True
)
print("Exported: throughput_monthly.csv")

# Residuals = actual - predicted (on log scale)
df_residuals = preds.withColumn("residual", F.col("log_total_paid_target") - F.col("prediction"))
df_residuals.select(
    "year", "month", "HCPCS_CODE",
    "log_total_paid_target", "prediction", "residual"
).coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/residuals.csv", 
    header=True
)
print("Exported: residuals.csv → for histogram, boxplot, residual vs predicted scatter")

# Flag high-volume months (e.g. > mean + 2 std)
mean_rec = df_throughput.select(F.mean("num_records")).collect()[0][0]
std_rec  = df_throughput.select(F.stddev("num_records")).collect()[0][0]

df_anomalies = df_throughput.withColumn("is_anomaly", 
    F.when(F.col("num_records") > mean_rec + 2 * std_rec, 1).otherwise(0)
)

df_anomalies.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/anomalies_high_volume.csv", 
    header=True
)
print("Exported: anomalies_high_volume.csv → annotate peaks in Tableau")

# Average actual paid per month across years (for seasonal cycle)
df_seasonal = preds.groupBy("month").agg(
    F.avg("TOTAL_PAID").alias("avg_paid"),
    F.avg("log_total_paid_target").alias("avg_log_actual"),
    F.count("*").alias("month_count")
).orderBy("month")

df_seasonal.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/seasonal_cycle.csv", 
    header=True
)
print("Exported: seasonal_cycle.csv → perfect for cycle plot / small multiples by month")

# Cumulative sum of records over time
from pyspark.sql.window import Window

window_spec = Window.orderBy("year", "month")

df_cumulative = df_throughput.withColumn("cumulative_records", 
    F.sum("num_records").over(window_spec)
)

df_cumulative.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/cumulative_footprint.csv", 
    header=True
)
print("Exported: cumulative_footprint.csv → area chart of data growth")

# Annual Volume Distribution
df_annual = df.groupBy("year").agg(
    F.sum(F.lit(1)).alias("num_records"),
    F.sum("TOTAL_PAID").alias("total_paid"),
    F.avg("TOTAL_PAID").alias("avg_paid")
).orderBy("year")

df_annual.coalesce(1).write.mode("overwrite").csv(
    "/Volumes/workspace/default/avi-coursework/annual_volume.csv", 
    header=True
)
print("Exported: annual_volume.csv → treemap with size = num_records or total_paid")