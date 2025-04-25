
%%configure -f
{
    "executorSize": "Small",
    "numExecutors": 3
}

# Step 1: Load Data
df = spark.sql("SELECT * FROM dbo.credcardfrauddata")

# Step 2: Preprocess and Scale
from pyspark.ml.feature import VectorAssembler, StandardScaler
feature_cols = [col for col in df.columns if col != 'Class']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
scaled = scaler.fit(assembled).transform(assembled)

data = scaled.select("features", "Class").withColumnRenamed("Class", "label")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 3: Models
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from synapse.ml.lightgbm import LightGBMClassifier

lr = LogisticRegression(featuresCol="features", labelCol="label")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
lgbm = LightGBMClassifier(labelCol="label", featuresCol="features", numLeaves=31, numIterations=100)

lr_model = lr.fit(train_data)
rf_model = rf.fit(train_data)
lgbm_model = lgbm.fit(train_data)

lr_pred = lr_model.transform(test_data)
rf_pred = rf_model.transform(test_data)
lgbm_pred = lgbm_model.transform(test_data)

# Step 4: Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def evaluate_model(pred_df, label="label", prediction="prediction", raw_pred="rawPrediction"):
    evaluator = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol=raw_pred, metricName="areaUnderROC")
    auc = evaluator.evaluate(pred_df)
    predictionAndLabels = pred_df.select(prediction, label).rdd.map(tuple)
    metrics = MulticlassMetrics(predictionAndLabels)
    accuracy = metrics.accuracy
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1 = metrics.fMeasure(1.0)
    return {"AUC": auc, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

lr_metrics = evaluate_model(lr_pred)
rf_metrics = evaluate_model(rf_pred)
lgbm_metrics = evaluate_model(lgbm_pred)

# Step 5: Visualization (Offline-friendly using matplotlib)
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Random Forest', 'LightGBM']
auc = [lr_metrics['AUC'], rf_metrics['AUC'], lgbm_metrics['AUC']]
f1 = [lr_metrics['F1'], rf_metrics['F1'], lgbm_metrics['F1']]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(models, auc)
plt.title("AUC Score")
plt.ylabel("AUC")

plt.subplot(1, 2, 2)
plt.bar(models, f1)
plt.title("F1 Score")
plt.ylabel("F1")

plt.tight_layout()
plt.savefig("model_performance.png")
plt.show()
