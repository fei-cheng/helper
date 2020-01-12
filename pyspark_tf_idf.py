import math

import pyspark.sql.functions as F
import pyspark.sql.window as W
import pyspark.sql.types as T


def tf_idf(inputDf):
    """
    Calculating tf-idf with pyspark
    Refer to: https://dzone.com/articles/calculating-tf-idf-with-apache-spark
    """

    def calcIdfUdf(docCount, df):
        return math.log((docCount + 1.0) / (df + 1.0))

    idf_udf = F.udf(calcIdfUdf, T.DoubleType())

    documents = (
        inputDf
        .withColumn("doc_id", F.monotonically_increasing_id())
        .withColumn("document", document_udf(F.col("content")))
    )

    unfoldedDocs = (
        documents
        .withColumn("token", F.explode("document"))
    )

    tokensWithTf = (
        unfoldedDocs
        .groupBy("doc_id", "token")
        .agg(F.count("document").alias("tf"))
    )

    tokensWithDf = (
        unfoldedDocs
        .groupBy("token")
        .agg(F.countDistinct("doc_id").alias("df"))
    )

    docCount = documents.count()

    tokensWithIdf = (
        tokensWithDf
        .withColumn("idf", idf_udf(F.lit(docCount), F.col("df")))
    )

    tokensWithTfIdf = (
        tokensWithTf
        .join(tokensWithIdf, "token", "left")
        .withColumn("tf_idf", F.col("tf") * F.col("idf"))
    )

    return tokensWithTfIdf
