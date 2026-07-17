"""Transformations DataFrame et Spark SQL (Séance 3 du cours)."""

from pyspark.sql import Column, DataFrame, SparkSession


def select_columns(df: DataFrame, *columns: str) -> DataFrame:
    return df.select(*columns)


def filter_rows(df: DataFrame, condition: Column) -> DataFrame:
    return df.filter(condition)


def group_and_count(df: DataFrame, column: str) -> DataFrame:
    return df.groupBy(column).count()


def sort_by(df: DataFrame, column: str, ascending: bool = True) -> DataFrame:
    return df.orderBy(column, ascending=ascending)


def create_temp_view(df: DataFrame, name: str) -> None:
    df.createOrReplaceTempView(name)


def run_sql(spark: SparkSession, query: str) -> DataFrame:
    return spark.sql(query)
