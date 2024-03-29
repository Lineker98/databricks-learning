# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Migration from pandas to pandas API on Spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object create - Series

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as ps # provide a simplified interface to use Spark Dataframe with familiar sintaxe as pandas dataframe

# COMMAND ----------

# create pandas series
pser = pd.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

print(pser)

# COMMAND ----------

type(pser)

# COMMAND ----------

# create pandas as spark series
psser = ps.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

print(psser)

# COMMAND ----------

type(psser)

# COMMAND ----------

# Create a pandas-on-spark series by passing a pandas series
psser_1 = ps.Series(pser)

# COMMAND ----------

psser_1

# COMMAND ----------

type(psser_1)

# COMMAND ----------

psser_2 = ps.from_pandas(pser)

# COMMAND ----------

type(psser_2)

# COMMAND ----------

# sort_index method
print(psser_1.sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object creation - DataFrame

# COMMAND ----------

my_dict = {"A": np.random.rand(5),
           "B": np.random.rand(5)}

# COMMAND ----------

my_dict

# COMMAND ----------

# Create pandas dataframe
pdf = pd.DataFrame(my_dict)
pdf

# COMMAND ----------

type(pdf)

# COMMAND ----------

# Create pandas-on-spark dataframe
psdf = ps.DataFrame(my_dict)

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

# Create a pandas-on-spark dataframe by passing a pandas dataframe
psdf_1 = ps.DataFrame(pdf)
psdf_2 = ps.from_pandas(pdf)

# COMMAND ----------

psdf_1

# COMMAND ----------

type(psdf_1)

# COMMAND ----------

psdf_2

# COMMAND ----------

type(psdf_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Data

# COMMAND ----------

# Create pandas-on-spark series
psser = ps.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

psdf = ps.DataFrame(my_dict)

# COMMAND ----------

psser

# COMMAND ----------

psdf

# COMMAND ----------

psser.head()

# COMMAND ----------

psdf.head()

# COMMAND ----------

# Summary statistics
psser.describe()

# COMMAND ----------

psdf.describe()

# COMMAND ----------

psser.sort_values()

# COMMAND ----------

psdf.sort_values(by="A")

# COMMAND ----------

psser.transpose()

# COMMAND ----------

# Max number to be displayed
ps.get_option('compute.max_rows')

# COMMAND ----------

ps.get_option('compute.max_rows', 2000)

# COMMAND ----------

ps.set_option('compute.max_rows', 2000)

# COMMAND ----------

ps.get_option('compute.max_rows', 2000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selection

# COMMAND ----------

psdf["A"]

# COMMAND ----------

psdf[["A", "B"]]

# COMMAND ----------

psser.loc[0:2]

# COMMAND ----------

# Slicing
psdf.iloc[0:4, 0:2]

# COMMAND ----------

psdf["C"] = psser

# COMMAND ----------

# Those are needed for managing options
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)
psdf['C'] = psser
 
# Reset to default to avoid potential expensive operation in the future
reset_option("compute.ops_on_diff_frames")
print(psdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Applying python function with pandas-on-spark object

# COMMAND ----------

psdf.apply(np.cumsum)

# COMMAND ----------

psdf.apply(np.cumsum, axis=1)

# COMMAND ----------

psdf.apply(lambda x: x ** 2)

# COMMAND ----------

def square(x) -> ps.Series[np.float64]:
    return x ** 2

# COMMAND ----------

psdf.apply(square)

# COMMAND ----------

# Working properly since size of data <= compute.shortcut_limit (1000)
ps.DataFrame({'A': range(1000)}).apply(lambda col: col.max())

# COMMAND ----------

# Not working properly since size of data > compute.shortcut_limit (1000)
ps.DataFrame({'A': range(1200)}).apply(lambda col: col.max())

# COMMAND ----------

# Set compute.shortcut_limit = 1200
ps.set_option('compute.shortcut_limit', 1200)

# COMMAND ----------

ps.DataFrame({'A': range(1200)}).apply(lambda col: col.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grouping data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ploting Data

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

# bar plot
 
speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
         
psdf = ps.DataFrame({'speed': speed,
                     'lifespan': lifespan}, index=index)
psdf.plot.bar()

# COMMAND ----------

psdf

# COMMAND ----------

# horizontal bar plot
psdf.plot.barh()

# COMMAND ----------

psdf = ps.DataFrame({'mass': [0.330, 4.87, 5.97],
                     'radius': [2439.7, 6051.8, 6378.1]},
                    index=['Mercury', 'Venus', 'Earth'])
psdf.plot.pie(y='radius')

# COMMAND ----------

psdf

# COMMAND ----------

psdf = ps.DataFrame({
    'sales': [3, 2, 3, 9, 10, 6, 3],
    'signups': [5, 5, 6, 12, 14, 13, 9],
    'visits': [20, 42, 28, 62, 81, 50, 90],
}, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                       freq='M'))
psdf.plot.area()

# COMMAND ----------

psdf

# COMMAND ----------

psdf = ps.DataFrame({'rabbit': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]},
                    index=[1990, 1997, 2003, 2009, 2014])
psdf.plot.line()

# COMMAND ----------

psdf = ps.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])
psdf.plot.scatter(x='length',
                  y='width',
                  c='species')

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Missing Functionalities and workarounds in pandas API on Spark

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Directly use pandas APIs through type conversion

# COMMAND ----------

psdf = ps.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

psidx = psdf.index

# COMMAND ----------

psidx

# COMMAND ----------

type(psidx)

# COMMAND ----------

psidx.to_list()

# COMMAND ----------

ps_list = psidx.to_list()

# COMMAND ----------

ps_list

# COMMAND ----------

type(ps_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Native Support for pandas Objects

# COMMAND ----------

psdf = ps.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3] * 4, dtype='int32'),
                     'F': 'foo'})

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed execution for pandas functions

# COMMAND ----------

i = pd.date_range('2018-04-09', periods=2000, freq='1D1min')
ts = ps.DataFrame({'A': ['timestamp']}, index=i)

# COMMAND ----------

ts

# COMMAND ----------

len(ts)

# COMMAND ----------

ts.between_time('0:15', '0:16')

# COMMAND ----------

ts.to_pandas().between_time('0:15', '0:16')

# COMMAND ----------

ts.pandas_on_spark.apply_batch(func=lambda pdf: pdf.between_time('0:15', '0:16'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## using SQL in Pandas API on Spark

# COMMAND ----------

psdf = ps.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                     'rabbit': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]})

# COMMAND ----------

pdf = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'sheep': [22, 50, 121, 445, 791],
                    'chicken': [250, 326, 589, 1241, 2118]})

# COMMAND ----------

psdf

# COMMAND ----------

pdf

# COMMAND ----------

ps.sql("SELECT * FROM {psdf} WHERE rabbit > 300")

# COMMAND ----------

ps.sql('''
    SELECT ps.rabbit, pd.chicken
    FROM {psdf} ps INNER JOIN {pdf} pd
    ON ps.year = pd.year
    ORDER BY ps.rabbit, pd.chicken''', psdf=psdf, pdf=pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Working with pyspark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversion from and to pyspark dataframe

# COMMAND ----------

# Creating a pandas-on-spark DataFrame
psdf = ps.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

sdf = psdf.to_spark()

# COMMAND ----------

sdf

# COMMAND ----------

sdf.show()

# COMMAND ----------

type(sdf)
# pyspark.sql.dataframe.DataFrame

# COMMAND ----------

psdf_2 = sdf.to_pandas_on_spark()

# COMMAND ----------

type(psdf_2)
#pyspark.pandas.frame.DataFrame

# COMMAND ----------

psdf_3 = sdf.pandas_api()

# COMMAND ----------

type(psdf_3)
# pyspark.pandas.frame.DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking Spark execution plans

# COMMAND ----------

from pyspark.pandas import option_context

with oprion_context(
    "compute.ops_on_diff_frames", True,
    "compute.default_index_type", 'distributed'):
    df = ps.range(10) + ps.range(10)
    df.spark.explain()

# COMMAND ----------

with option_context(
        "compute.ops_on_diff_frames", False,
        "compute.default_index_type", 'distributed'):
    df = ps.range(10)
    df = df + df
    df.spark.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caching DataFrames

# COMMAND ----------

with option_context("compute.default_index_type", 'distributed'):
    df = ps.range(10)
    new_df = (df + df).spark.cache()  # `(df + df)` is cached here as `df`
    new_df.spark.explain()

# COMMAND ----------

new_df.spark.unpersist()

# COMMAND ----------

with (df + df).spark.cache() as df:
    df.spark.explain()
