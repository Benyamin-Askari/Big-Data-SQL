# Databricks notebook source

# MAGIC # Task 1: Analysis of Clinical Trial Data Using Spark SQL

# MAGIC %md
# MAGIC ##1. Load Data
# MAGIC In this section, we first inspect the raw CSV file as text to understand its structure before proceeding with schema inference and formal loading.

# COMMAND ----------

# MAGIC %md
# MAGIC ###1.1. Header Inspection
# MAGIC Before formally reading the CSV as a structured DataFrame, it is important to validate:
# MAGIC
# MAGIC Whether the file contains a header row.
# MAGIC
# MAGIC The delimiter used in the file (e.g., comma, pipe, etc.).
# MAGIC
# MAGIC General data cleanliness.
# MAGIC
# MAGIC **To do this, the raw CSV was read as plain text**

# COMMAND ----------

# Loading the raw CSV content as plain text to inspect its structure
raw_df = spark.read.text("/FileStore/tables/Clinicaltrial.csv")

# COMMAND ----------

# raw CSV content
raw_df.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC The first row contains the column names (NCT Number, Study Title, Acronym, etc.).
# MAGIC
# MAGIC The second row is actual data from a clinical trial.
# MAGIC
# MAGIC Comma (,) is the separator.
# MAGIC
# MAGIC **This confirms that the file does include a header, and is comma-separated as expected.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###1.2. Load the Dataset Properly with Schema
# MAGIC After inspection, we now load the dataset properly:
# MAGIC
# MAGIC Specify that the file has a header.
# MAGIC
# MAGIC Ask Spark to infer the schema automatically based on data types.

# COMMAND ----------

# loading the datset
trials_df = spark.read\
    .option("header", True)\
    .option("inferSchema", True)\
    .csv("/FileStore/tables/Clinicaltrial.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC **This creates a structured DataFrame with:**
# MAGIC
# MAGIC Properly labeled columns.
# MAGIC
# MAGIC Automatically detected types like string, date, or integer based on the column contents.

# COMMAND ----------

# MAGIC %md
# MAGIC ##2. Exploratory Data Analysis (EDA)
# MAGIC This section aims to develop an initial understanding of the structure, completeness, and characteristics of the clinical trial dataset before moving forward to answering the core questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.1. Visual Inspection of DataFrame
# MAGIC To preview the dataset visually within the Databricks notebook, the following command was used.
# MAGIC
# MAGIC This renders a scrollable, tabular interface showing column names and their sample values.

# COMMAND ----------

# data summary
display(trials_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **So far, the datset apears to have a lot of missing and invalid values. it is up to the following analysis to decide how to deal with them**

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.2. Schema Inspection

# COMMAND ----------

# Importing required libraries for the data exploration
from pyspark.sql.functions import (
    col, count, when, split, explode, months_between,
    to_date, year, trim
)

# COMMAND ----------

# MAGIC %md
# MAGIC Before performing computations, it's essential to verify the structure of the DataFrame.
# MAGIC
# MAGIC **This includes checking:**
# MAGIC
# MAGIC Number of columns
# MAGIC
# MAGIC Column names
# MAGIC
# MAGIC Data types
# MAGIC
# MAGIC Nullability of fields
# MAGIC
# MAGIC To do this, we call .printSchema() on the DataFrame.

# COMMAND ----------

# Schema and basic statistics
trials_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC All columns are of string type including dates and numeric fields like Enrollment.
# MAGIC
# MAGIC Spark inferred all fields as nullable, which is standard but reinforces the need to check for missing values in the next step.
# MAGIC
# MAGIC Dates will need to be explicitly cast to date type for time-based analysis such as calculating duration (Q3).

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.3. Summary Statistics
# MAGIC To gain a quick overview of the dataset’s numerical and string-based characteristics, we use the built-in describe()**** function. While primarily designed for numerical values, it still gives us thw count of non-null entries per column, min, max values (useful for string sorting or ID ranges), and mean, stddev (only when numeric).

# COMMAND ----------

# summary stats for numeric columns
display(trials_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC The column Acronym only has ~145k rows, suggesting high missingness, something to address in the next step.
# MAGIC
# MAGIC Fields like Study Title and Study Status appear mostly populated and will be useful for analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.4. Dataset Dimensions
# MAGIC Before performing any analysis, it is useful to understand the overall shape of the dataset like:
# MAGIC
# MAGIC Total number of rows (records)
# MAGIC
# MAGIC Total number of columns (fields per record)

# COMMAND ----------

# Total rows
trials_df.count()

# COMMAND ----------

# Total columns
len(trials_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.5. Missing Values Count per Column
# MAGIC To ensure data quality, it is essential to identify columns that have missing or empty ("") values. These nulls must be understood before running any transformations or SQL queries, especially when columns are central to answering project questions.

# COMMAND ----------

# Missing values count per column
missing_exprs = [
    count(when(col(c).isNull() | (col(c) == ""), c)).alias(c)
    for c in trials_df.columns
]
missing_df = trials_df.select(*missing_exprs)
display(missing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Fields like NCT Number, Study Title, and Study Status are almost completely filled — these will form reliable axes for our grouping and filtering later.
# MAGIC
# MAGIC Some fields are only relevant to specific trials (e.g., Acronym, Collaborators) and thus nullable by design.
# MAGIC
# MAGIC For core columns used in the four analysis questions (like Study Type, Conditions, Completion Date), missing data will need to be filtered or handled.

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.6. Value Counts of Key Categorical Fields
# MAGIC To understand the most common values in key fields and guide filtering and grouping operations for SQL tasks later, we count frequency distributions for:
# MAGIC
# MAGIC **Study Type**
# MAGIC
# MAGIC **Study Status**
# MAGIC
# MAGIC **Funder Type**

# COMMAND ----------

#  Value counts of key categorical fields
display(trials_df.groupBy("Study Type").count().orderBy(col("count").desc()))
display(trials_df.groupBy("Study Status").count().orderBy(col("count").desc()))
display(trials_df.groupBy("Funder Type").count().orderBy(col("count").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC **For all columns, there are several anomalous values are present and will need to be filtered out before aggregations**

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.7 Date Normalization
# MAGIC Clinical trials in this dataset have Start Date and Completion Date recorded using inconsistent formats like **yyyy-MM**, **dd/MM/yyyy**, or **yyyy / MM / dd**. These must be normalized before calculating trial durations.
# MAGIC
# MAGIC The code below uses PySpark abd Removes all delimiters, Splits parts and identifies Year, Month, Day. Also, falls back to the 1st day when it's missing.
# MAGIC
# MAGIC Then extracts year, month, day no matter their order and constructs a clean DateType column using **make_date()**. This creates a proper date safely for Spark operations.
# MAGIC
# MAGIC we also overwrite the original Start Date and Completion Date columns directly with fully parsed, standardized Spark DateType columns.

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace, trim, split, expr, lit, when
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

# Overwrite original Start Date and Completion Date columns
def normalize_date_column(df, input_col):
    """
    Normalize and overwrite a mixed-format date column in place.
    Supports formats like:
    - yyyy-MM-dd, yyyy-MM, yyyy/MM, yyyy MM
    - dd/MM/yyyy, dd / MM / yyyy
    - Handles spaces, slashes, dashes
    """
    cleaned_col = "cleaned_" + input_col.replace(" ", "_")

    # Step 1: Remove all non-digits and standardize delimiters as space
    df = df.withColumn(cleaned_col, regexp_replace(trim(col(input_col)), "[^0-9]", " "))
    df = df.withColumn("parts", split(col(cleaned_col), " "))
    df = df.withColumn("parts", expr("filter(parts, x -> x != '')"))

    # Step 2: Extract parts and infer year-month-day positions
    df = df.withColumn("part1", col("parts").getItem(0).cast(IntegerType()))
    df = df.withColumn("part2", col("parts").getItem(1).cast(IntegerType()))
    df = df.withColumn("part3", col("parts").getItem(2).cast(IntegerType()))

    df = df.withColumn("year", when(col("part1") > 1900, col("part1"))
                                 .when(col("part3") > 1900, col("part3"))
                                 .otherwise(None))

    df = df.withColumn("month", when(col("part1") > 1900, col("part2"))
                                  .when(col("part3") > 1900, col("part2"))
                                  .otherwise(col("part1")))

    df = df.withColumn("day", when(col("part1") > 1900, col("part3"))
                                .when(col("part3") > 1900, col("part1"))
                                .otherwise(lit(1)))

    # Step 3: Overwrite the original column with the parsed DateType version
    df = df.withColumn(input_col, F.expr("make_date(year, month, day)"))

    # Step 4: Clean up helper columns
    return df.drop(cleaned_col, "parts", "part1", "part2", "part3", "year", "month", "day")


# COMMAND ----------

# MAGIC %md
# MAGIC Next, we update both columns in Spark’s DateType to ensure downstream calculations (like durations) work reliably.

# COMMAND ----------

# Apply in-place normalization
trials_df = normalize_date_column(trials_df, "Start Date")
trials_df = normalize_date_column(trials_df, "Completion Date")


# COMMAND ----------

# MAGIC %md
# MAGIC After this, it uses Spark’s **months_between()** function to compute how long each clinical trial lasted
# MAGIC
# MAGIC Results stored in new column: Duration_Months
# MAGIC
# MAGIC Summary statistics (count, mean, min, max, quartiles) are calculated using .summary()

# COMMAND ----------

from pyspark.sql.functions import months_between

# Calculate trial length in months
duration_df = trials_df.withColumn("Duration_Months", months_between(col("End"), col("Start")))
display(duration_df.select("Start", "End", "Duration_Months").summary())


# COMMAND ----------

# MAGIC %md
# MAGIC ###2.8 Identifying Rows Missing All Required Fields
# MAGIC To check if there are any rows in the dataset that are completely missing all essential information required to answer any of the core questions (Q1–Q4). This is a defensive step to:
# MAGIC
# MAGIC Prevent data skew during filtering
# MAGIC
# MAGIC Validate data integrity after transformations (especially post date normalization)

# COMMAND ----------

# Identify rows missing *all* required columns for any question
# Define all columns used across the 4 questions
required_cols = [
    "Study Type",      # Q1
    "Conditions",      # Q2 & Q4
    "Start Date",      # Q3
    "Completion Date", # Q3 & Q4
    "Study Status"     # Q4
]
# Build predicate: True if every required column is null or empty
from functools import reduce
import operator

all_blank = reduce(
    operator.and_,
    [(col(c).isNull() | (col(c) == "")) for c in required_cols]
)

# Filter to those fully blank rows
totally_blank = trials_df.filter(all_blank)
# Display them and their count
display(totally_blank)
print(f"Rows missing every required field: {totally_blank.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Now, based on the missing value results, we decide to keep the rows with missing values to prevent affecting the columns that are totaly useful for a particular question and filter out missing values based on quesions.**

# COMMAND ----------

# MAGIC %md
# MAGIC ##3. Registering the Cleaned DataFrame for SQL Queries
# MAGIC the next step is to prepare the dataset for SQL-based analysis. This involves registering the trials_df DataFrame as a temporary SQL view. This line makes the trials_df available inside Spark SQL using the alias "trials"

# COMMAND ----------

# Register the existing DataFrame as a temp view
trials_df.createOrReplaceTempView("trials")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Question 1 - Most Frequent Study Types

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create a Filtered View of Study Types with Frequency ≥ 8
# MAGIC As we saw in the EDA steps, the values more than 8 characters were invalide; therefore the threshold of 8 is set to only use the valid study types.
# MAGIC
# MAGIC The code groups all records by type of study, filters out very rare or erroneous categories, ensures missing data is not counted, saves the result as a temporary SQL view for re-use

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create temp view of study types with at least 8 entries
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW q1_filtered AS
# MAGIC SELECT `Study Type`, COUNT(*) AS frequency
# MAGIC FROM trials
# MAGIC WHERE `Study Type` IS NOT NULL
# MAGIC GROUP BY `Study Type`
# MAGIC HAVING COUNT(*) >= 8;

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 2: Total Count of Included Records

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Calculates the total number of rows from the original 'trials'
# MAGIC SELECT SUM(frequency) AS contributed_rows_q1 FROM q1_filtered;

# COMMAND ----------

# MAGIC %md
# MAGIC This confirms that majority of the dataset is valid and usable for this analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: Display the Frequency Table
# MAGIC the code below uses the **q1_filtered** to count frequency and order the study types

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show all study types from 'q1_filtered', sorted by highest frequency
# MAGIC
# MAGIC SELECT * FROM q1_filtered
# MAGIC ORDER BY frequency DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the results, the majority of studies are **INTERVENTIONAL** and the least is **OTHER_GOV**

# COMMAND ----------

# MAGIC %md
# MAGIC ##Question 2 - Top 10 Most Frequent Medical Conditions
# MAGIC The "Conditions" column in the clinical trials dataset contains one or more medical conditions per trial, separated by pipes (|).
# MAGIC
# MAGIC **The code below will:**
# MAGIC
# MAGIC Split multivalued entries
# MAGIC
# MAGIC Normalize each individual condition
# MAGIC
# MAGIC Identify the top 10 most common medical conditions studied across all trials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Normalize and Split Conditions Column
# MAGIC The code flattens the array so each condition gets its own row and cleans whitespace from condition names. Then ensures blank/missing entries are excluded

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create temp view of individual, non-empty conditions split from the 'Conditions' column
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW q2_filtered AS
# MAGIC SELECT TRIM(c) AS Condition
# MAGIC FROM trials
# MAGIC LATERAL VIEW explode(split(Conditions, '\\|')) AS c
# MAGIC WHERE Conditions IS NOT NULL AND Conditions <> '';

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 2: Count Total Number of Contributing Condition Records

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Count total number of extracted conditions in 'q2_filtered'
# MAGIC
# MAGIC SELECT COUNT(*) AS contributed_rows_q2 FROM q2_filtered;

# COMMAND ----------

# MAGIC %md
# MAGIC This confirms that over 914,000 condition records exist, after exploding multi-condition trials.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: Rank Top 10 Most Common Conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get top 10 most frequent non-empty conditions from 'q2_filtered'
# MAGIC
# MAGIC SELECT Condition, COUNT(*) AS frequency
# MAGIC FROM q2_filtered
# MAGIC WHERE Condition IS NOT NULL AND Condition <> ''
# MAGIC GROUP BY Condition
# MAGIC ORDER BY frequency DESC
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Based on the results:**
# MAGIC
# MAGIC Cancer-related conditions (Breast, Prostate, general Cancer) are heavily represented.
# MAGIC
# MAGIC Cardiometabolic disorders like Obesity and Hypertension are equally dominant.
# MAGIC
# MAGIC “Healthy” individuals show up strongly; indicating frequent inclusion of control groups.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Question 3 - Mean Duration of Clinical Trials (Months)
# MAGIC Since the date normalization step overwrote the raw columns and parsed them into Spark DateType, we now rely directly on the cleaned "Start Date" and "Completion Date" columns.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1: Filter for Valid Dates and Calculate Duration
# MAGIC The code below, calculates the time in months between start and end dates, ensures we exclude trials with missing dates so we don't calculate incorrect durations, and 	saves this clean subset for later analysis and counting.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create temp view with trials that have start and completion dates, adding duration in months
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW q3_filtered AS
# MAGIC SELECT *,
# MAGIC   months_between(`Completion Date`, `Start Date`) AS Duration_Months
# MAGIC FROM trials
# MAGIC WHERE `Start Date` IS NOT NULL AND `Completion Date` IS NOT NULL;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The q3_normalized view now contains just a single column: the year each completed study ended.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 2: Count Contributing Records
# MAGIC This tells us how many rows had valid Start Date and Completion Date fields and thus contributed to the duration calculation.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Count rows with valid start and completion dates in 'q3_filtered'
# MAGIC
# MAGIC SELECT COUNT(*) AS contributed_rows_q3 FROM q3_filtered;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC this shows there are a lot of missing values in the two date columns

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: Compute Mean Duration in Months
# MAGIC The code uses **AVG(...)** to compute the arithmetic mean across all valid durations. the results are rounded to 2 decimal places for clarity

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Calculate average trial duration (in months), rounded to 2 decimals
# MAGIC
# MAGIC SELECT ROUND(AVG(Duration_Months), 2) AS average_duration_months
# MAGIC FROM q3_filtered;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **The average clinical trial lasts 32.12 months, just over 2.5 years**

# COMMAND ----------

# MAGIC %md
# MAGIC ##Question 4 - Completed Diabetes-Related Trials by Year

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1: Filter and Extract Completion Year
# MAGIC The code below, Extracts the year from the cleaned Completion Date, Ensures we only analyze finished trials, Finds all entries that mention diabetes (case-sensitive match), and makes a Temporary view for reuse in counting and plotting.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create temp view of completion years for completed diabetes-related studies
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW q4_filtered AS
# MAGIC SELECT
# MAGIC   year(`Completion Date`) AS Completion_Year
# MAGIC FROM trials
# MAGIC WHERE `Completion Date` IS NOT NULL
# MAGIC   AND `Study Status` = 'COMPLETED'
# MAGIC   AND (Conditions LIKE '%Diabetes%' OR Conditions LIKE '%diabetes%');
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 2: Count Contributing Records
# MAGIC This gives us the number of diabetes-related, completed trials with a known end date.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Count completed diabetes-related studies with known completion years
# MAGIC
# MAGIC SELECT COUNT(*) AS contributed_rows_q4 FROM q4_filtered;

# COMMAND ----------

# MAGIC %md
# MAGIC only **5053** studies meet the criteria

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: Count Per Year to Analyze Trend
# MAGIC The code below extracts **Completion_Year** from the parsed Completion Date then counts the number of records (trials) completed in each year. Afterwards the trials are grouped by year so counts are calculated. In the end, it ensures the output is sorted chronologically for trend visualization.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Count diabetes-related completed trials per year
# MAGIC
# MAGIC SELECT Completion_Year, COUNT(*) AS diabetes_trials
# MAGIC FROM q4_filtered
# MAGIC GROUP BY Completion_Year
# MAGIC ORDER BY Completion_Year;

# COMMAND ----------

# MAGIC %md
# MAGIC As evident, the number of studies had remained steady until **2010** when it started to increase gradually. Then from **2013**, the increase became exponentially sharper and peaked at **2019** followed by fluctuations until **now**.
