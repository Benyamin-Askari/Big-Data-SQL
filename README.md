# Big Data-SQL
This is one of the assignments for the Big data module

# Task 1: Analysis of Clinical Trial Data Using Spark SQL

#  **Table of Contents**

### **1. Load Data**
- 1.1 Header Inspection  
- 1.2 Load the Dataset Properly with Schema

### **2. Exploratory Data Analysis (EDA)**
- 2.1 Visual Inspection of DataFrame  
- 2.2 Schema Inspection  
- 2.3 Summary Statistics  
- 2.4 Dataset Dimensions  
- 2.5 Missing Values Count per Column  
- 2.6 Value Counts of Key Categorical Fields  
- 2.7 Date Normalization and Overwriting Original Columns  
- 2.8 Identifying Rows Missing All Required Fields  

### **3. Registering Cleaned Data for SQL Queries**

---

### **4. SQL-Based Analytical Tasks**

####  **Question 1 — Most Frequent Study Types**
- Step 1: Create a Filtered View of Study Types with Frequency ≥ 8  
- Step 2: Total Count of Included Records
- Step 3: Display the Frequency Table 

####  **Question 2 — Top 10 Most Frequent Medical Conditions**
- Step 1: Split and Normalize Conditions  
- Step 2: Count Total Number of Contributing Condition Records 
- Step 3: Rank Top 10 Most Common Conditions 

####  **Question 3 — Mean Duration of Clinical Trials**
- Step 1: Filter for Valid Dates and Calculate Duration  
- Step 2: Count Contributing Records  
- Step 3: Compute Mean Duration in Months  

####  **Question 4 — Completed Diabetes-Related Trials by Year**
- Step 1: Filter and Extract Completion Year 
- Step 2: Count Contributing Records  
- Step 3: Count Per Year to Analyze Trend
