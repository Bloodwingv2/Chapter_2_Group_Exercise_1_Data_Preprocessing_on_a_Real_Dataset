# Chapter 2 Group Exercise 1: Data Preprocessing on a Real Dataset

## 📋 Overview

This project applies comprehensive data preprocessing techniques to a real-world dataset—**New York City Airbnb Open Data** from Kaggle. The exercise demonstrates practical skills in handling missing values, scaling features, managing noise, detecting outliers, and performing feature selection using scikit-learn and pandas.

---

## 👥 Group Members

| Name | Matriculation Number | Role |
|------|----------------------|------|
| Arya Shinde | 100006646 | Group Co-Ordinator |
| Yash Annapure | 100006547 | Member |
| Mirang Bhandari | 100007049 | Member |
| Anushka Sawant | 100006644 | Member |

---

## 🎯 Project Objectives

Apply data preprocessing techniques learned in Chapter 2 to a real-world dataset. The goal is to understand how raw data is prepared before applying machine learning models.

### Key Learning Outcomes:
- Handle missing values using appropriate imputation strategies
- Apply feature scaling techniques (Z-score standardization and Min-Max normalization)
- Manage noise through smoothing techniques
- Detect and handle outliers
- Perform feature selection using statistical methods

---

## 📊 Dataset Information

**Dataset Name:** New York City Airbnb Open Data 2019

**Source:** [Kaggle - New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

**Description:**
This dataset contains Airbnb listing activity and related metrics for New York City in 2019, including:
- Host information and identification
- Location details (latitude, longitude, neighborhood)
- Pricing information
- Availability metrics
- Review statistics

The dataset includes ~49,000 records with substantial missing values and potential outliers, making it ideal for data preprocessing exercises.

**Dataset Characteristics:**
- **Records:** ~49,000
- **Features:** Multiple numerical and categorical
- **Missing Values:** Present in names, host_names, last_review, and reviews_per_month
- **Outliers:** Present in price feature (luxury properties vs. budget rooms)

---

## 📁 Project Structure

```
Chapter_2_Group_Exercise_1_Data_Preprocessing_on_a_Real_Dataset/
├── README.md                          # This file
├── pyproject.toml                     # Project configuration and dependencies
├── .gitignore                         # Git ignore rules
├── Dataset/
│   └── AB_NYC_2019.csv               # Airbnb NYC 2019 dataset
└── Jupyter Notebook/
    └── Chapter_2_Group_Exercise_1_Data_Preprocessing_on_a_Real_Dataset.ipynb
```

---

## 🚀 Environment Setup

### Prerequisites
- Python 3.13 or higher
- `uv` package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Installation & Virtual Environment Setup with `uv`

The project uses `uv` for fast, reliable Python package management.

#### Step 1: Install uv (if not already installed)
```bash
# On Windows (using PowerShell as admin)
powershell -ExecutionPolicy BypassCurrentUser -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 2: Navigate to the project directory
```bash
cd "e:\Master's Notes\1st Semester\Statistics and Machine Learning\Chapter_2_Group_Exercise_1_Data_Preprocessing_on_a_Real_Dataset"
```

#### Step 3: Create and activate virtual environment with uv
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

#### Step 4: Install dependencies with uv sync
```bash
# Synchronize and install all project dependencies
uv sync
```

This command will:
- Install all dependencies specified in `pyproject.toml`
- Create a consistent environment matching the project requirements
- Install the required packages:
  - `matplotlib>=3.10.8` - Data visualization
  - `pandas>=3.0.0` - Data manipulation
  - `scikit-learn>=1.8.0` - Machine learning and preprocessing
  - `scipy>=1.17.0` - Statistical functions
  - `seaborn>=0.13.2` - Advanced data visualization

---

## 📖 Dependencies

All project dependencies are defined in [pyproject.toml](pyproject.toml):

```toml
[project]
name = "chapter-2-group-exercise-1-data-preprocessing-on-a-real-dataset"
version = "0.1.0"
description = "Data preprocessing on NYC Airbnb dataset"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "matplotlib>=3.10.8",  # Plotting and visualization
    "pandas>=3.0.0",       # Data manipulation and analysis
    "scikit-learn>=1.8.0", # Machine learning preprocessing tools
    "scipy>=1.17.0",       # Scientific computing
    "seaborn>=0.13.2",     # Statistical data visualization
]
```

---

## 📊 Tasks & Implementation

### Task 1: Handling Missing Values
**Objective:** Identify and impute missing values appropriately

**Implementation:**
- Dropped 16 missing `names` and 21 missing `host_names` (~0.03% of data)
- Filled `reviews_per_month` with 0 for listings without reviews
- Kept `last_review` as NaN for listings without reviews (maintains data integrity)

**Code Location:** Cells 3-7 in notebook

---

### Task 2: Scaling Numerical Features
**Objective:** Apply standardization and normalization techniques

**Numerical Features Scaled:**
- latitude, longitude
- price
- minimum_nights
- number_of_reviews
- reviews_per_month
- calculated_host_listings_count
- availability_365

**Methods Applied:**

#### 2.1 Z-score Standardization
- Centers data around mean = 0
- Scales based on standard deviation
- Formula: $z = \frac{x - \mu}{\sigma}$
- Use case: When features have different units or scales

#### 2.2 Min-Max Normalization
- Scales data to fixed range [0, 1]
- Preserves original distribution shape
- Formula: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- Use case: When you need bounded features

**Key Difference:**
- **Z-score:** Suitable for algorithms assuming normally distributed data (linear regression, logistic regression)
- **Min-Max:** Suitable for algorithms sensitive to feature magnitude (neural networks, distance-based methods)

**Code Location:** Cells 9-12 in notebook

---

### Task 3: Handling Noise
**Objective:** Demonstrate noise injection and smoothing techniques

**Implementation:**
- Selected `price` feature from first 100 records
- Injected artificial Gaussian noise (mean=0, std=20)
- Applied rolling mean smoothing (window size=5)

**Before & After Statistics:**
| Metric | Original | Noisy | Smoothed |
|--------|----------|-------|----------|
| Mean | Original mean | Increased variance | Approximates original |
| Std Dev | Original std | Increased | Reduced |

**Visualization:** Dual plots showing original vs. noisy and noisy vs. smoothed data

**Code Location:** Cells 13-14 in notebook

---

### Task 4: Handling Outliers
**Objective:** Detect and remove outliers using statistical methods

**Method:** Z-score based detection
- Threshold: |Z-score| > 3 (captures 99.7% of normal distribution)
- NYC Airbnb context: Luxury penthouses vs. shared rooms create natural outliers

**Results:**
- Detected outliers in price feature
- Removed outliers using threshold method
- Preserved data integrity with justified approach

**Justification:**
Removing extreme price values helps:
- Improve model generalization
- Reduce bias from luxury properties
- Focus analysis on typical market conditions

**Visualization:** Box plots comparing original and cleaned price distribution

**Code Location:** Cells 15-16 in notebook

---

### Task 5: Feature Selection
**Objective:** Identify most relevant features for target variable

**Method: Filter Method (Correlation Matrix)**

**Rationale:**
- Computationally efficient
- Clear visual interpretation
- Identifies feature relationships and redundancy
- Reveals correlation strength with target variable

**Key Findings:**
- Calculated Pearson correlation matrix for all numerical features
- Identified features with strongest relationship to price
- Detected potential multicollinearity
- Visualized with heatmap

**Code Location:** Cells 17-19 in notebook

---

## 🔧 Running the Jupyter Notebook

### Option 1: Using VS Code
1. Ensure virtual environment is activated (`.venv\Scripts\Activate.ps1` on Windows)
2. Open the notebook: `Jupyter Notebook/Chapter_2_Group_Exercise_1_Data_Preprocessing_on_a_Real_Dataset.ipynb`
3. VS Code will detect the virtual environment automatically
4. Run cells sequentially using Shift+Enter

### Option 2: Using Jupyter Lab/Notebook
```bash
# Activate virtual environment first
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # macOS/Linux

# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

### Running All Cells
To execute the entire notebook:
1. Press `Ctrl+Shift+Enter` in VS Code
2. Or select "Run All" from the notebook menu

---

## 📈 Expected Output

When running the notebook, you'll see:

1. **Data Summary**
   - Dataset shape and basic statistics
   - Missing value counts and percentages

2. **Missing Value Handling**
   - Before/after missing value counts
   - Number of rows dropped

3. **Feature Scaling Results**
   - Z-scored features with mean ≈ 0, std ≈ 1
   - Min-max scaled features with range [0, 1]

4. **Noise Handling Visualization**
   - Dual subplots comparing original, noisy, and smoothed data
   - Statistical comparison table

5. **Outlier Detection**
   - Number and percentage of outliers
   - Box plots showing before/after cleaning

6. **Feature Correlation**
   - Heatmap showing correlation matrix
   - Feature correlations with price sorted

---

## 🐛 Troubleshooting

### Issue: `uv` command not found
**Solution:** Ensure `uv` is installed and added to PATH. Restart terminal after installation.

### Issue: Virtual environment not activating
**Solution:** Try running with full path:
```bash
& ".\.venv\Scripts\Activate.ps1"  # PowerShell
.venv\Scripts\activate  # Command Prompt
```

### Issue: Dataset file not found
**Solution:** Ensure you're running the notebook from the correct directory. The dataset path is relative: `../Dataset/AB_NYC_2019.csv`

### Issue: Missing dependencies
**Solution:** Run `uv sync` again to ensure all packages are installed

---

## 📚 Concepts Covered

- **Data Exploration:** Understanding dataset structure and characteristics
- **Missing Value Imputation:** Strategic approaches based on data type
- **Feature Scaling:** Standardization vs. normalization tradeoffs
- **Noise Handling:** Smoothing techniques and their effects
- **Outlier Detection:** Statistical methods for anomaly detection
- **Feature Engineering:** Selection methods and correlation analysis
- **Data Visualization:** Using matplotlib and seaborn for insights

---

## 🎓 References

- [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## 📄 License

This is an academic exercise for educational purposes.

---

**Last Updated:** January 2026
**Python Version:** 3.13+
**Project Status:** Complete
Embedded method (e.g. Lasso)
Explain why this method was chosen.