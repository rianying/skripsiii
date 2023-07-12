
# Import required libraries
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.api import het_goldfeldquandt
import pandas as pd

# Load the data
data = pd.read_csv('/mnt/data/skripsi.csv')

# Define the independent and dependent variables
independent_vars = ['Harga 1', 'Harga 2', 'Harga 3', 'Harga 4', 'Harga 5', 'Kualitas 1', 'Kualitas 2', 'Kualitas 3', 'Kualitas 4', 'Kualitas 5', 'Sistem 1', 'Sistem 2', 'Sistem 3', 'Sistem 4']
dependent_vars = ['Keputusan 1', 'Keputusan 2', 'Keputusan 3', 'Keputusan 4', 'Keputusan 5']

# Flatten the data for correlation and reliability tests
flat_data = data[independent_vars + dependent_vars]

# Test validity by checking the correlation of each item with the total score
validity_results = pd.DataFrame(index=flat_data.columns, columns=["Correlation", "P-value"])
for column in flat_data.columns:
    correlation, p_value = pearsonr(flat_data[column], data["Total Skor"])
    validity_results.loc[column, "Correlation"] = correlation
    validity_results.loc[column, "P-value"] = p_value

# Test reliability using Cronbach's alpha
def cronbach_alpha(df):
    df_corr = df.corr()
    N = df.shape[1]
    rs = np.array(df_corr)
    mean_r = np.mean(df_corr.values)
    return (N * mean_r) / (1 + (N - 1) * mean_r)

reliability = cronbach_alpha(flat_data)

# Test normality using Kolmogorov-Smirnov test
normality_results = pd.DataFrame(index=flat_data.columns, columns=["Statistic", "P-value"])
for column in flat_data.columns:
    statistic, p_value = stats.kstest(flat_data[column], 'norm')
    normality_results.loc[column, "Statistic"] = statistic
    normality_results.loc[column, "P-value"] = p_value

# Define the independent and dependent variables for regression
X = data[independent_vars]
y = data['Total Skor']

# Add a constant to the independent value
X1 = sm.add_constant(X)

# Carry out the multiple linear regression
model = sm.OLS(y, X1)
results = model.fit()

# Test multicollinearity using Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Test heteroskedasticity using Goldfeld-Quandt test
gq_test = het_goldfeldquandt(y, X)

print(validity_results)
print("Reliability: ", reliability)
print(normality_results)
print(vif_data)
print("Goldfeld-Quandt test: ", gq_test)
print(results.summary())
