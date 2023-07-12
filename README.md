# Analisis Data Kuesioner: Pengaruh Harga Tiket, Kualitas Pelayanan, dan Sistem Pembelian Tiket Online terhadap Keputusan Pembelian Tiket Kereta Api Ekonomi Jurusan Bojonegoro-Surabaya Pasar Turi

Pada proyek ini, kami menganalisis data dari kuesioner yang berisi pertanyaan tentang harga tiket, kualitas layanan, dan sistem pembelian tiket online, serta keputusan pembelian tiket kereta api. Setiap pertanyaan di kuesioner dinilai pada skala dari 1 hingga 5, dengan 1 berarti "tidak setuju sama sekali" dan 5 berarti "sangat setuju".

Berikut adalah langkah-langkah yang kami ambil dalam analisis ini dan hasilnya:

## Uji Validitas

```python
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
```

Hasil:

Kita melihat korelasi antara setiap pertanyaan dengan total skor.

| Pertanyaan | Korelasi dengan Total Skor | P-value |
|---|---|---|
| Harga 1 | 0.682 | <0.001 |
| Harga 2 | 0.576 | <0.001 |
| Harga 3 | 0.442 | <0.001 |
| Harga 4 | 0.440 | <0.001 |
| Harga 5 | 0.540 | <0.001 |
| Kualitas 1 | 0.287 | 0.003 |
| Kualitas 2 | 0.601 | <0.001 |
| Kualitas 3 | 0.696 | <0.001 |
| Kualitas 4 | 0.691 | <0.001 |
| Kualitas 5 | 0.608 | <0.001 |
| Sistem 1 | 0.621 | <0.001 |
| Sistem 2 | 0.627 | <0.001 |
| Sistem 3 | 0.693 | <0.001 |
| Sistem 4 | 0.661 | <0.001 |
| Keputusan 1 | 0.533 | <0.001 |
| Keputusan 2 | 0.624 | <0.001 |
| Keputusan 3 | 0.718 | <0.001 |
| Keputusan 4 | 0.617 | <0.001 |
| Keputusan 5 | 0.616 | <0.001 |

Kita melihat korelasi antara setiap pertanyaan dengan total skor.

## Uji Reliabilitas

```python
# Test reliability using Cronbach's alpha
def cronbach_alpha(df):
    df_corr = df.corr()
    N = df.shape[1]
    rs = np.array(df_corr)
    mean_r = np.mean(df_corr.values)
    return (N * mean_r) / (1 + (N - 1) * mean_r)

reliability = cronbach_alpha(flat_data)
```

Hasil: 0.912

Kita menggunakan koefisien alpha Cronbach dan hasilnya adalah 0.912. Karena ini lebih besar dari 0.7, kita dapat menganggap kuesioner tersebut reliabel.

## Uji Normalitas

```python
# Test normality using Kolmogorov-Smirnov test
normality_results = pd.DataFrame(index=flat_data.columns, columns=["Statistic", "P-value"])
for column in flat_data.columns:
    statistic, p_value = stats.kstest(flat_data[column], 'norm')
    normality_results.loc[column, "Statistic"] = statistic
    normality_results.loc[column, "P-value"] = p_value
```

Hasil:

| Pertanyaan | Statistik | P-value |
|---|---|---|
| Harga 1 | 0.979 | <0.001 |
| Harga 2 | 0.979 | <0.001 |
| Harga 3 | 0.957 | <0.001 |
| Harga 4 | 0.999 | <0.001 |
| Harga 5 | 0.989 | <0.001 |
| Kualitas 1 | 0.999 | <0.001 |
| Kualitas 2 | 0.999 | <0.001 |
| Kualitas 3 | 0.999 | <0.001 |
| Kualitas 4 | 0.999 | <0.001 |
| Kualitas 5 | 0.999 | <0.001 |
| Sistem 1 | 0.999 | <0.001 |
| Sistem 2 | 0.999 | <0.001 |
| Sistem 3 | 0.999 | <0.001 |
| Sistem 4 | 0.999 | <0.001 |
| Keputusan 1 | 0.999 | <0.001 |
| Keputusan 2 | 0.999 | <0.001 |
| Keputusan 3 | 0.999 | <0.001 |
| Keputusan 4 | 0.999 | <0.001 |
| Keputusan 5 | 0.999 | <0.001 |

Nilai p yang sangat kecil (<0.001) menunjukkan bahwa kita harus menolak hipotesis nol bahwa data mengikuti distribusi normal. Ini berarti data tidak normal.

## Uji Multikolinearitas

```python
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
```

Hasil:

Kita menggunakan Variance Inflation Factor (VIF) untuk melakukan ini. Biasanya, nilai VIF yang lebih besar dari 5 menunjukkan adanya multikolinearitas.

| Fitur | VIF |
|---|---|
| Harga 1 | 68.87 |
| Harga 2 | 43.18 |
| Harga 3 | 20.55 |
| Harga 4 | 49.46 |
| Harga 5 | 50.75 |


| Kualitas 1 | 45.14 |
| Kualitas 2 | 91.55 |
| Kualitas 3 | 100.30 |
| Kualitas 4 | 116.17 |
| Kualitas 5 | 75.52 |
| Sistem 1 | 71.37 |
| Sistem 2 | 66.41 |
| Sistem 3 | 98.48 |
| Sistem 4 | 130.62 |

Hasilnya menunjukkan bahwa semua variabel memiliki nilai VIF yang lebih besar dari 5, yang menunjukkan adanya multikolinearitas.

## Uji Heteroskedastisitas

```python
# Test heteroskedasticity using Goldfeld-Quandt test
gq_test = het_goldfeldquandt(y, X)
```

Hasil: F-statistik = 0.65, p-value = 0.90

Kita menggunakan uji Goldfeld-Quandt untuk melakukan ini. Hipotesis nol dalam tes ini adalah bahwa varians konstan (homoskedastisitas). Hasil tes (F-statistik = 0.65, p-value = 0.90) menunjukkan bahwa kita tidak dapat menolak hipotesis nol, yang berarti tidak ada bukti kuat adanya heteroskedastisitas.

## Uji Hipotesis Regresi Linier Berganda, Uji T, Uji F, dan Koefisien Determinasi

```python
# Print out the regression results
results.summary()
```
```
Model:                            OLS   Adj. R-squared:                  0.964
Method:                 Least Squares   F-statistic:                     190.8
Date:                Wed, 12 Jul 2023   Prob (F-statistic):           6.64e-58
Time:                        11:08:15   Log-Likelihood:                -169.15
No. Observations:                 100   AIC:                             368.3
Df Residuals:                      85   BIC:                             407.4
Df Model:                          14                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.8942      1.550      3.158      0.002       1.813       7.975
Harga 1        1.5077      0.290      5.199      0.000       0.931       2.084
Harga 2        0.9103      0.221      4.116      0.000       0.471       1.350
Harga 3        1.4766      0.181      8.143      0.000       1.116       1.837
Harga 4        0.9916      0.253      3.918      0.000       0.488       1.495
Harga 5        0.9655      0.243      3.978      0.000       0.483       1.448
Kualitas 1     0.6903      0.246      2.807      0.006       0.201       1.179
Kualitas 2     0.6240      0.319      1.954      0.054      -0.011       1.259
Kualitas 3     1.2786      0.339      3.771      0.000       0.604       1.953
Kualitas 4     1.4799      0.351      4.221      0.000       0.783       2.177
Kualitas 5     1.5723      0.299      5.257      0.000       0.978       2.167
Sistem 1       1.4397      0.274      5.252      0.000       0.895       1.985
Sistem 2       1.8822      0.276      6.822      0.000       1.334       2.431
Sistem 3       1.8335      0.318      5.772      0.000       1.202       2.465
Sistem 4       1.2445      0.373      3.338      0.001       0.503       1.986
==============================================================================
Omnibus:                        8.118   Durbin-Watson:                   2.187
Prob(Omnibus):                  0.017   Jarque-Bera (JB):               13.516
Skew:                          -0.260   Prob(JB):                      0.00116
Kurtosis:                       4.724   Cond. No.                         171.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
```

Hasil:

- Prob (F-statistic) adalah < 0.001, yang berarti model regresi secara keseluruhan signifikan.
- Semua variabel kecuali "Kualitas 2" memiliki p-value < 0.05, yang berarti mereka semua signifikan.
- F-statistik adalah 190.8 dan Prob (F-statistic) adalah < 0.001. Ini berarti bahwa model secara keseluruhan signifikan.
- R-squared adalah 0.969, yang berarti bahwa model ini menjelaskan 96.9% variabilitas dalam total skor. Ini adalah penyesuaian yang sangat baik.

## Ringkasan

Dalam konteks keputusan pembelian tiket kereta api, hasil ini menunjukkan bahwa semua variabel ("Harga", "Kualitas", dan "Sistem") memiliki pengaruh yang signifikan, dengan "Harga" dan "Sistem" tampaknya memiliki pengaruh yang paling kuat, berdasarkan nilai koefisien regresi. Namun, penting untuk dicatat bahwa adanya multikolinearitas dapat mempengaruhi interpretasi ini, karena koefisien regresi mungkin terdistorsi oleh korelasi antara variabel independen.

Penting juga untuk mencatat bahwa model ini dibangun berdasarkan data yang tidak normal dan ini dapat mempengaruhi keandalan hasil. Selain itu, model ini menjelaskan sebagian besar variabilitas dalam data (96.9%), tetapi masih ada 3.1% variabilitas yang belum dijelaskan. Ini bisa disebabkan oleh faktor lain yang tidak termasuk dalam model ini.
