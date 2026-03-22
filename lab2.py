import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- BÀI 1: HOUSING DATA ---
print("--- ĐANG CHẠY BÀI 1: HOUSING ---")
df1 = pd.DataFrame({
    'Area': np.random.normal(70, 15, 100), 
    'Price': np.random.normal(3000, 800, 100)
})
df1.loc[98], df1.loc[99] = [500, 15000], [15, 9000] # Tạo điểm ngoại lệ

print(df1.describe())
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); sns.boxplot(data=df1); plt.title("1.3 Boxplot Raw")
plt.subplot(1, 3, 2); sns.scatterplot(data=df1, x='Area', y='Price'); plt.title("1.4 Scatter Raw")

# Xử lý bằng Clip (giới hạn 10% - 90%)
df1_fix = df1.clip(lower=df1.quantile(0.1), upper=df1.quantile(0.9), axis=1)
plt.subplot(1, 3, 3); sns.boxplot(data=df1_fix); plt.title("1.10 After Clipping")
plt.tight_layout(); plt.show()

# --- BÀI 2: IOT / SENSOR ---
print("\n--- ĐANG CHẠY BÀI 2: IOT ---")
t = pd.date_range('2026-01-01', periods=100, freq='H')
temp = np.random.normal(25, 2, 100)
temp[30], temp[70] = 45, 10 # Lỗi sensor
df2 = pd.DataFrame({'Temperature': temp}, index=t)

# Rolling mean +/- 3*std (Phát hiện biến động nhanh)
rm = df2['Temperature'].rolling(window=10).mean()
rs = df2['Temperature'].rolling(window=10).std()
outliers = (df2['Temperature'] > rm + 3*rs) | (df2['Temperature'] < rm - 3*rs)

plt.figure(figsize=(10, 4))
plt.plot(df2.index, df2['Temperature'], label='Actual Temp')
plt.scatter(df2.index[outliers], df2['Temperature'][outliers], color='red', label='Outliers')
plt.title("2.3 Rolling Outlier Detection")
plt.legend(); plt.show()

# --- BÀI 3: E-COMMERCE ---
print("\n--- ĐANG CHẠY BÀI 3: E-COMMERCE ---")
df3 = pd.DataFrame({
    'Price': [15, 20, 18, 2000, 22, 19, 21],
    'Qty': [1, 2, 1, 1, 2, 80, 2],
    'Rating': [4, 5, 4, 3, 5, 12, 4]
})
print("Dữ liệu gốc:\n", df3)

# Xử lý: Xóa Rating lỗi (>5), dùng Log cho Price
df3_fix = df3[df3['Rating'] <= 5].copy()
df3_fix['Price'] = np.log1p(df3_fix['Price'])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); sns.boxplot(data=df3); plt.title("3.2 Raw Data")
plt.subplot(1, 2, 2); sns.boxplot(data=df3_fix); plt.title("3.7 Cleaned Data")
plt.show()

# --- BÀI 4: MULTIVARIATE ---
print("\n--- ĐANG CHẠY BÀI 4: MULTIVARIATE ---")
x, y = np.random.normal(50, 10, 50), np.random.normal(50, 10, 50)
x = np.append(x, [100]); y = np.append(y, [100]) # Ngoại lệ đa biến
df4 = pd.DataFrame({'Var1': x, 'Var2': y})

# Tính Z-score để highlight màu
z4 = (np.abs(stats.zscore(df4)) > 2).any(axis=1)

plt.figure(figsize=(7, 5))
# ĐÃ FIX: Thêm data=df4 vào lệnh vẽ
sns.scatterplot(data=df4, x='Var1', y='Var2', hue=z4, palette={True:'red', False:'blue'}, s=100)
plt.title("4.3 Multivariate Outliers (Red)")
plt.show()

print("\n--- TẤT CẢ HOÀN TẤT ---")