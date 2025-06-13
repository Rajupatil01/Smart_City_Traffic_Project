import pandas as pd
import mysql.connector

# Step 1: Load the CSV file
csv_file = 'traffic_data_10k.csv'  # Or provide full path if needed
df = pd.read_csv(csv_file)

# Step 2: Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',              # replace with your username
    password='system1234', # replace with your password
    database='traffic_analysis'  # make sure this DB already exists
)

cursor = conn.cursor()

# Step 3: Create table (if not already created)
create_table_query = """
CREATE TABLE IF NOT EXISTS traffic_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    location VARCHAR(100),
    car_count INT,
    truck_count INT,
    bus_count INT,
    weather VARCHAR(50),
    accident_reported TINYINT
)
"""
cursor.execute(create_table_query)
conn.commit()

# Step 4: Insert CSV rows into MySQL
insert_query = """
INSERT INTO traffic_data 
(timestamp, location, car_count, truck_count, bus_count, weather, accident_reported)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

data = [
    (
        row['timestamp'],
        row['location'],
        int(row['car_count']),
        int(row['truck_count']),
        int(row['bus_count']),
        row['weather'],
        int(row['accident_reported'])
    )
    for index, row in df.iterrows()
]

cursor.executemany(insert_query, data)
conn.commit()

print(f"✅ Successfully inserted {cursor.rowcount} rows into traffic_data.")

# Close connection
cursor.close()
conn.close()


import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Connect to MySQL and fetch data
conn = mysql.connector.connect(
    host='localhost',
    user='root',              # ← change this
    password='system1234', # ← change this
    database='traffic_analysis'
)

query = "SELECT * FROM traffic_data"
df = pd.read_sql(query, conn)
conn.close()

# Step 2: Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day_name()

# Step 3: EDA Plots

## 1. Peak vs Off-Peak Hours
hourly_traffic = df.groupby('hour')[['car_count', 'truck_count', 'bus_count']].sum()
hourly_traffic.plot(kind='bar', stacked=True, figsize=(12,6), colormap='viridis')
plt.title('Traffic Volume by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Total Vehicles')
plt.grid(True)
plt.tight_layout()
plt.show()

## 2. Most Congested Locations
df['total_vehicles'] = df['car_count'] + df['truck_count'] + df['bus_count']
top_locations = df.groupby('location')['total_vehicles'].sum().sort_values(ascending=False)
top_locations.plot(kind='bar', figsize=(10,5), color='orange')
plt.title('Most Congested Locations')
plt.ylabel('Total Vehicle Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 3. Weather Impact on Traffic
weather_traffic = df.groupby('weather')['total_vehicles'].mean().sort_values(ascending=False)
weather_traffic.plot(kind='bar', figsize=(8,5), color='green')
plt.title('Average Traffic by Weather Condition')
plt.ylabel('Average Vehicles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 4. Accident Count by Hour
accidents_by_hour = df[df['accident_reported'] == 1].groupby('hour').size()
accidents_by_hour.plot(kind='bar', color='red', figsize=(10,5))
plt.title('Accidents by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

## 5. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['car_count', 'truck_count', 'bus_count', 'accident_reported']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import mysql.connector

# Step 1: Load data from MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='system1234',
    database='traffic_analysis'
)
df = pd.read_sql("SELECT * FROM traffic_data", conn)
conn.close()

# Step 2: Preprocess Data
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['total_vehicles'] = df['car_count'] + df['truck_count'] + df['bus_count']

# Convert categorical column 'weather' into numeric using one-hot encoding
df = pd.get_dummies(df, columns=['weather'], drop_first=True)

# Drop unneeded columns
df = df.drop(columns=['timestamp', 'location', 'id'])

# ----------- 🔹 Model 1: Linear Regression - Predict Traffic Volume --------------
X1 = df.drop(columns=['total_vehicles', 'accident_reported'])
y1 = df['total_vehicles']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X1_train, y1_train)

y1_pred = lr_model.predict(X1_test)
mse = mean_squared_error(y1_test, y1_pred)
print("🔹 Linear Regression (Traffic Prediction)")
print(f"Mean Squared Error: {mse:.2f}\n")

# ----------- 🔸 Model 2: Logistic Regression - Predict Accident Risk -------------
X2 = df.drop(columns=['accident_reported', 'total_vehicles'])
y2 = df['accident_reported']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X2_train, y2_train)

y2_pred = log_model.predict(X2_test)
acc = accuracy_score(y2_test, y2_pred)
print("🔸 Logistic Regression (Accident Risk Prediction)")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y2_test, y2_pred))
