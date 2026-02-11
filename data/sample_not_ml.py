"""
Regular data processing script - NOT ML
"""
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('sales_data.csv')

# Data cleaning
data['date'] = pd.to_datetime(data['date'])
data = data.dropna()

# Calculate statistics
monthly_avg = data.groupby(data['date'].dt.month)['revenue'].mean()
total_revenue = data['revenue'].sum()

# Generate report
report = {
    'total_revenue': total_revenue,
    'monthly_averages': monthly_avg.to_dict(),
    'record_count': len(data)
}

print("Sales Report:")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Monthly: ${monthly_avg.mean():,.2f}")