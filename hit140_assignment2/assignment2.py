import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
merged_df = pd.merge(dataset1, dataset2, on='ID')
merged_df = pd.merge(merged_df, dataset3, on='ID')


#____________________________________investigation-1__________________________________________

# Average screen time by gender
gender_screen_time = merged_df.groupby('gender')[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean()

# Average screen time by minority status
minority_screen_time = merged_df.groupby('minority')[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean()

# Average screen time by deprivation level
deprived_screen_time = merged_df.groupby('deprived')[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean()

# Display results
print(gender_screen_time)
print(minority_screen_time)
print(deprived_screen_time)



# Bar chart for average screen time by gender
gender_screen_time.plot(kind='bar', figsize=(10, 6))
plt.title('Average Screen Time by Gender')
plt.xlabel('Gender')
plt.ylabel('Hours of Screen Time')
plt.show()

# Bar chart for average screen time by minority
minority_screen_time.plot(kind='bar', figsize=(10, 6))
plt.title('Average Screen Time by minority')
plt.xlabel('minority')
plt.ylabel('Hours of Screen Time')
plt.show()

# Bar chart for average screen time by deprived
deprived_screen_time.plot(kind='bar', figsize=(10, 6))
plt.title('Average Screen Time by deprived')
plt.xlabel('deprived')
plt.ylabel('Hours of Screen Time')
plt.show()

#________________________________________Investigation-2_____________________________________________


# Define well-being columns
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                      'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Plot histograms for well-being indicators
merged_df[well_being_columns].hist(bins=5, figsize=(15, 10))
plt.tight_layout()
plt.show()


#__________________________________________Investigation-3_________________________________________________





# Calculate total weekday screen time
merged_df['weekday_screen_time'] = merged_df[['C_wk', 'G_wk', 'S_wk', 'T_wk']].sum(axis=1)

# Calculate mean and standard error
mean_screen_time = merged_df['weekday_screen_time'].mean()
std_error = merged_df['weekday_screen_time'].sem()

# Calculate confidence interval of weekdays screen time
confidence_level = 0.95
degrees_freedom = len(merged_df) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * std_error
#calculate ci_lower and ci_upper
ci_lower = mean_screen_time - margin_of_error
ci_upper = mean_screen_time + margin_of_error

print(f"Mean weekday screen time: {mean_screen_time:.2f} hours")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f}) hours")

print("-----------------------------------------------------------------------------------------------")

# Calculate total weekend screen time
merged_df['weekend_screen_time'] = merged_df[['C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)

# Calculate mean and standard error of weekend screen time
mean_screen_time = merged_df['weekend_screen_time'].mean()
std_error = merged_df['weekend_screen_time'].sem()

# Calculate confidence interval for weekend screen time
confidence_level = 0.95
degrees_freedom = len(merged_df) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * std_error
#calculate ci_lower and ci_upper 
ci_lower = mean_screen_time - margin_of_error
ci_upper = mean_screen_time + margin_of_error

print(f"Mean weekend screen time: {mean_screen_time:.2f} hours")
print(f"95% Confidence Interval for weekend screen time: ({ci_lower:.2f}, {ci_upper:.2f}) hours")

#__________________________________________Investigation-4_________________________________________________

# Step 1: Hypothesis testing - Is there a significant effect of screen time reduction on well-being?

# Group respondents based on their screen time
low_screen_time = merged_df[merged_df['C_we'] <= 2]  # Less screen time on weekends
high_screen_time = merged_df[merged_df['C_we'] > 2]  # More screen time on weekends

# Perform t-test on well-being scores between low and high screen time groups for 'Optm' (Optimism)
t_stat, p_value = stats.ttest_ind(low_screen_time['Optm'], high_screen_time['Optm'])

print(f"\nT-test for 'Optm' (Optimism): t-stat = {t_stat}, p-value = {p_value}")

# Interpret the result
if p_value < 0.05:
    print("The result is statistically significant; reducing screen time significantly affects well-being.")
else:
    print("The result is not statistically significant; screen time reduction may not affect well-being.")