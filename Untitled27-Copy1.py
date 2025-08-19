#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load your file
df = pd.read_csv("latest data.csv")

# Drop irrelevant columns (keep only Likert-scale ones)
likert_df = df.select_dtypes(include='number')

# Calculate average sentiment score per question
avg_scores = likert_df.mean()

# Display the result
print(avg_scores)


# In[2]:


from pingouin import cronbach_alpha
perceived_usefulness = df.iloc[:, 10:15]
perceived_ease_of_use = df.iloc[:, 16:21]
pricing_and_promotion = df.iloc[:, 22:27]
service_quality = df.iloc[:, 28:33]
consumer_purchasing_behavior = df.iloc[:, 34:39]
alpha_perceived_usefulness = cronbach_alpha(perceived_usefulness)
alpha_perceived_ease_of_use = cronbach_alpha(perceived_ease_of_use)
alpha_pricing_and_promotion = cronbach_alpha(pricing_and_promotion)
alpha_service_quality = cronbach_alpha(service_quality)
alpha_consumer_purchasing_behavior = cronbach_alpha(consumer_purchasing_behavior)
print("Cronbach's Alpha:")
print(f"Perceived Usefulness: {alpha_perceived_usefulness}")
print(f"Perceived Ease of Use: {alpha_perceived_ease_of_use}")
print(f"Pricing and Promotion: {alpha_pricing_and_promotion}")
print(f"Service Quality: {alpha_service_quality}")
print(f"Consumer Purchasing Behavior: {alpha_consumer_purchasing_behavior}")



# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set the plotting style
sns.set(style="whitegrid")

# Bar chart: Age Group vs Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Age Group", hue="Gender")
plt.title("Number of Respondents by Age Group and Gender")
plt.xlabel("Age Group")
plt.ylabel("Number of Respondents")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Bar chart: Age Group vs Education Background
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Age Group", hue="Education Background")
plt.title("Number of Respondents by Age Group and Education Background")
plt.xlabel("Age Group")
plt.ylabel("Number of Respondents")
plt.legend(title="Education")
plt.tight_layout()
plt.show()


# In[4]:


import matplotlib.pyplot as plt

# First, normalize education categories for consistency
df['Education Background'] = df['Education Background'].replace({
    "Master in Bachelor": "Master's Degree"
})

# Group and count: Age Group × Gender × Education
grouped = df.groupby(['Age Group', 'Gender', 'Education Background']).size().reset_index(name='Count')

# Pivot for stacked bar plot
pivot_df = grouped.pivot_table(index=['Age Group', 'Gender'], columns='Education Background', values='Count', fill_value=0)
pivot_df = pivot_df.sort_index()

# Plot
ax = pivot_df.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set3')
plt.title("Respondents by Age Group and Gender, Colored by Education Background")
plt.xlabel("Age Group and Gender")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Education Background', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace "Master in Bachelor" with a unified label
df['Education Background'] = df['Education Background'].replace({
    "Master in Bachelor": "Master's Degree"
})

# Group and count
grouped = df.groupby(['Age Group', 'Gender', 'Education Background']).size().reset_index(name='Count')
pivot_df = grouped.pivot_table(index=['Age Group', 'Gender'], columns='Education Background', values='Count', fill_value=0)
pivot_df = pivot_df.sort_index()

# Convert MultiIndex to string for plotting
index_labels = [f"{age} / {gender}" for age, gender in pivot_df.index]

# Plot manually with annotations
fig, ax = plt.subplots(figsize=(14, 7))
bottom = [0] * len(pivot_df)
colors = plt.cm.Set3.colors

for i, column in enumerate(pivot_df.columns):
    bars = ax.bar(index_labels, pivot_df[column], bottom=bottom, label=column, color=colors[i % len(colors)])
    
    for j, value in enumerate(pivot_df[column]):
        if value > 0:
            ax.text(j, bottom[j] + value / 2, str(value), ha='center', va='center', fontsize=8)
    
    bottom = [bottom[k] + pivot_df[column].iloc[k] for k in range(len(bottom))]

# Final touches
ax.set_title("Respondents by Age Group and Gender, Colored by Education Background", fontsize=14)
ax.set_xlabel("Age Group / Gender", fontsize=12)
ax.set_ylabel("Number of Respondents", fontsize=12)
ax.set_xticks(range(len(index_labels)))
ax.set_xticklabels(index_labels, rotation=45, ha='right')
ax.legend(title='Education Background', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have the DataFrame `df` loaded

# Create a cross-tabulation of State vs Most Preferred App
state_app_counts = pd.crosstab(df['State'], df['Most Preferred App'])

# Plotting
plt.figure(figsize=(14, 8))
state_app_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
plt.title('Most Preferred App by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Respondents', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Most Preferred App')
plt.tight_layout()
plt.show()



# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have the DataFrame `df` loaded

# Create a cross-tabulation of State vs Most Preferred App
state_app_counts = pd.crosstab(df['State'], df['Most Preferred App'])

# Plotting
ax = state_app_counts.plot(
    kind='bar', stacked=True, colormap='viridis', figsize=(14, 8)
)

plt.title('Most Preferred App by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Respondents', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Most Preferred App')

# Add counts on top of each bar section
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=8)

plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Set up plot style
sns.set(style="whitegrid")

# Plot using seaborn's catplot for grouped bar chart
plt.figure(figsize=(14, 7))
sns.countplot(
    data=df,
    x='Gender',
    hue='Education Background',
    palette='tab10',
    dodge=True,
    hue_order=sorted(df['Education Background'].dropna().unique()),
)

# Split by Age Group using facet if needed
sns.catplot(
    data=df,
    kind='count',
    x='Gender',
    hue='Education Background',
    col='Age Group',
    col_wrap=3,
    height=5,
    aspect=1,
    palette='tab10'
)

# Titles and labels
plt.suptitle('Age Against Gender & Highest Education', fontsize=16, y=1.03)
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Set plot style
sns.set(style="whitegrid")

# Count the number of respondents per state
state_counts = df['State'].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette='viridis')

# Labels and titles
plt.title('Distribution of Respondents by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Respondents', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[10]:





import pandas as pd
import plotly.graph_objects as go

# Load your CSV file
df = pd.read_csv("latest data.csv")
# Count respondents by state
state_counts = df['State'].value_counts().reset_index()
state_counts.columns = ['State', 'Respondent Count']

# Manual centroids for Malaysian states/federal territories
state_coords = {
    'Johor': (1.4854, 103.7618),
    'Kedah': (6.1184, 100.3685),
    'Kelantan': (6.1254, 102.2381),
    'Kuala Lumpur': (3.1390, 101.6869),
    'Malacca': (2.1896, 102.2501),
    'Negeri Sembilan': (2.7258, 101.9424),
    'Pahang': (3.8126, 103.3256),
    'Penang': (5.4164, 100.3327),
    'Perak': (4.5975, 101.0901),
    'Perlis': (6.4440, 100.2048),
    'Putrajaya': (2.9264, 101.6964),
    'Sabah': (5.9804, 116.0735),
    'Sarawak': (1.5533, 110.3592),
    'Selangor': (3.0738, 101.5183),
    'Terengganu': (5.3117, 103.1324),
}

# Add coordinates
state_counts['Latitude'] = state_counts['State'].map(lambda x: state_coords.get(x, (None, None))[0])
state_counts['Longitude'] = state_counts['State'].map(lambda x: state_coords.get(x, (None, None))[1])
state_counts.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Create the map with only text
fig = go.Figure(go.Scattergeo(
    lon=state_counts['Longitude'],
    lat=state_counts['Latitude'],
    text=state_counts['State'] + ': ' + state_counts['Respondent Count'].astype(str),
    mode='text',
    textposition='top center',
    textfont=dict(size=13, color='black', family="Arial"),
    showlegend=False
))

# Updated layout: zoom in tighter on Malaysia
fig.update_layout(
    geo=dict(
        projection_type="mercator",
        center=dict(lat=4.5, lon=107),  # Shift slightly east to capture East Malaysia
        showland=True,
        landcolor="lightgray",
        subunitcolor="black",
        coastlinecolor="black",
        lataxis=dict(range=[0.5, 7.5]),     # Tight vertical focus on Malaysia
        lonaxis=dict(range=[99, 119.5]),    # Tight horizontal focus
        resolution=50,
    ),
    title='Respondent Distribution by State in Malaysia',
    title_x=0.5,
    title_font=dict(size=16, color='black', family='Arial'),
    height=400,
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

fig.show()


# In[12]:


import pandas as pd

# Rename the relevant columns (if not already done)
df_renamed = df.rename(columns={
    ' Using online food delivery services helps me save time compared to dining out.  ': 'PU1',
    'Online food delivery services allow me to access a wide variety of restaurants conveniently. ': 'PU2',
    'I find online food delivery services helpful for fulfilling my food needs when I am busy. ': 'PU3',
    'The availability of food delivery services improves the quality of my daily routine. ': 'PU4',
    'Online food delivery platforms provide a practical alternative to cooking at home. ': 'PU5',
    'I feel that online food delivery services enhance my ability to plan meals more efficiently. ': 'PU6',

    'Online food delivery apps or websites are easy to navigate. ': 'PEOU1',
    'Placing an order through an online food delivery platform is straightforward. ': 'PEOU2',
    'It is easy for me to track my food delivery status through the app. ': 'PEOU3',
    'The registration or login process on online food delivery platforms is simple. ': 'PEOU4',
    'I find it easy to make payments through online food delivery platforms ': 'PEOU5',
    'Resolving issues (e.g., wrong orders) through the app’s customer support is hassle-free. ': 'PEOU6',

    'The prices offered by online food delivery platforms are reasonable. ': 'PP1',
    'Discounts and promotional offers influence my decision to use online food delivery services. ': 'PP2',
    'I am more likely to choose a platform that offers loyalty rewards or cashback.  ': 'PP3',
    'I feel that online food delivery platforms offer good value for the money I spend. ': 'PP4',
    'Price transparency (e.g., showing fees and charges) is important when I use online food delivery services. ': 'PP5',
    'I tend to compare prices across platforms before placing an order. ': 'PP6',

    'My food orders are consistently delivered on time. ': 'SQ1',
    'The quality of food delivered meets my expectations. ': 'SQ2',
    'The food I order is always packaged securely and hygienically. ': 'SQ3',
    'I am satisfied with the accuracy of the items I receive in my food orders. ': 'SQ4',
    'The customer support provided by the platform resolves my issues effectively and efficiently': 'SQ5',
    'I trust the platform to handle my orders professionally and reliably. ': 'SQ6',

    'I frequently use online food delivery services to purchase meals. ': 'CPB1',
    'I am likely to choose online food delivery services for my future meal orders.  ': 'CPB2',
    'I prefer online food delivery services over dining in or takeout options. ': 'CPB3',
    'Promotional offers and discounts significantly influence my decision to purchase through Online Food Delivery services.  ': 'CPB4',
    'I trust online food delivery platforms for consistent and reliable meal delivery.  ': 'CPB5',
    'My satisfaction with past experiences influences my decision to continue using online food delivery services. ': 'CPB6',
})

# Group variables
PU = ['PU1', 'PU2', 'PU3', 'PU4', 'PU5', 'PU6']
PEOU = ['PEOU1', 'PEOU2', 'PEOU3', 'PEOU4', 'PEOU5', 'PEOU6']
PP = ['PP1', 'PP2', 'PP3', 'PP4', 'PP5', 'PP6']
SQ = ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6']
CPB = ['CPB1', 'CPB2', 'CPB3', 'CPB4', 'CPB5', 'CPB6']

# Compute average score for each construct
constructs = df_renamed[PU + PEOU + PP + SQ + CPB].copy()
constructs['Perceived Usefulness'] = constructs[PU].mean(axis=1)
constructs['Perceived Ease of Use'] = constructs[PEOU].mean(axis=1)
constructs['Pricing and Promotions'] = constructs[PP].mean(axis=1)
constructs['Service Quality'] = constructs[SQ].mean(axis=1)
constructs['Consumer Purchasing Behaviour'] = constructs[CPB].mean(axis=1)

# Create correlation matrix from the composite scores
construct_means = constructs[['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality', 'Consumer Purchasing Behaviour']]
correlation_matrix = construct_means.corr()

# Print the correlation matrix
print(correlation_matrix)


# In[13]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming `construct_means` is your DataFrame with the computed mean scores:
X = construct_means[['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality']]
y = construct_means['Consumer Purchasing Behaviour']

# Fit the MLR model without adding a constant
mlr_model = sm.OLS(y, X).fit()

# Create regression summary table (without constant)
mlr_table = pd.DataFrame({
    'Variable': mlr_model.params.index,
    'Coefficient': mlr_model.params.values,
    'Std. Error': mlr_model.bse.values,
    't-Statistic': mlr_model.tvalues.values,
    'P-Value': mlr_model.pvalues.values
}).round(4)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Merge regression results with VIF
mlr_results_with_vif = pd.merge(mlr_table, vif_data, on="Variable")
print(mlr_results_with_vif)


# In[14]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your dataset
df = pd.read_csv("latest data.csv")  # Update path if needed

# Define constructs using iloc index ranges (same as your Cronbach alpha example)
perceived_usefulness = df.iloc[:, 10:15]
perceived_ease_of_use = df.iloc[:, 16:21]
pricing_and_promotion = df.iloc[:, 22:27]
service_quality = df.iloc[:, 28:33]
consumer_purchasing_behavior = df.iloc[:, 34:39]

# Compute average scores for each respondent
df_constructs = pd.DataFrame({
    'Perceived Usefulness': perceived_usefulness.mean(axis=1),
    'Perceived Ease of Use': perceived_ease_of_use.mean(axis=1),
    'Pricing and Promotions': pricing_and_promotion.mean(axis=1),
    'Service Quality': service_quality.mean(axis=1),
    'Consumer Purchasing Behaviour': consumer_purchasing_behavior.mean(axis=1)
})

# Define independent and dependent variables
X = df_constructs[['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality']]
y = df_constructs['Consumer Purchasing Behaviour']

# Run MLR (without constant)
model = sm.OLS(y, X).fit()
print(model.summary())

# Calculate VIF
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif)


# In[15]:


import pandas as pd
import statsmodels.api as sm

# Create a DataFrame with the average mean scores for each construct
# Assuming the number of respondents is 100 (you can change this to your actual number of respondents)
num_respondents = 154  # Update this as per your dataset size

df_constructs = pd.DataFrame({
    'Perceived Usefulness': [3.35] * num_respondents,  # Replace with your actual number of respondents
    'Perceived Ease of Use': [3.39] * num_respondents,  # Replace with your actual number of respondents
    'Pricing and Promotions': [3.31] * num_respondents,  # Replace with your actual number of respondents
    'Service Quality': [3.27] * num_respondents,  # Replace with your actual number of respondents
    'Consumer Purchasing Behaviour': [3.25] * num_respondents  # Example, replace with your actual average score for dependent variable
})

# Define independent and dependent variables
X = df_constructs[['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality']]
y = df_constructs['Consumer Purchasing Behaviour']

# Add a constant to the independent variables matrix (for the intercept)
X = sm.add_constant(X)

# Run MLR (Multiple Linear Regression)
model = sm.OLS(y, X).fit()

# Display the model summary
print(model.summary())


# In[16]:


import pandas as pd
import statsmodels.api as sm

# Load your data
df = pd.read_csv('latest data.csv')

# Define constructs using iloc index ranges
perceived_usefulness = df.iloc[:, 10:15]  # Columns for perceived usefulness
perceived_ease_of_use = df.iloc[:, 16:21]  # Columns for perceived ease of use
pricing_and_promotion = df.iloc[:, 22:27]  # Columns for pricing and promotion
service_quality = df.iloc[:, 28:33]  # Columns for service quality
consumer_purchasing_behavior = df.iloc[:, 34:39]  # Columns for consumer purchasing behavior

# Combine independent variables (X)
X = pd.concat([perceived_usefulness, perceived_ease_of_use, pricing_and_promotion, service_quality], axis=1)

# Define the dependent variable (Y)
# Assuming consumer_purchasing_behavior is the dependent variable
Y = consumer_purchasing_behavior.mean(axis=1)  # You can adjust this if you want a specific column

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the MLR model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression results
print(model.summary())


# In[17]:


import pandas as pd
import statsmodels.api as sm

# Load your data
df = pd.read_csv('latest data.csv')

# Define constructs using iloc index ranges
perceived_usefulness = df.iloc[:, 10:15]  # Columns for perceived usefulness
perceived_ease_of_use = df.iloc[:, 16:21]  # Columns for perceived ease of use
pricing_and_promotion = df.iloc[:, 22:27]  # Columns for pricing and promotion
service_quality = df.iloc[:, 28:33]  # Columns for service quality
consumer_purchasing_behavior = df.iloc[:, 34:39]  # Adjusted to correct column range

# Convert consumer purchasing behavior to numeric, forcing errors to NaN
consumer_purchasing_behavior = consumer_purchasing_behavior.apply(pd.to_numeric, errors='coerce')

# Check for any NaN values and handle them (e.g., by filling with the mean or dropping)
Y = consumer_purchasing_behavior.mean(axis=1)  # Calculate mean across the specified columns

# Combine the predictors (independent variables)
X = pd.concat([perceived_usefulness, perceived_ease_of_use, pricing_and_promotion, service_quality], axis=1)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the MLR model
model = sm.OLS(Y, X).fit()

# Get the summary of the regression results
summary = model.summary()

# Extract coefficients, standard errors, t-statistics, and p-values
coefficients = model.params
standard_errors = model.bse
t_statistics = model.tvalues
p_values = model.pvalues

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Coefficient': coefficients,
    'Standard Error': standard_errors,
    't-statistic': t_statistics,
    'p-value': p_values
})

# Calculate VIF for each feature (excluding the constant)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns[1:]  # Exclude the constant column
vif_data["VIF"] = [1 / (1 - sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared) 
                   for i in range(1, X.shape[1])]

# Display the results
print("MLR Results:")
print(results_df)
print("\nVariance Inflation Factor (VIF):")
print(vif_data)


# In[18]:


import pandas as pd
import statsmodels.api as sm

# Load your data
df = pd.read_csv('latest data.csv')

# Define constructs using iloc index ranges (you already have this part)
perceived_usefulness = df.iloc[:, 10:15]  # Columns for perceived usefulness
perceived_ease_of_use = df.iloc[:, 16:21]  # Columns for perceived ease of use
pricing_and_promotion = df.iloc[:, 22:27]  # Columns for pricing and promotion
service_quality = df.iloc[:, 28:33]  # Columns for service quality
consumer_purchasing_behavior = df.iloc[:, 34:39]  # Adjusted to correct column range

# Convert consumer purchasing behavior to numeric, forcing errors to NaN
consumer_purchasing_behavior = consumer_purchasing_behavior.apply(pd.to_numeric, errors='coerce')

# Compute the average score for each construct (for all respondents)
Y = consumer_purchasing_behavior.mean(axis=1)  # Dependent variable: mean of purchasing behavior
X = pd.concat([perceived_usefulness.mean(axis=1),
               perceived_ease_of_use.mean(axis=1),
               pricing_and_promotion.mean(axis=1),
               service_quality.mean(axis=1)], axis=1)  # Independent variables: mean scores of constructs

# Name the columns for X
X.columns = ['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the MLR model
model = sm.OLS(Y, X).fit()

# Get the summary of the regression results
summary = model.summary()

# Extract coefficients, standard errors, t-statistics, and p-values
coefficients = model.params
standard_errors = model.bse
t_statistics = model.tvalues
p_values = model.pvalues

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Coefficient': coefficients,
    'Standard Error': standard_errors,
    't-statistic': t_statistics,
    'p-value': p_values
})

# Calculate VIF for each feature (excluding the constant)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns[1:]  # Exclude the constant column
vif_data["VIF"] = [1 / (1 - sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared) 
                   for i in range(1, X.shape[1])]

# Display the results
print("MLR Results:")
print(results_df)
print("\nVariance Inflation Factor (VIF):")
print(vif_data)


# In[19]:


import pandas as pd
import statsmodels.api as sm

# Load your data
df = pd.read_csv('latest data.csv')

# Define constructs using iloc index ranges
perceived_usefulness = df.iloc[:, 10:15]
perceived_ease_of_use = df.iloc[:, 16:21]
pricing_and_promotion = df.iloc[:, 22:27]
service_quality = df.iloc[:, 28:33]
consumer_purchasing_behavior = df.iloc[:, 34:39]

# Convert to numeric to handle any non-numeric entries
consumer_purchasing_behavior = consumer_purchasing_behavior.apply(pd.to_numeric, errors='coerce')

# Dependent variable: average of consumer purchasing behavior
Y = consumer_purchasing_behavior.mean(axis=1)

# Independent variables: average scores for each construct
X = pd.concat([
    perceived_usefulness.mean(axis=1),
    perceived_ease_of_use.mean(axis=1),
    pricing_and_promotion.mean(axis=1),
    service_quality.mean(axis=1)
], axis=1)

# Rename columns
X.columns = ['Perceived Usefulness', 'Perceived Ease of Use', 'Pricing and Promotions', 'Service Quality']

# Add constant for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(Y, X).fit()

# Create summary table with 4 decimal places
results_df = pd.DataFrame({
    'Coefficient': model.params.round(4),
    'Standard Error': model.bse.round(4),
    't-statistic': model.tvalues.round(4),
    'p-value': model.pvalues.round(4)
})

# Calculate VIF (exclude constant)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns[1:]
vif_data["VIF"] = [round(1 / (1 - sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared), 4)
                   for i in range(1, X.shape[1])]

# Display results
print("MLR Results:")
print(results_df)
print("\nVariance Inflation Factor (VIF):")
print(vif_data)


# In[20]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming df is already loaded
# Combine all independent variable columns
X = pd.concat([
    df.iloc[:, 10:15],
    df.iloc[:, 16:21],
    df.iloc[:, 22:27],
    df.iloc[:, 28:33]
], axis=1)

# Reduce the dependent variable to a single target value, e.g., mean of the columns
y = df.iloc[:, 34:39].mean(axis=1)

# Split into train and test sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the MLR model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R-squared:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# In[21]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Combine independent variables
X = pd.concat([
    df.iloc[:, 10:15],  # perceived_usefulness
    df.iloc[:, 16:21],  # perceived_ease_of_use
    df.iloc[:, 22:27],  # pricing_and_promotion
    df.iloc[:, 28:33],  # service_quality
], axis=1)

# Name columns if they don't have names
X.columns = [f'X{i}' for i in range(X.shape[1])]

# Dependent variable (average across the consumer_purchasing_behavior columns)
y = df.iloc[:, 34:39].mean(axis=1)

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X_with_const).fit()

# Print the regression results
print(model.summary())

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)


# In[22]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Create aggregated predictors
X = pd.DataFrame({
    'perceived_usefulness': df.iloc[:, 10:15].mean(axis=1),
    'perceived_ease_of_use': df.iloc[:, 16:21].mean(axis=1),
    'pricing_and_promotion': df.iloc[:, 22:27].mean(axis=1),
    'service_quality': df.iloc[:, 28:33].mean(axis=1),
})

# Step 2: Create dependent variable (aggregate)
y = df.iloc[:, 34:39].mean(axis=1)

# Step 3: Add constant
X_with_const = sm.add_constant(X)

# Step 4: Fit the model
model = sm.OLS(y, X_with_const).fit()

# Step 5: Show regression output
print(model.summary())

# Step 6: Calculate and show VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors:")
print(vif_data)


# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[24]:


# Combine everything before modeling
full_data = pd.DataFrame({
    'perceived_usefulness': df.iloc[:, 10:15].mean(axis=1),
    'perceived_ease_of_use': df.iloc[:, 16:21].mean(axis=1),
    'pricing_and_promotion': df.iloc[:, 22:27].mean(axis=1),
    'service_quality': df.iloc[:, 28:33].mean(axis=1),
    'consumer_purchasing_behavior': df.iloc[:, 34:39].mean(axis=1)
})

# Drop rows with any missing values
full_data = full_data.dropna()

# Define X and y
X = full_data.drop(columns='consumer_purchasing_behavior')
y = full_data['consumer_purchasing_behavior']


# In[25]:


import statsmodels.api as sm

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set up plot style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Count the preferred apps
preferred_app_counts = df['Most Preferred App'].value_counts()

# Create the bar plot
ax = sns.barplot(x=preferred_app_counts.index, y=preferred_app_counts.values, palette='viridis')

# Add labels on bars
for i, value in enumerate(preferred_app_counts.values):
    ax.text(i, value + 1, str(value), ha='center', va='bottom', fontweight='bold')

# Customize plot
plt.title('Most Preferred App by Respondents', fontsize=14, fontweight='bold')
plt.xlabel('App')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[27]:


import statsmodels.api as sm
import matplotlib.pyplot as plt


# Generate the Q-Q plot for the residuals
sm.qqplot(model.resid, line='45', fit=True)
plt.title("Q-Q Plot of MLR Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()


# In[28]:


import matplotlib.pyplot as plt

# Plot histogram of residuals
plt.figure(figsize=(8, 5))
plt.hist(model.resid, bins=20, edgecolor='black', color='skyblue')
plt.title("Histogram of MLR Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[29]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

# Assuming your data is already loaded and cleaned:
# X = your independent variables (with sm.add_constant(X) already applied)
# y = your dependent variable
model = sm.OLS(y, X).fit()

# Fit the model if not already done
# model = sm.OLS(y, X).fit()

# Calculate Residual Standard Error (RSE)
residuals = model.resid
degrees_of_freedom = model.df_resid
residual_std_error = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)

# Extract key metrics
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
f_pvalue = model.f_pvalue  # Overall model significance

# Display results
print(f"Residual Standard Error: {residual_std_error:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.4f}")
print(f"Overall Model p-value: {f_pvalue:.4f}")


# In[30]:


import pandas as pd
import statsmodels.api as sm

# Step 1: Load your data
df = pd.read_csv("latest data.csv")

# Step 2: Define constructs based on column ranges
perceived_usefulness = df.iloc[:, 10:15]
perceived_ease_of_use = df.iloc[:, 16:21]
pricing_and_promotion = df.iloc[:, 22:27]
service_quality = df.iloc[:, 28:33]
consumer_purchasing_behavior = df.iloc[:, 34:39]

# Step 3: Compute average scores per respondent
df_model = pd.DataFrame({
    'PU': perceived_usefulness.mean(axis=1),
    'PEOU': perceived_ease_of_use.mean(axis=1),
    'PP': pricing_and_promotion.mean(axis=1),
    'SQ': service_quality.mean(axis=1),
    'CPB': consumer_purchasing_behavior.mean(axis=1)
})

# Step 4: Stepwise regression function using p-values
def stepwise_selection(X, y, 
                       initial_features=[], 
                       threshold_in=0.05, 
                       threshold_out=0.10, 
                       verbose=True):
    included = list(initial_features)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]

        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature} with p-value {best_pval:.4f}")

        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude constant
        worst_pval = pvalues.max() if not pvalues.empty else None
        if worst_pval is not None and worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Drop {worst_feature} with p-value {worst_pval:.4f}")

        if not changed:
            break
    return included

# Step 5: Define X and y
X = df_model[['PU', 'PEOU', 'PP', 'SQ']]
y = df_model['CPB']

# Step 6: Run stepwise regression
selected_features = stepwise_selection(X, y)

# Step 7: Fit final model using selected features
X_selected = sm.add_constant(X[selected_features])
final_model = sm.OLS(y, X_selected).fit()

# Step 8: Display results
print("\nSelected Features:", selected_features)
print("\nFinal Model Summary:")
print(final_model.summary())





# In[35]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Now use the cleaned column name
column_name = "What improvements would you suggest for online food delivery platforms?"

# Drop NaN, join responses into one string
text_data = df[column_name].dropna().astype(str).str.cat(sep=' ')

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Display
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Suggested Improvements for Online Food Delivery Platforms")
plt.show()


# In[ ]:




