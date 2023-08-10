#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer 
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars ,RANSACRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
import shap


# In[2]:


df = pd.read_excel(r"D:\Articles\Steel shear wall\Dataset.xlsx"  ,header = 0 )


# In[3]:


y = df.loc[:, 'Vmax(kN)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4]].to_numpy()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8), dpi=600)
sns.pairplot(df, kind="reg", plot_kws={"scatter_kws": {'color': 'red'}})
plt.savefig(r"D:\Articles\Steel shear wall\Visual\1.jpeg", dpi=1000 ,format='jpeg')
plt.show()


# In[7]:


df.describe()


# In[4]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
scalerX=StandardScaler()
Xtr1=scalerX.fit_transform(Xtr)
Xte1=scalerX.transform(Xte)
SGD=SGDRegressor(random_state=1)
SGD.fit(Xtr1 , ytr)
yprtr = SGD.predict(Xtr1)
yprte = SGD.predict(Xte1)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,2)
msete=round(mean_squared_error(yte , yprte)**0.5,2)
a = min([np.min(yprtr), np.min(yprte), 0])
b = max([np.max(yprtr), np.max(yprte), 1])
plt.scatter(ytr, yprtr, s=80, facecolors='crimson', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}')
plt.scatter(yte , yprte,s=80, marker='s',facecolors='cyan', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'SGD ',fontsize=14)
plt.xlabel('V max [kN]_Finite elements',fontsize=15)
plt.ylabel('V max [kN]_predicted',fontsize=15)
# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\SGD.jpeg", dpi=1000 ,format='jpeg')
plt.show()
print(sqrt(mean_squared_error(yte, yprte)))


# In[21]:


Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=42)
scalerX = StandardScaler()
Xtr1 = scalerX.fit_transform(Xtr)
Xte1 = scalerX.transform(Xte)
model = SGDRegressor()
model.fit(Xtr1, ytr)
yprtr = model.predict(Xtr1)
yprte = model.predict(Xte1)
r2tr = round(r2_score(ytr, yprtr), 2)
r2te = round(r2_score(yte, yprte), 2)
a = min([np.min(yprtr), np.min(yprte), 0])
b = max([np.max(yprtr), np.max(yprte), 1])

plt.scatter(ytr, yprtr, s=80, facecolors='none', edgecolors='blue',
            label=f'\n Train \n R2 = {r2tr}  \n')  # Using empty circles (facecolors='none')
plt.scatter(yte, yprte, s=80, marker='s',facecolors='none', edgecolors='magenta',
            label=f'\n Test \n R2 = {r2te} \n')  # Using empty circles (facecolors='none')

plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title('SGD', fontsize=14)
plt.xlabel('V max [kN]_Finite elements', fontsize=14)
plt.ylabel('V max [kN]_predicted', fontsize=14)

# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)

plt.legend()
plt.tight_layout()
# plt.savefig(r"D:\Articles\Steel shear wall\result\SGD.jpeg", dpi=1000, format='jpeg')
plt.show()

print(sqrt(mean_squared_error(yte, yprte)))


# In[5]:


# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Generate the formula
formula = "Target = "
for i in range(len(coefficients)):
    feature_name = df.columns[i]
    coefficient = coefficients[i]
    formula += f"({coefficient:.4f} * {feature_name}) + "

formula += f"({intercept[0]:.4f})"

print("Formula:", formula)
mean_value = scalerX.mean_
std_value = scalerX.scale_

# Print the mean and standard deviation
print("Mean: ", mean_value)
print("Standard Deviation: ", std_value)


# In[19]:


a=(2.5-1.725)/0.5616
b=(150-202.142)/60.318
c=(400-360.714)/64.027
d=(2- 2.528)/ 0.6028
e=(30-24.535)/5.9430
m=(197.9909 * a) + (59.5034 * b) + (56.5588 * c) + (95.4537 * d) + (-59.0735 * e) + (978.1202)
m


# In[21]:


a = 198.0947/0.56164618
b = 59.2319/60.31803467
c = 56.4296/64.02725823
d = 96.1915/0.60288422
e = -59.1290/5.94307606
print(a,b,c,d,e)


# In[18]:


print(Xte[0])
print(Xte1[0])
model.predict([Xte1[0]])


# In[22]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
model=DecisionTreeRegressor(random_state=0)
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,2)
msete=round(mean_squared_error(yte , yprte)**0.5,2)
a = min([np.min(yprtr), np.min(yprte), 0])
b = max([np.max(yprtr), np.max(yprte), 1])
plt.scatter(ytr, yprtr, s=80, facecolors='orange', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}')
plt.scatter(yte , yprte,s=80, marker='s',facecolors='lime', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'DT ',fontsize=14)
plt.xlabel('V max [kN]_Finite elements',fontsize=15)
plt.ylabel('V max [kN]_predicted',fontsize=15)
# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\DT.jpeg", dpi=1000 ,format='jpeg')
plt.show()
print(sqrt(mean_squared_error(yte, yprte)))


# In[33]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
model=RandomForestRegressor(random_state=0)
model.fit(Xtr , ytr)
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,2)
msete=round(mean_squared_error(yte , yprte)**0.5,2)
a = min([np.min(yprtr), np.min(yprte), 0])
b = max([np.max(yprtr), np.max(yprte), 1])
plt.scatter(ytr, yprtr, s=80, facecolors='cornflowerblue', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}')
plt.scatter(yte , yprte,s=80, marker='s',facecolors='tomato', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'RF ',fontsize=14)
plt.xlabel('V max [kN]_Finite elements',fontsize=15)
plt.ylabel('V max [kN]_predicted',fontsize=15)
# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\RF.jpeg", dpi=1000 ,format='jpeg')
plt.show()
print(sqrt(mean_squared_error(yte, yprte)))


# In[28]:


import numpy as np

def gepModelVmax_kN(d):
    G1C6 = -11.995864741966
    G1C1 = 5.54918057802057
    G1C0 = 6.34937589648122
    G2C7 = 4.40839869380779
    G2C1 = 7.3711355937376

    y = np.empty_like(d[:, 0])

    y = (((d[:, 2] / d[:, 4]) / (d[:, 3] / G1C6)) - ((d[:, 4] * G1C1) - np.exp(G1C0)))
    y = y + ((np.sqrt(np.power(d[:, 3], G2C7)) / (d[:, 4] / d[:, 2])) + ((G2C7 * G2C1) - (d[:, 4] + d[:, 2])))
    y = y + (d[:, 1] + ((d[:, 0] * d[:, 2]) + d[:, 3]))

    return y

geptr = gepModelVmax_kN(Xtr)
gepte = gepModelVmax_kN(Xte)

r2tr=round(r2_score(ytr , geptr),2)
r2te=round(r2_score(yte , gepte),2)
msetr=round(mean_squared_error(ytr , geptr)**0.5,2)
msete=round(mean_squared_error(yte , gepte)**0.5,2)

a = min([np.min(geptr), np.min(gepte), 0])
b = max([np.max(geptr), np.max(gepte), 1])

plt.scatter(ytr, yprtr, s=80, facecolors='deeppink', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}')
plt.scatter(yte , yprte,s=80, marker='s',facecolors='khaki', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'GEP ',fontsize=14)
plt.xlabel('V max [kN]_Finite elements',fontsize=15)
plt.ylabel('V max [kN]_predicted',fontsize=15)
# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\GEP.jpeg", dpi=1000 ,format='jpeg')
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import numpy as np

# R2 values for each model (example data)
model_names = ['SGD', 'DT', 'RF',"GEP"]
model1_r2 = 0.96
model2_r2 = 0.9
model3_r2 = 0.95
model4_r2 = 0.98
# Combine the R2 values into a list
r2_values = [model1_r2, model2_r2, model3_r2,model4_r2]

# Define colors for each model
colors = ['darksalmon', 'aquamarine', 'violet','gainsboro']

# Plot the R2 values for each model with different colors
bars = plt.bar(model_names, r2_values, color=colors)

# Set labels and title
plt.xlabel('\n Models')
plt.ylabel('R2 Value')
# plt.title('Comparison of R2 Values for Machine Learning Models')

# Attach a text label above each bar displaying its height
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

# Show the graph
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\R2.jpeg", dpi=1000 ,format='jpeg')
plt.show()




# R2 values for each model (example data)
model_names = ['SGD', 'DT', 'RF',"GEP"]
model1_r2 = 49.64
model2_r2 = 80.66
model3_r2 = 55.56
model4_r2 = 39.77

# Combine the R2 values into a list
r2_values = [model1_r2, model2_r2, model3_r2,model4_r2]

# Define colors for each model
colors = ['darksalmon', 'aquamarine', 'violet','gainsboro']

# Plot the R2 values for each model with different colors
bars = plt.bar(model_names, r2_values, color=colors)

# Set labels and title
plt.xlabel('\n Models')
plt.ylabel('RMSE Value')
# plt.title('Comparison of R2 Values for Machine Learning Models')

# Attach a text label above each bar displaying its height
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

# Show the graph
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\RMSE.jpeg", dpi=1000 ,format='jpeg')
plt.show()


# In[6]:


features=['L/h','fyp  (MPa)','fyf (MPa)','tp (mm)','Ao/Ap (%)']
explainer = shap.Explainer(SGD,Xtr1,feature_names=features)
shap_values = explainer(Xte1)
fig, ax = plt.subplots(figsize=(12, 8))

# Create a beeswarm plot using shap.plots.beeswarm()
# shap.plots.beeswarm(shap_values, show=False)
shap.summary_plot(shap_values, Xte1, color='coolwarm', show=False)
# Customize the plot using matplotlib functions
ax.set_xlabel('SHapley values:impact on model output', fontsize=12, fontname='Times New Roman',fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontname='Times New Roman',fontweight='bold')
# ax.set_title('Shapley Values', fontsize=16, fontname='Times New Roman')

# Set the font of the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper right', fontsize=12)
for text in legend.texts:
    text.set_fontname('Times New Roman')

# Set the font of the ticks
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')

# Save the figure
fig.savefig(r"D:\Articles\Steel shear wall\Visual\shap1.jpeg", dpi=2000, bbox_inches='tight')


# In[21]:


shap.plots.bar(shap_values)
plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
plt.savefig(r"D:\Articles\Steel shear wall\Visual\s2.jpeg", dpi=2000, bbox_inches='tight',format='jpeg')
plt.show()


# In[22]:


c=['green','orange','deeppink','b']
shap.summary_plot(shap_values, Xte1, plot_type="bar",axis_color='black',color=c)
plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
plt.savefig(r"D:\Articles\Steel shear wall\Visual\shap2.jpeg", dpi=2000,format='jpeg')
plt.show()


# In[29]:


import shap
import matplotlib.pyplot as plt

# Assuming you have a single sample Xte1 for which you have computed SHAP values
# If you have multiple samples, you can choose the desired index, e.g., Xte1[0]
shap_values_single_sample = shap_values[0]

# Create the bar plot
shap.plots.bar(shap_values_single_sample, show=False)

# Adjust the figure size
plt.figure(figsize=(20, 10))

# Set title and labels
plt.title("Custom Title")
plt.xlabel("Custom X-Label")

# Set the x-axis limit
plt.xlim([-12, 12])

# Show the plot
plt.show()


# In[11]:


shap_values[0]


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['L/h', 'tp(mm)', 'Ao/Ap(%)', 'fyp(MPa)', 'fyf(MPa)']
values = [171, 80.92, 52.94, 49.45, 49.41]
colors = ['blue', 'green', 'orange', 'red', 'purple']

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))

plt.barh(labels, values, color=colors)

for i, v in enumerate(values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.xlabel('Average impact on model from base value = 987.48')
# plt.ylabel('Average impact on model from base value = 999')
# plt.title('Horizontal Bar Chart with Different Colors and Data Labels')
plt.savefig(r"D:\Articles\Steel shear wall\Visual\shap2.jpeg", dpi=1000,format='jpeg')
plt.show()


# In[9]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
model=GradientBoostingRegressor()
# Fit the model
model.fit(Xtr, ytr)

# Make predictions
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)

# Calculate R2 scores
r2tr = round(r2_score(ytr, yprtr), 2)
r2te = round(r2_score(yte, yprte), 2)

# Calculate RMSE for the test set
msetr = round(mean_squared_error(ytr, yprtr)**0.5, 2)
msete = round(mean_squared_error(yte, yprte)**0.5, 2)

# Flatten yte and yprte arrays
yte = yte.flatten()
yprte = yprte.flatten()

# Plot the results for the training set
data_index = np.arange(1, len(ytr) + 1)

# Plotting the actual data points as (index, real) with circles and a continuous line
plt.plot(data_index, ytr, 'bo-', label='Actual Data')

# Plotting the predicted data points as (index, model) with squares and a dotted line
plt.plot(data_index, yprtr, 'gs:', label='Train Data')

# Plotting the errors as (index, |error|) with triangles and a dashed line
errors = [abs(a - p) for a, p in zip(ytr, yprtr)]
plt.plot(data_index, errors, 'r^--', label=f'Absolute Errors')

# Adding labels and legend
plt.xlabel('Data Index (Factor of 3)')
plt.ylabel('Target Values')
plt.legend()

# Set x-axis ticks to show only integers as factors of 3
x_ticks = np.arange(1, len(data_index) + 1, 10)
plt.xticks(x_ticks)

# Display the plot
plt.show()

# Plot the results for the test set
data_index1 = np.arange(1, len(yte) + 1)

# Plotting the actual data points as (index, real) with circles and a continuous line
plt.plot(data_index1, yte, 'bo-', label='Actual Data')

# Plotting the predicted data points as (index, model) with squares and a dotted line
plt.plot(data_index1, yprte, 'gs:', label=f'Test Data (R2={r2te}, RMSE={msete})')

# Plotting the errors as (index, |error|) with triangles and a dashed line
errors = [abs(a - p) for a, p in zip(yte, yprte)]
plt.plot(data_index1, errors, 'r^--', label='Absolute Errors')

# Adding labels and legend
plt.xlabel('Data Index (Factor of 3)')
plt.ylabel('Target Values')
plt.legend(loc=10)  # Place the legend at an appropriate location

# Set x-axis ticks to show only integers as factors of 3
x_ticks1 = np.arange(1, len(data_index1) + 1, 5)
plt.xticks(x_ticks1)

# Display the plot
plt.show()


# In[20]:


import numpy as np

def gepModelVmax_kN(d):
    G1C6 = -11.995864741966
    G1C1 = 5.54918057802057
    G1C0 = 6.34937589648122
    G2C7 = 4.40839869380779
    G2C1 = 7.3711355937376

    y = np.empty_like(d[:, 0])

    y = (((d[:, 2] / d[:, 4]) / (d[:, 3] / G1C6)) - ((d[:, 4] * G1C1) - np.exp(G1C0)))
    y = y + ((np.sqrt(np.power(d[:, 3], G2C7)) / (d[:, 4] / d[:, 2])) + ((G2C7 * G2C1) - (d[:, 4] + d[:, 2])))
    y = y + (d[:, 1] + ((d[:, 0] * d[:, 2]) + d[:, 3]))

    return y

geptr = gepModelVmax_kN(Xtr)
gepte = gepModelVmax_kN(Xte)

r2tr=round(r2_score(ytr , geptr),2)
r2te=round(r2_score(yte , gepte),2)

a = min([np.min(geptr), np.min(gepte), 0])
b = max([np.max(geptr), np.max(gepte), 1])

plt.scatter(ytr, yprtr, s=80, facecolors='red', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \n')
plt.scatter(yte , yprte,s=80, facecolors='blueviolet', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \n')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'GEP ',fontsize=14)
plt.xlabel('V max [kN]_real',fontsize=14)
plt.ylabel('V max [kN]_predicted',fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\Articles\Steel shear wall\result\GEP.jpeg", dpi=1000 ,format='jpeg')
plt.show()
print(sqrt(mean_squared_error(yte, yprte)))


# In[16]:


r2=round(r2_score(yte , m),2)
r2


# In[6]:


from lazypredict.Supervised import LazyRegressor


# In[7]:


reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(Xtr1, Xte1, ytr, yte)

print(models)


# In[16]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

correlation = df.corr()
sns.heatmap(correlation , vmin=-1, vmax=1)


# In[9]:


corr = df.corr()
corr.style.background_gradient(cmap='OrRd').set_precision(2)


# In[ ]:


df = pd.read_excel(r"D:\Articles\Steel shear wall\Dataset.xlsx"  ,header = 0 )
y = df.loc[:, 'Vmax(kN)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4]].to_numpy()
Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
scalerX=StandardScaler()
Xtr1=scalerX.fit_transform(Xtr)
Xte1=scalerX.transform(Xte)
model=SGDRegressor()
model.fit(Xtr1 , ytr)


# In[9]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
scalerX=StandardScaler()
Xtr1=scalerX.fit_transform(Xtr)
Xte1=scalerX.transform(Xte)
model1=SGDRegressor()
model1.fit(Xtr1 , ytr)
y_pred1 = model1.predict(Xte1)

Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
model2=DecisionTreeRegressor()
model2.fit(Xtr , ytr)
y_pred2 = model2.predict(Xte)

results_df = pd.DataFrame({
    'Target': yte,
    'Model1 Predicted': y_pred1,
    'Model2 Predicted': y_pred2
 
})

# Calculate the error values for each model
results_df['Error Model1'] = results_df['Model1 Predicted'] - results_df['Target']
results_df['Error Model2'] = results_df['Model2 Predicted'] - results_df['Target']


# Display the DataFrame
print(results_df)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Split the data for Model 1
Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, train_size=0.7, random_state=42)

scalerX = StandardScaler()
Xtr1 = scalerX.fit_transform(Xtr1)
Xte1 = scalerX.transform(Xte1)

model1 = SGDRegressor()
model1.fit(Xtr1, ytr1)
y_pred1 = model1.predict(Xte1)

# Split the data for Model 2
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, train_size=0.7, random_state=42)

model2 = DecisionTreeRegressor()
model2.fit(Xtr2, ytr2)
y_pred2 = model2.predict(Xte2)

Xtr3, Xte3, ytr3, yte3 = train_test_split(X, y, train_size=0.7, random_state=42)

model3 = RandomForestRegressor()
model3.fit(Xtr3, ytr3)
y_pred3 = model3.predict(Xte3)

def gepModelVmax_kN(d):
    G1C6 = -11.995864741966
    G1C1 = 5.54918057802057
    G1C0 = 6.34937589648122
    G2C7 = 4.40839869380779
    G2C1 = 7.3711355937376

    y = np.empty_like(d[:, 0])

    y = (((d[:, 2] / d[:, 4]) / (d[:, 3] / G1C6)) - ((d[:, 4] * G1C1) - np.exp(G1C0)))
    y = y + ((np.sqrt(np.power(d[:, 3], G2C7)) / (d[:, 4] / d[:, 2])) + ((G2C7 * G2C1) - (d[:, 4] + d[:, 2])))
    y = y + (d[:, 1] + ((d[:, 0] * d[:, 2]) + d[:, 3]))

    return y

y_pred4 = gepModelVmax_kN(Xte)

results_df = pd.DataFrame({
    'Target': yte1.flatten(),  # Flatten to make sure the arrays are 1-dimensional
    'SGD': y_pred1.flatten(),
    'DT': y_pred2.flatten(),
    'RF': y_pred3.flatten(),
    'GEP': y_pred4.flatten(),
})

# Calculate the error values for each model
results_df['SGD_Error'] = results_df['SGD'] - results_df['Target']
results_df['DT_Error'] = results_df['DT'] - results_df['Target']
results_df['RF_Error'] = results_df['RF'] - results_df['Target']
results_df['GEP_Error'] = results_df['GEP'] - results_df['Target']
# Display the DataFrame
df1 = results_df.round(2)
df1


# In[26]:


df1.to_excel(r"C:\Users\ALIREZA\Desktop\Book1.xlsx", index=False)


# In[ ]:




