import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = read_data('data/Customer_train.csv')
test_data = read_data('data/Customer_test.csv')
# print(data.head())

# print(data['Segmentation'].unique())

# drop 'ID' column
data = data.drop(columns=['ID'])
test_data = test_data.drop(columns=['ID'])

print(data.isnull().sum())
print(data.shape[0])

# ---------------------------- handle missing values ----------------------------

# drop rows with missing values in more than 2 columns
data = data.dropna(thresh=data.shape[1]-2)
test_data = test_data.dropna(thresh=test_data.shape[1]-2)

# drop rows with missing values in the 'Graduated' column
data = data.dropna(subset=['Graduated'])
test_data = test_data.dropna(subset=['Graduated'])

# show rows with missing values in the 'Work_Experience' column
print(data[data['Work_Experience'].isnull()])

# if Graduated is 'No', missing values in 'Work_Experience' should be 0
data.loc[(data['Graduated'] == 'No') & (data['Work_Experience'].isnull()), 'Work_Experience'] = 0
test_data.loc[(test_data['Graduated'] == 'No') & (test_data['Work_Experience'].isnull()), 'Work_Experience'] = 0

print(data.isnull().sum())

# drop rows with missing values in any column
data = data.dropna()
test_data = test_data.dropna()

# length of the data after dropping rows with missing values
print(data.shape[0])
print(test_data.shape[0])


# ---------------------------- handle categorical data ----------------------------

# one-hot encoding for 'Gender', 'Profession'
data = pd.get_dummies(data, columns= ['Gender','Profession'])
test_data = pd.get_dummies(test_data, columns= ['Gender','Profession'])

# label encoding for 'Spendings_Score', 'Graduated', 'Ever_Married'
data['Spending_Score'] = data['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})
test_data['Spending_Score'] = test_data['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})

data['Graduated'] = data['Graduated'].map({'Yes': 1, 'No': 0})
test_data['Graduated'] = test_data['Graduated'].map({'Yes': 1, 'No': 0})

data['Ever_Married'] = data['Ever_Married'].map({'Yes': 1, 'No': 0})
test_data['Ever_Married'] = test_data['Ever_Married'].map({'Yes': 1, 'No': 0})

# label encoding for 'Var_1'
data['Var_1'] = data['Var_1'].map({'Cat_1': 1, 'Cat_2': 2, 'Cat_3': 3, 'Cat_4': 4, 'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7})
test_data['Var_1'] = test_data['Var_1'].map({'Cat_1': 1, 'Cat_2': 2, 'Cat_3': 3, 'Cat_4': 4, 'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7})

# label encoding for 'Segmentation'
data['Segmentation'] = data['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
# test_data['Segmentation'] = test_data['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})


# # ---------------------------- scaling ----------------------------


scaler = MinMaxScaler()

# scale 'Work_Experience', 'Family_Size'
# data[['Work_Experience', 'Family_Size']] = scaler.fit_transform(data[['Work_Experience', 'Family_Size']])
# test_data[['Work_Experience', 'Family_Size']] = scaler.transform(test_data[['Work_Experience', 'Family_Size']])

# data[['Work_Experience', 'Family_Size', 'Age']] = scaler.fit_transform(data[['Work_Experience', 'Family_Size', 'Age']])
# test_data[['Work_Experience', 'Family_Size', 'Age']] = scaler.transform(test_data[['Work_Experience', 'Family_Size', 'Age']])


# # ---------------------------- binning ----------------------------

# data['Age'] =pd.qcut(data['Age'], q=5, labels=False)
# test_data['Age'] =pd.qcut(test_data['Age'], q=5, labels=False)



# save the data to a new csv file
data.to_csv('data/train_preprocessed.csv', index=False)
test_data.to_csv('data/test_preprocessed.csv', index=False)