import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

'''
Because the football_wages set contains 3 times less data than the autograder data
we will not use any data from "football_wages.csv" exclusively for test. We will train and tune
10 models in different splits, then make them predict the the entire dataset "football_wages.csv"
and we will calculate the average error.

Since each model will be trying to overfit to the validation set it has been tuned, but
there is way less data it will be trying to overfit, the average performance might be a good
representative of how it will perform in "football_autograder.csv".

'''

# Load the data
df_train = pd.read_csv('football_wages.csv')
df_test = pd.read_csv('football_autograder.csv')

#Remove non sensitive data
df_train.drop('nationality_name',axis=1,inplace=True)
df_test.drop('nationality_name',axis=1,inplace=True)


#Collect target values
Y = np.array(df_train['log_wages'])

#Remove target values from dataframe
df_train.drop('log_wages',axis=1,inplace=True)

#Collect list of objects
X = np.array(df_train.values)
X_test = np.array(df_test.values)

#Normalize X
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std 
X_test = (X_test-X_mean)/X_std # X test is being normalized with same mean and std as the models are being trained upon

def train_and_test_knn(K, x_train, y_train, x_val,y_val):
    # Initialize the KNN regressor
    model = KNeighborsRegressor(n_neighbors=K)

    # Train the KNN regressor
    model.fit(x_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(x_val)

    # Calculate error
    error = np.mean(np.abs(y_val - y_pred))

    return model,error

def find_best_k_store_model(k,x_train,y_train,x_val,y_val):
    min_error = np.inf
    for k in range(1,40):
        model,error = train_and_test_knn(k,x_train,y_train,x_val,y_val)
        #print('K: ',k,' Error: ',error)

        if error < min_error:
            min_error = error
            best_k = k
            best_model = model
    print("Best K:", best_k)
    return best_k,best_model,min_error

#########################################
#--------KNN Cross validation Training 
#########################################
number_of_models = 15
best_KNN_model_list =[]
best_k_list = []
val_error_list = []

# Perform cross validation
for i in range(number_of_models):
    print("-----\nModel: ", i)
    x_train, x_val, y_train, y_val = train_test_split(X,Y,test_size=0.2,random_state=i)
    best_k,best_model,min_error = find_best_k_store_model(40,x_train,y_train,x_val,y_val)
    best_KNN_model_list.append(best_model)
    val_error_list.append(min_error)
    best_k_list.append(best_k)

print('Best K list: ',best_k_list)
norm_val_error = np.mean(np.array(val_error_list))
print('Normalized MAE on validation set: ',norm_val_error)
    
#########################################
#--------KNN Cross validation prediction
#########################################

Y_pred_list = []

#-----Predition of Y

for model in best_KNN_model_list:
    Y_pred_list.append(model.predict(X)) 

avrg_Y_predictions = np.mean(Y_pred_list,axis=0)
error_Y_norm = np.mean(np.abs(Y - np.array(avrg_Y_predictions)))

print("MAE prediction of Y (in log10):", error_Y_norm)

# #------Predition on X_test



# Y_test_pred_list = []

# for model in best_KNN_model_list:
#     Y_test_pred_list.append(model.predict(X_test))




# denorm_predictions = avrg_Y_norm_predictions * Y_std + Y_mean
# log_10_denorm_prediction = np.log10(denorm_predictions)


# file = open("test_predictions.txt", "w")

# with file:
#     string_to_write = '['
#     for pred in log_10_denorm_prediction:
#         string_to_write += str(pred)+","
#     string_to_write += ']'

#     file.write(string_to_write)
