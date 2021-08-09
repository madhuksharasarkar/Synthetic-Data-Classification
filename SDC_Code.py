import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



def test_data_points(df):
    test_points = df.as_matrix(columns=df.columns[0:2])
    test_labels = np.array(df['Label'])
    return (test_points, test_labels)

def data_label_split(df):
    matrix = df.values
    data = matrix[:,0:2]
    label = matrix[:,2]
    return (data, label)

def misclassified_points(test_df, predicted_class_label):
    misclassified_points = []
    test_df['Predicted Label'] = predicted_class_label
    for index, row in test_df.iterrows():
        if (row['Label'] != row['Predicted Label']):
            misclassified_points.append((row['x'], row['y'], row['Label'], row['Predicted Label']))
    
    return (misclassified_points)



# Function for generating Half-Moons Dataset
def halfmoons(n_points, r, sep, dist, noise):
    theta = np.linspace(0, np.pi, n_points)
    x1 = r*np.cos(theta) + (np.random.rand(n_points, 1)*noise).T
    y1 = r*np.sin(theta) + (np.random.rand(n_points, 1)*noise).T
    
    theta1 = np.linspace(0, -np.pi, n_points)
    x2 = r*np.cos(theta1) + sep + (np.random.rand(n_points, 1)*noise).T
    y2 = r*np.sin(theta1) + dist + (np.random.rand(n_points, 1)*noise).T
    
    return (np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T)),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


X1_1, y1_1 = halfmoons(1000, 1.0, 1.0, -0.8, 0.5)
df_moon_1 = pd.DataFrame(dict(x=X1_1[:,0], y=X1_1[:,1], Label=y1_1))

plt.title('Half Moons Dataset: Case 1')
plt.plot(X1_1[y1_1==0,0], X1_1[y1_1==0,1], '.', label='0')
plt.plot(X1_1[y1_1==1,0], X1_1[y1_1==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()


X1_2, y1_2 = halfmoons(1000, 1.2, 1.2, 0.5, 0.5)
df_moon_2 = pd.DataFrame(dict(x=X1_2[:,0], y=X1_2[:,1], Label=y1_2))

plt.title('Half Moons Dataset: Case 2')
plt.plot(X1_2[y1_2==0,0], X1_2[y1_2==0,1], '.', label='0')
plt.plot(X1_2[y1_2==1,0], X1_2[y1_2==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()


X1_3, y1_3 = halfmoons(1000, 0.7, 0.65, 0.25, 0.5)
df_moon_3 = pd.DataFrame(dict(x=X1_3[:,0], y=X1_3[:,1], Label=y1_3))

plt.title('Half Moons Dataset: Case 3')
plt.plot(X1_3[y1_3==0,0], X1_3[y1_3==0,1], '.', label='0')
plt.plot(X1_3[y1_3==1,0], X1_3[y1_3==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()


# Function for splitting Half-Moons Dataset into Train and Test Data
def half_moon_split(df):
    n = int(len(df) * 0.15)
    m = int(len(df) * 0.05)
    overlap_area = []
    non_overlap_area = []
    
    for index, row in df.iterrows():
        if (row['y'] >= 0.00 and row['y'] <= 0.75 and row['x'] >= 0.0 and row['x'] <= 1.0):
            overlap_area.append(row)
        else:
            non_overlap_area.append(row)
    
    df1 = pd.DataFrame(overlap_area)
    df2 = pd.DataFrame(non_overlap_area)
    
    train_1, test_1 = train_test_split(df1, test_size = n, random_state=10)
    train_2, test_2 = train_test_split(df2, test_size = m, random_state=10)
    
    train_df = pd.concat([train_1, train_2], ignore_index = True)
    test_df = pd.concat([test_1, test_2], ignore_index = True)

    return (train_df, test_df)


train_df_moon_1, test_df_moon_1 = train_test_split(df_moon_1, test_size = 0.20, random_state=10)
train_df_moon_2, test_df_moon_2 = train_test_split(df_moon_2, test_size = 0.20, random_state=10)
train_df_moon_3, test_df_moon_3 = half_moon_split(df_moon_3)



u1_1, v1_1 = test_data_points(test_df_moon_1)

plt.plot(X1_1[y1_1==0,0], X1_1[y1_1==0,1], '.', label='0')
plt.plot(X1_1[y1_1==1,0], X1_1[y1_1==1,1], '.', label='1', color = 'yellow')
plt.plot(u1_1[v1_1==0,0], u1_1[v1_1==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u1_1[v1_1==1,0], u1_1[v1_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()

u1_2, v1_2 = test_data_points(test_df_moon_2)

plt.plot(X1_2[y1_2==0,0], X1_2[y1_2==0,1], '.', label='0')
plt.plot(X1_2[y1_2==1,0], X1_2[y1_2==1,1], '.', label='1', color = 'yellow')
plt.plot(u1_2[v1_2==0,0], u1_2[v1_2==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u1_2[v1_2==1,0], u1_2[v1_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()

u1_3, v1_3 = test_data_points(test_df_moon_3)

plt.plot(X1_3[y1_3==0,0], X1_3[y1_3==0,1], '.', label='0')
plt.plot(X1_3[y1_3==1,0], X1_3[y1_3==1,1], '.', label='1', color = 'yellow')
plt.plot(u1_3[v1_3==0,0], u1_3[v1_3==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u1_3[v1_3==1,0], u1_3[v1_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()



# Function for generating Two-Spirals Dataset
def twospirals(n_points, noise):
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))



X2_1, y2_1 = twospirals(1000, 0.5)
df_spiral_1 = pd.DataFrame(dict(x=X2_1[:,0], y=X2_1[:,1], Label=y2_1))

plt.title('Two Spiral Dataset: Case 1')
plt.plot(X2_1[y2_1==0,0], X2_1[y2_1==0,1], '.', label='0')
plt.plot(X2_1[y2_1==1,0], X2_1[y2_1==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()

X2_2, y2_2 = twospirals(1000, 1.0)
df_spiral_2 = pd.DataFrame(dict(x=X2_2[:,0], y=X2_2[:,1], Label=y2_2))

plt.title('Two Spiral Dataset: Case 2')
plt.plot(X2_2[y2_2==0,0], X2_2[y2_2==0,1], '.', label='0')
plt.plot(X2_2[y2_2==1,0], X2_2[y2_2==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()

X2_3, y2_3 = twospirals(1000, 1.5)
df_spiral_3 = pd.DataFrame(dict(x=X2_3[:,0], y=X2_3[:,1], Label=y2_3))

plt.title('Two Spiral Dataset: Case 3')
plt.plot(X2_3[y2_3==0,0], X2_3[y2_3==0,1], '.', label='0')
plt.plot(X2_3[y2_3==1,0], X2_3[y2_3==1,1], '.', label='1', color = 'yellow')
plt.legend()
plt.show()



train_df_spiral_1, test_df_spiral_1 = train_test_split(df_spiral_1, test_size = 0.2, random_state=10)
train_df_spiral_2, test_df_spiral_2 = train_test_split(df_spiral_2, test_size = 0.2, random_state=10)
train_df_spiral_3, test_df_spiral_3 = train_test_split(df_spiral_3, test_size = 0.2, random_state=10)



u2_1, v2_1 = test_data_points(test_df_spiral_1)

plt.plot(X2_1[y2_1==0,0], X2_1[y2_1==0,1], '.', label='0')
plt.plot(X2_1[y2_1==1,0], X2_1[y2_1==1,1], '.', label='1', color = 'yellow')
plt.plot(u2_1[v2_1==0,0], u2_1[v2_1==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u2_1[v2_1==1,0], u2_1[v2_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()

u2_2, v2_2 = test_data_points(test_df_spiral_2)

plt.plot(X2_2[y2_2==0,0], X2_2[y2_2==0,1], '.', label='0')
plt.plot(X2_2[y2_2==1,0], X2_2[y2_2==1,1], '.', label='1', color = 'yellow')
plt.plot(u2_2[v2_2==0,0], u2_2[v2_2==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u2_2[v2_2==1,0], u2_2[v2_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()

u2_3, v2_3 = test_data_points(test_df_spiral_3)

plt.plot(X2_3[y2_3==0,0], X2_3[y2_3==0,1], '.', label='0')
plt.plot(X2_3[y2_3==1,0], X2_3[y2_3==1,1], '.', label='1', color = 'yellow')
plt.plot(u2_3[v2_3==0,0], u2_3[v2_3==0,1], '.', color = 'Red', label = 'Test Points')
plt.plot(u2_3[v2_3==1,0], u2_3[v2_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()




train_moon_data_1, train_moon_label_1 = data_label_split(train_df_moon_1)
test_moon_data_1, test_moon_label_1 = data_label_split(test_df_moon_1)

train_moon_data_2, train_moon_label_2 = data_label_split(train_df_moon_2)
test_moon_data_2, test_moon_label_2 = data_label_split(test_df_moon_2)

train_moon_data_3, train_moon_label_3 = data_label_split(train_df_moon_3)
test_moon_data_3, test_moon_label_3 = data_label_split(test_df_moon_3)


train_spiral_data_1, train_spiral_label_1 = data_label_split(train_df_spiral_1)
test_spiral_data_1, test_spiral_label_1 = data_label_split(test_df_spiral_1)

train_spiral_data_2, train_spiral_label_2 = data_label_split(train_df_spiral_2)
test_spiral_data_2, test_spiral_label_2 = data_label_split(test_df_spiral_2)

train_spiral_data_3, train_spiral_label_3 = data_label_split(train_df_spiral_3)
test_spiral_data_3, test_spiral_label_3 = data_label_split(test_df_spiral_3)


#######################################################################################
############################### Support Vector Machine ################################
#######################################################################################

def svm_classifier(training_set, training_label, test_set, test_label, kernel):
    param_grid_svm = [{'C': [0.001, 0.01, 0.1, 10, 100]}]
    svm = SVC(kernel = kernel, degree=3)
    #svm = LinearSVC()
    
    grid = GridSearchCV(svm, param_grid_svm, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
    grid.fit(training_set, training_label)
    svm = grid.best_estimator_
    
    predicted_class_label = svm.predict(test_set)
    predicted_class_label = list(predicted_class_label)
    
    conf_matrix = confusion_matrix(test_label, predicted_class_label)
    precision = precision_score(test_label, predicted_class_label, average='macro')
    recall = recall_score(test_label, predicted_class_label, average='macro')
    f_score = f1_score(test_label, predicted_class_label, average='macro') 
    
    return (svm, conf_matrix, precision, recall, f_score, predicted_class_label)


est11, conf11, pr11, re11, f11, label11 = svm_classifier(train_moon_data_1, train_moon_label_1, 
                                           test_moon_data_1, test_moon_label_1,
                                           kernel='poly')

est12, conf12, pr12, re12, f12, label12 = svm_classifier(train_moon_data_2, train_moon_label_2, 
                                           test_moon_data_2, test_moon_label_2,
                                           kernel='poly')

est13, conf13, pr13, re13, f13, label13 = svm_classifier(train_moon_data_3, train_moon_label_3, 
                                           test_moon_data_3, test_moon_label_3,
                                           kernel='poly')


mis_points_11 = pd.DataFrame(misclassified_points(test_df_moon_1, label11), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_1, b1_1 = test_data_points(mis_points_11)

plt.title('Polynomial')
plt.plot(X1_1[y1_1==0,0], X1_1[y1_1==0,1], '.', label='0')
plt.plot(X1_1[y1_1==1,0], X1_1[y1_1==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_1[b1_1==0,0], a1_1[b1_1==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_1[b1_1==1,0], a1_1[b1_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_12 = pd.DataFrame(misclassified_points(test_df_moon_2, label12), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_2, b1_2 = test_data_points(mis_points_12)

plt.title('Polynomial')
plt.plot(X1_2[y1_2==0,0], X1_2[y1_2==0,1], '.', label='0')
plt.plot(X1_2[y1_2==1,0], X1_2[y1_2==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_2[b1_2==0,0], a1_2[b1_2==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_2[b1_2==1,0], a1_2[b1_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_13 = pd.DataFrame(misclassified_points(test_df_moon_3, label13), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_3, b1_3 = test_data_points(mis_points_13)

plt.title('Polynomial')
plt.plot(X1_3[y1_3==0,0], X1_3[y1_3==0,1], '.', label='0')
plt.plot(X1_3[y1_3==1,0], X1_3[y1_3==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_3[b1_3==0,0], a1_3[b1_3==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_3[b1_3==1,0], a1_3[b1_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()



est21, conf21, pr21, re21, f21, label21 = svm_classifier(train_spiral_data_1, train_spiral_label_1, 
                                           test_spiral_data_1, test_spiral_label_1,
                                           kernel='linear')

est22, conf22, pr22, re22, f22, label22 = svm_classifier(train_spiral_data_2, train_spiral_label_2, 
                                           test_spiral_data_2, test_spiral_label_2,
                                           kernel='linear')

est23, conf23, pr23, re23, f23, label23 = svm_classifier(train_spiral_data_3, train_spiral_label_3, 
                                           test_spiral_data_3, test_spiral_label_3,
                                           kernel='linear')

mis_points_21 = pd.DataFrame(misclassified_points(test_df_spiral_1, label21), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_1, b2_1 = test_data_points(mis_points_21)

plt.title('Polynomial')
plt.plot(X2_1[y2_1==0,0], X2_1[y2_1==0,1], '.', label='0')
plt.plot(X2_1[y2_1==1,0], X2_1[y2_1==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_1[b2_1==0,0], a2_1[b2_1==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_1[b2_1==1,0], a2_1[b2_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_22 = pd.DataFrame(misclassified_points(test_df_spiral_2, label22), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_2, b2_2 = test_data_points(mis_points_22)

plt.title('Polynomial')
plt.plot(X2_2[y2_2==0,0], X2_2[y2_2==0,1], '.', label='0')
plt.plot(X2_2[y2_2==1,0], X2_2[y2_2==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_2[b2_2==0,0], a2_2[b2_2==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_2[b2_2==1,0], a2_2[b2_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_23 = pd.DataFrame(misclassified_points(test_df_spiral_3, label23), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_3, b2_3 = test_data_points(mis_points_23)

plt.title('Polynomial')
plt.plot(X2_3[y2_3==0,0], X2_3[y2_3==0,1], '.', label='0')
plt.plot(X2_3[y2_3==1,0], X2_3[y2_3==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_3[b2_3==0,0], a2_3[b2_3==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_3[b2_3==1,0], a2_3[b2_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()

######################################################################################
############################### Multilayer Perceptron ################################
######################################################################################

def mlp_classifier(training_set, training_label, test_set, test_label, hidden_layer_sizes, activation):
    param_grid_mlp = [{'alpha': [0.001, 0.01, 0.1, 10, 100]}]
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation=activation, solver='lbfgs')
    
    grid = GridSearchCV(mlp, param_grid_mlp, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
    grid.fit(training_set, training_label)
    mlp = grid.best_estimator_
    
    predicted_class_label = mlp.predict(test_set)
    predicted_class_label = list(predicted_class_label)
    
    conf_matrix = confusion_matrix(test_label, predicted_class_label)
    precision = precision_score(test_label, predicted_class_label, average='macro')
    recall = recall_score(test_label, predicted_class_label, average='macro')
    f_score = f1_score(test_label, predicted_class_label, average='macro') 
    
    return (mlp, conf_matrix, precision, recall, f_score, predicted_class_label)


est11, conf11, pr11, re11, f11, label11 = mlp_classifier(train_moon_data_1, train_moon_label_1, 
                                           test_moon_data_1, test_moon_label_1,
                                           hidden_layer_sizes=(3,2), activation='logistic')

est12, conf12, pr12, re12, f12, label12 = mlp_classifier(train_moon_data_2, train_moon_label_2, 
                                           test_moon_data_2, test_moon_label_2,
                                           hidden_layer_sizes=(3,2), activation='relu')

est13, conf13, pr13, re13, f13, label13 = mlp_classifier(train_moon_data_3, train_moon_label_3, 
                                           test_moon_data_3, test_moon_label_3,
                                           hidden_layer_sizes=(3,2), activation='relu')

mis_points_11 = pd.DataFrame(misclassified_points(test_df_moon_1, label11), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_1, b1_1 = test_data_points(mis_points_11)

plt.title('ReLU')
plt.plot(X1_1[y1_1==0,0], X1_1[y1_1==0,1], '.', label='0')
plt.plot(X1_1[y1_1==1,0], X1_1[y1_1==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_1[b1_1==0,0], a1_1[b1_1==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_1[b1_1==1,0], a1_1[b1_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_12 = pd.DataFrame(misclassified_points(test_df_moon_2, label12), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_2, b1_2 = test_data_points(mis_points_12)

plt.title('ReLU')
plt.plot(X1_2[y1_2==0,0], X1_2[y1_2==0,1], '.', label='0')
plt.plot(X1_2[y1_2==1,0], X1_2[y1_2==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_2[b1_2==0,0], a1_2[b1_2==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_2[b1_2==1,0], a1_2[b1_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_13 = pd.DataFrame(misclassified_points(test_df_moon_3, label13), columns = ['x', 'y', 'Label', 'Predicted Label'])
a1_3, b1_3 = test_data_points(mis_points_13)

plt.title('ReLU')
plt.plot(X1_3[y1_3==0,0], X1_3[y1_3==0,1], '.', label='0')
plt.plot(X1_3[y1_3==1,0], X1_3[y1_3==1,1], '.', label='1', color = 'yellow')
plt.plot(a1_3[b1_3==0,0], a1_3[b1_3==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a1_3[b1_3==1,0], a1_3[b1_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()



est21, conf21, pr21, re21, f21, label21 = mlp_classifier(train_spiral_data_1, train_spiral_label_1, 
                                           test_spiral_data_1, test_spiral_label_1,
                                           hidden_layer_sizes=(25,23), activation='relu')

est22, conf22, pr22, re22, f22, label22 = mlp_classifier(train_spiral_data_2, train_spiral_label_2, 
                                           test_spiral_data_2, test_spiral_label_2,
                                           hidden_layer_sizes=(25,23), activation='relu')

est23, conf23, pr23, re23, f23, label23 = mlp_classifier(train_spiral_data_3, train_spiral_label_3, 
                                           test_spiral_data_3, test_spiral_label_3,
                                           hidden_layer_sizes=(25,23), activation='relu')


mis_points_21 = pd.DataFrame(misclassified_points(test_df_spiral_1, label21), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_1, b2_1 = test_data_points(mis_points_21)

plt.title('ReLU')
plt.plot(X2_1[y2_1==0,0], X2_1[y2_1==0,1], '.', label='0')
plt.plot(X2_1[y2_1==1,0], X2_1[y2_1==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_1[b2_1==0,0], a2_1[b2_1==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_1[b2_1==1,0], a2_1[b2_1==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_22 = pd.DataFrame(misclassified_points(test_df_spiral_2, label22), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_2, b2_2 = test_data_points(mis_points_22)

plt.title('ReLU')
plt.plot(X2_2[y2_2==0,0], X2_2[y2_2==0,1], '.', label='0')
plt.plot(X2_2[y2_2==1,0], X2_2[y2_2==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_2[b2_2==0,0], a2_2[b2_2==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_2[b2_2==1,0], a2_2[b2_2==1,1], '.', color = 'Red')
plt.legend()
plt.show()


mis_points_23 = pd.DataFrame(misclassified_points(test_df_spiral_3, label23), columns = ['x', 'y', 'Label', 'Predicted Label'])
a2_3, b2_3 = test_data_points(mis_points_23)

plt.title('ReLU')
plt.plot(X2_3[y2_3==0,0], X2_3[y2_3==0,1], '.', label='0')
plt.plot(X2_3[y2_3==1,0], X2_3[y2_3==1,1], '.', label='1', color = 'yellow')
plt.plot(a2_3[b2_3==0,0], a2_3[b2_3==0,1], '.', color = 'Red', label = 'Misclassified Points')
plt.plot(a2_3[b2_3==1,0], a2_3[b2_3==1,1], '.', color = 'Red')
plt.legend()
plt.show()
