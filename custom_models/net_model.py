from keras.models import Model, Input
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from keras.metrics import SparseCategoricalAccuracy

""" 
We have two implementation for the problem: LightGBM and Multilayer Perceptron Network.
"""
data_directory = 'C:\\Users\\NWEKE-PC\\OneDrive\\Documents\\Python\\Projects\\AI-Lab\\Digital-Agriculture\\data'


# Using boosted decision tree algorithm on structured data produces better performance
# An example of a structured data is the dataset used in this exercise while images are categorized as unstructured.
def agro_lgbm(train_X, train_y, eval_X, eval_y, test_X, test_y, test_dt):
    cat_cols = ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season', 'Crop_Type_c1', 'Soil_Type_c1',
                'Pesticide_Use_Category_c1', 'Season_c1']

    params = {'learning_rate': 0.04, 'max_depth': 18, 'n_estimators': 3000, 'objective': 'multiclass',
              'boosting_type': 'gbdt', 'subsample': 0.7, 'random_state': 42, 'colsample_bytree': 0.7,
              'min_data_in_leaf': 55, 'reg_alpha': 1.7, 'reg_lambda': 1.11}
    params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}

    clf = lgb.LGBMClassifier(**params)

    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (eval_X, eval_y)], eval_metric='multi_error',
            early_stopping_rounds=100, categorical_feature=cat_cols)

    # score = accuracy_score(train_y, prediction)

    # print(f'\nAccuracy: {score * 100}%')

    # Check for over-fitting
    print(f'Training Score: {clf.score(train_X, train_y) * 100}%')
    print(f'Testing Score: {clf.score(test_X, test_y) * 100}%')
    print(f'Validation Score: {clf.score(eval_X, eval_y) * 100}%')

    # clf.fit(dataset_X, dataset_y, eval_metric='multi_error', categorical_feature=cat_cols)
    prediction = clf.predict(test_dt)

    print(Counter(prediction))

    submission = pd.read_csv('data/sample_submission.csv')
    submission['Crop_Damage'] = prediction.data
    submission.to_csv('data/submit_lightGBM.csv')

    plt.rcParams['figure.figsize'] = (12, 6)
    lgb.plot_importance(clf)
    plt.show()


# Build a fully connected network - Multilayer perceptron neural network implementation for the problem
def agro_mlp(train_X, train_y, eval_X, eval_y, test_X, test_y, test_dt, num_classes):
    visible_layer = Input(shape=(36,))
    fc_layer_1 = Dense(360, activation='relu')(visible_layer)
    fc_layer_2 = Dense(250, activation='relu')(fc_layer_1)
    fc_layer_3 = Dense(226, activation='relu')(fc_layer_2)
    fc_layer_4 = Dense(175, activation='relu')(fc_layer_3)
    dropout_layer = Dropout(rate=0.2)(fc_layer_4)
    fc_layer_5 = Dense(120, activation='relu')(dropout_layer)
    fc_layer_6 = Dense(82, activation='relu')(fc_layer_5)
    fc_layer_7 = Dense(51, activation='relu')(fc_layer_6)
    fc_layer_8 = Dense(24, activation='relu')(fc_layer_7)
    output_layer = Dense(num_classes, activation='softmax')(fc_layer_8)

    model = Model(visible_layer, output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[SparseCategoricalAccuracy()])

    # Fit model
    model.fit(train_X, train_y, validation_data=(eval_X, eval_y), epochs=70, batch_size=700, verbose=2)

    # Evaluation
    loss_function, accuracy = model.evaluate(test_X, test_y, verbose=2)
    print(f'\nLoss Function: {loss_function}')
    print(f'Evaluation Accuracy: {accuracy * 100}')

    train_pred = model.predict(train_X)
    classes_X = np.argmax(train_pred, axis=1)
    print(classes_X.shape)

    score = accuracy_score(train_y, classes_X)
    print(f'\nAccuracy Score: {score * 100:.2f}%')
    # print(classification_report(train_y, model.predict(train_X), target_names=['0', '1', '2']))

    # Actual prediction
    prediction = model.predict(test_dt)
    classes_X = np.argmax(prediction, axis=1)
    print(classes_X.shape)

    submission = pd.read_csv('data/sample_submission.csv')
    submission['Crop_Damage'] = classes_X
    submission.to_csv('data/submit_mlp.csv')


"""
model = agro_net(num_classes)

plot_model(model, to_file='data/agro_net.png')

model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=45, batch_size=2000, verbose=2)

model.save('data/agro_net.h5')

loss_func, accuracy = model.evaluate(test_X, test_y, verbose=2)

print(f'\nLoss function: {loss_func}\nAccuracy: {accuracy * 100}%')
"""
