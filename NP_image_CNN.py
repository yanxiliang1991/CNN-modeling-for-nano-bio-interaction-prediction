'''
Created by Xiliang Yan
2019-06-06
BSB, CCIB, Rutgers, Camden, NJ
'''

'''
Build CNN model
'''
from keras import models
from keras import layers
import os
import numpy as np
from sklearn.model_selection import KFold
from keras.preprocessing import image
import pandas as pd

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

'''
Split image folders into train and validation 
'''
df_image_folder_names = pd.read_excel('index_with_fold_names.xlsx', index_col=0)
df_test_index = pd.read_excel('testset_index.xlsx', index_col=0)

test_index = df_test_index['testset_index'].values
train_index = [x for x in df_image_folder_names.index if x not in test_index]

test_image_folder_names = df_image_folder_names.loc[test_index].values
train_image_folder_names = df_image_folder_names.loc[train_index].values

test_image_folder_names = np.asarray(test_image_folder_names).reshape(len(test_image_folder_names), )
train_image_folder_names = np.asarray(train_image_folder_names).reshape(len(train_image_folder_names), )

print(test_image_folder_names)
print(train_image_folder_names)

'''
5-fold cross validation
'''

df_activity = pd.read_excel('activity.xlsx', index_col=0)

kf = KFold(n_splits=5, random_state=0, shuffle=True)


i = 0
num_epochs = 300

all_scores = []
all_mae_histories = []
y_pred_5cv = []
y_exp_5cv = []
index_5cv = []


for partial_train_index, val_index in kf.split(train_image_folder_names):
    x_partial_train = []
    x_validation = []
    y_partial_train = []
    y_validation = []
    partial_train_image_folder_names = train_image_folder_names[partial_train_index]
    val_image_folder_names = train_image_folder_names[val_index]
    print(val_image_folder_names)
    print('processing fold #', i)
    for partial_train_image_folder_name in partial_train_image_folder_names:
        for partial_train_image_file_name in os.listdir(partial_train_image_folder_name):
            partial_train_image = image.load_img(os.path.join(partial_train_image_folder_name + '/' + partial_train_image_file_name).replace('\\', '/'),
                                         target_size=(200, 200))
            x_partial_train_image = image.img_to_array(partial_train_image)
            x_partial_train.append(x_partial_train_image)
            y_partial_train.append(df_activity.loc[partial_train_image_folder_name].values)
    x_partial_train = np.asarray(x_partial_train)
    x_partial_train = x_partial_train.astype('float32')/255
    y_partial_train = np.asarray(y_partial_train)

    for val_image_folder_name in val_image_folder_names:
        val_image_file_names = os.listdir(val_image_folder_name)
        val_image = image.load_img(os.path.join(val_image_folder_name + '/' + val_image_file_names[0]).replace('\\', '/'),
                                   target_size=(200, 200))
        x_val_image = image.img_to_array(val_image)
        x_validation.append(x_val_image)
        y_validation.append(df_activity.loc[val_image_folder_name].values)
    x_validation = np.asarray(x_validation)
    x_validation = x_validation.astype('float32')/255
    y_validation = np.asarray(y_validation)



    model = build_model()
    # history = model.fit(x_partial_train, y_partial_train, validation_data=(x_validation, y_validation),
    #                     epochs=num_epochs, batch_size=32)
    # mae_history = history.history['val_mean_absolute_error']
    # all_mae_histories.append(mae_history)
    # with open('historydict_{}'.format(i), 'wb') as f:
    #     pickle.dump(history.history, f)

    model.fit(x_partial_train, y_partial_train, epochs=num_epochs, batch_size=32)
    # val_mse, val_mae = model.evaluate(x_validation, y_validation)
    y_pre = model.predict(x_validation)
    y_pred_5cv.extend(y_pre)
    y_exp_5cv.extend(y_validation)
    index_5cv.extend(val_image_folder_names)
    # all_scores.append(val_mae)
    model.save('GNP_CNN_5cv_without_ferrocene_{}.h5'.format(i))
    i += 1


y_pred_5cv = np.asarray(y_pred_5cv).reshape(len(train_image_folder_names), )
y_exp_5cv = np.asarray(y_exp_5cv).reshape(len(train_image_folder_names), )
index_5cv = np.asarray(index_5cv).reshape(len(train_image_folder_names), )

ExpvsPred_logP_CNN_5CV = {'Exp': y_exp_5cv, 'Pred': y_pred_5cv}
pd.DataFrame(ExpvsPred_logP_CNN_5CV, index=index_5cv).to_excel('ExpvsPred_logP_CNN_5CV_without_ferrocene.xlsx')

'''
External testset validation
'''

x_train = []
y_train = []

for train_image_folder_name in train_image_folder_names:
    for train_image_file_name in os.listdir(train_image_folder_name):
        train_image = image.load_img(
            os.path.join(train_image_folder_name + '/' + train_image_file_name).replace('\\', '/'),
            target_size=(200, 200))
        x_train_image = image.img_to_array(train_image)
        x_train.append(x_train_image)
        y_train.append(df_activity.loc[train_image_folder_name].values)


x_train = np.asarray(x_train)
x_train = x_train.astype('float32')/255
y_train = np.asarray(y_train)

x_test = []
y_test = []

for test_image_folder_name in test_image_folder_names:
    test_image_file_names = os.listdir(test_image_folder_name)
    test_image = image.load_img(os.path.join(test_image_folder_name + '/' + test_image_file_names[0]).replace('\\', '/'),
                               target_size=(200, 200))
    x_test_image = image.img_to_array(test_image)
    x_test.append(x_test_image)
    y_test.append(df_activity.loc[test_image_folder_name].values)


x_test = np.asarray(x_test)
x_test = x_test.astype('float32')/255
y_test = np.asarray(y_test)


model = build_model()
model.fit(x_train, y_train, epochs=num_epochs, batch_size=32)
model.save('GNP_CNN_test_without_ferrocene.h5')
y_pred_test = model.predict(x_test)
y_pred_test = np.asarray(y_pred_test).reshape(len(test_image_folder_names), )
y_exp_test = y_test.reshape(len(test_image_folder_names), )

ExpvsPred_logP_CNN_test = {'Exp': y_exp_test, 'Pred': y_pred_test}
pd.DataFrame(ExpvsPred_logP_CNN_test, index=test_image_folder_names).to_excel('ExpvsPred_logP_CNN_test_without_ferrocene.xlsx')















