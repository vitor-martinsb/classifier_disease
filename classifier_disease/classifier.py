import pandas as pd
import numpy as np
from data_treat import data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf


class linear_classfier:
    def __init__(self, fold='./Data', file='laura1.csv', perc_train=0.7, num_loop=100):
        self.Data = data(fold='./Data', file='laura1.csv',
                         perc_train=perc_train)
        self.df_complete = self.Data.database
        self.df_ob, self.df_me = self.Data.treat_base()
        self.df_ob_rand = self.Data.random_base(self.df_ob)
        self.df_me_rand = self.Data.random_base(self.df_me)
        self.list_dataframe_t_v = self.Data.separete_base(
            self.df_ob_rand, self.df_me_rand)
        self.num_loop = num_loop

    def get_rot(self, df):
        return df['status'].to_numpy()

    def get_data(self, df, columns):
        for k in range(0, len(columns)):
            df[columns[k]] = df[columns[k]].min()/(df[columns[k]].max() -
                                                   df[columns[k]].min())
        return df[columns].to_numpy()

    def get_weights(self, X, R):
        return np.dot(np.linalg.pinv(X), R)

    def ret_result(self, X, w):
        return(np.dot(X, w))


class ANN:
    def __init__(self, fold='./Data', file='laura1.csv', perc_train=0.7, num_loop=100):
        self.Data = data(fold='./Data', file='laura1.csv',
                         perc_train=perc_train)
        self.df_complete = self.Data.database
        self.df_ob, self.df_me = self.Data.treat_base()
        self.df_ob_rand = self.Data.random_base(self.df_ob)
        self.df_me_rand = self.Data.random_base(self.df_me)
        self.list_dataframe_t_v = self.Data.separete_base(
            self.df_ob_rand, self.df_me_rand)
        self.num_loop = num_loop

    def get_rot(self, df):
        return df['status'].to_numpy()

    def get_data(self, df, columns):
        for k in range(0, len(columns)):
            df[columns[k]] = df[columns[k]].min()/(df[columns[k]].max() -
                                                   df[columns[k]].min())
        return df[columns].to_numpy()

    def create_model(self, n_neurons=10, func_activation='relu', input_shape=0):
        input_shape = np.shape(Data_T)[1]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=n_neurons,
                  activation='relu', input_shape=(input_shape,)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[
                      'sparse_categorical_accuracy'])

        return model


if __name__ == "__main__":

    linear_classifier = False
    columns_name = ['temperatura', 'freq_resp',
                    'pa_sis', 'pa_dia', 'pa_med', 'sa_02']
    num_loop = 20
    n_neuron = 10
    if linear_classifier:
        lc = linear_classfier(fold='./Data', file='laura1.csv')
        lc.num_loop = num_loop
        lc.df_complete = lc.Data.database
        lc.df_ob, lc.df_me = lc.Data.treat_base()
        result_ger = np.zeros([lc.num_loop, 2])
        result_ger_mean = np.zeros(2)
        result_ger_std = np.zeros(2)

        for it in range(0, lc.num_loop):
            lc.df_ob_rand = lc.Data.random_base(lc.df_ob)
            lc.df_me_rand = lc.Data.random_base(lc.df_me)
            lc.list_dataframe_t_v = lc.Data.separete_base(
                lc.df_ob_rand, lc.df_me_rand)

            df_T = lc.list_dataframe_t_v[0]
            R_T = lc.get_rot(df_T)
            Data_T = lc.get_data(df_T, columns_name)

            df_V = lc.list_dataframe_t_v[1]
            R_V = lc.get_rot(df_V)
            Data_V = lc.get_data(df_V, columns_name)

            w = lc.get_weights(Data_T, R_T)
            result = lc.ret_result(Data_V, w)

            for k in range(0, len(result)):
                if result[k] < 0 and R_V[k] < 0:
                    result_ger[it, 0] = result_ger[it, 0] + 1
                elif result[k] > 0 and R_V[k] > 0:
                    result_ger[it, 1] = result_ger[it, 1] + 1

        if np.mean(100*result_ger[:,0]/len(R_V)) < 50:
            result_ger = 100 - 100*result_ger/len(R_V)
            result_ger = 100 - 100 * result_ger / len(R_V)
        else:
            result_ger = 100*result_ger/len(R_V)
        result_ger_mean[0] = np.mean(result_ger[:, 0])
        result_ger_mean[1] = np.mean(result_ger[:, 1])
        result_ger_std[0] = np.std(result_ger[:, 0])
        result_ger_std[1] = np.std(result_ger[:, 1])

        print(' \n Óbito \n Média: ',
              result_ger_mean[0], '% \n std: ', result_ger_std[0], '% \n')
        print(' \n Melhora \n Média: ',
              result_ger_mean[1], '% \n std: ', result_ger_std[1], '% \n')

    else:
        nlc = ANN(fold='./Data', file='laura1.csv',
                  perc_train=0.7, num_loop=100)
        accur = np.zeros([num_loop])

        for k in range(0, num_loop):
            nlc.df_ob_rand = nlc.Data.random_base(nlc.df_ob)
            nlc.df_me_rand = nlc.Data.random_base(nlc.df_me)
            nlc.list_dataframe_t_v = nlc.Data.separete_base(
                nlc.df_ob_rand, nlc.df_me_rand)
            df_T = nlc.list_dataframe_t_v[0]
            R_T = nlc.get_rot(df_T)+1
            Data_T = nlc.get_data(df_T, columns_name)

            df_V = nlc.list_dataframe_t_v[1]
            R_V = nlc.get_rot(df_V)+1
            Data_V = nlc.get_data(df_V, columns_name)

            model = nlc.create_model( n_neurons=5, func_activation='relu', input_shape=np.shape(Data_T)[1])
            model.fit(Data_T, R_T, epochs=5)
            _, accur[k] = model.evaluate(Data_V, R_V)

            if accur[k] < 1:
                accur[k] = 1 - accur[k]
        print('\n Acuracia do sistema: ', np.mean(
            accur), '\n Desvio padrão: ', np.std(accur))
