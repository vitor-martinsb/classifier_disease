import pandas as pd
import numpy as np

class data:
    def __init__(self, fold='./Data',file='laura1.csv',perc_train=0.7):
        self.fold = fold
        self.file = file
        self.database = pd.read_csv(fold+'/'+file, skiprows=0,index_col=False)
        self.perc_train = perc_train
        self.perc_valid = 1-perc_train
        columns_name = ['idade','setor','temperatura','freq_resp','pa_sis','pa_dia','pa_med','sa_02','status']
        self.database.columns = columns_name
        
    def random_base(self,df):
        '''
        Torna aleatorio a base

        Parameters
        ----------
        df : dataframe
            dataframe que será randomizado

        Returns
        -------
        df: dataframe
            dataframe randomizado

        '''
        return df.sample(n=len(df))
    
    def treat_base(self):
        '''
        Separa a base de dados com base nos óbitos e melhorias
        
        Returns
        -------
        df_ob : dataframe
            dataframe de óbitos
        df_me : dataframe
            dataframe de melhora

        '''
        
        df_comp = self.database.replace({'status': {'Obito': -1, 'Melhorado': 1}}) #Define os rótulos
        df_comp = df_comp.dropna() #Retira da base todos os dados que possuem nan
        
        #Separar base por rótulo 
        df_ob = df_comp[df_comp['status'] == -1] 
        df_me = df_comp[df_comp['status'] == 1]
        
        dim_data_DFOB = len(df_ob)
        dim_data_DFME = len(df_me)
        
        if dim_data_DFOB > dim_data_DFME:
            dim_data = dim_data_DFME
            
        else:
            dim_data = dim_data_DFOB
            
        df_ob=df_ob[0:dim_data][:]
        df_me=df_me[0:dim_data][:] 
            
        return df_ob, df_me
    
    def separete_base(self,df1,df2):
        '''
        
        Separa a base para treino e validação

        Parameters
        ----------
        df1 : dataframe
            
        df2 : dataframe
            

        Returns
        -------
        list
            lista de dataframes para treinamento e validação

        '''
        train_dim = int(np.floor(len(df1)*self.perc_train)) #dimensao de dados treinamento
        val_dim = int(np.floor(len(df2)*self.perc_valid)) #dimensao de dados para validacao
        
        df1_t = df1[0:train_dim][:] #separa treinamento
        df1_v =df1[train_dim:train_dim+val_dim][:] #separa validação
        
        df2_t = df2[0:train_dim][:] #separa treinamento
        df2_v = df2[train_dim:train_dim+val_dim][:] #separa validação
        
        df_valid = pd.concat([df1_v, df2_v], axis=0, join='inner') #junta dos dois dataframes (df_me e df_ob)
        df_train = pd.concat([df1_t, df2_t], axis=0, join='inner')
        
        return [df_train,df_valid]
        
if __name__ == "__main__":
    data = data(fold='./Data',file='laura1.csv')
    df_comp = data.database
    df_ob, df_me = data.treat_base()
    df_ob_rand = data.random_base(df_ob)
    df_me_rand = data.random_base(df_me)
    list_dataframe_tv = data.separete_base(df_ob_rand,df_me_rand)
    
    
