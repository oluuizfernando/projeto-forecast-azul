 # Importar Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# Importe de Dados

df=#Coloque aqui a conecxão com o banco de dados
periodo = pd.read_excel('C:/Users/bruno/OneDrive/Área de Trabalho/Curso Data Expert DNC/AZUL/De_Para_Periodo.xlsx')


#Variavel Faixa horaria - hora cheia extraida do capo DepTime
df['faixa_horaria']=(df['DepTime'].astype('str')).str[:2].astype('int')

#Variavel Chave - Usada na construção de dataFrames auxiliares
df['chave2']=df['Dep Date']+df['Segment']+(df['faixa_horaria'].astype('int')).astype('str')+df['Nest']

#Calculo dos indicadores RASK e Ocupação
df['Rask']=df['Revenue']/df['ASKs']
df['Ocupacao']=df['RPKs']/df['ASKs']

#Unindo o Periodo com o Data Frame
df=pd.merge(df,periodo, on= 'faixa_horaria')


# Ajuste do nome das colunas, retirada de espaços
df=df.rename(columns={'Dep Date':'Dep_Date','Event Name':'Event_Name'})

#Ajustando Variaveis
df['Capture']= df['Capture'].astype('datetime64')
df['Dep_Date']= df['Dep_Date'].astype('datetime64')


#Transformação de campo Categorico em Numérico
df['Nest'] = df['Nest'].map({'Y':0, 'C':1})


#Seleção de Dados para construção d modelo

dforiglimpo2 = df[['Capture', 'FlightID', 'NDO', 'Dep_Date', 'Event_Name', 
                   'EvDef',  'DOW', 'Month', 'Description', 'Nest', 
                   'Segment', 'Dest', 'Flts', 'Lid','faixa_horaria', 
                   'chave2', 'Rask', 'Ocupacao', 'Periodo','Distance',
                   'ASKs', 'RPKs', 'Booked', 'NSKBooked', 'GDSBooked', 
                   'TudoAzulBooked','AzulViagensBooked', 'CNXBooked', 'Revenue', 'RevenueNSK', 
                   'RevenueGDS', 'RevenueTudoAzul','RevenueAzulViagens', 'DLRev']]


del df


#Transformação de Variaveis cátegoricas em Numéricas
lab = LabelEncoder()
dforiglimpo2['Event_Name']=lab.fit_transform(dforiglimpo2['Event_Name'])
dforiglimpo2['EvDef']=lab.fit_transform(dforiglimpo2['EvDef'])
dforiglimpo2['DOW']=lab.fit_transform(dforiglimpo2['DOW'])
dforiglimpo2['Description']=lab.fit_transform(dforiglimpo2['Description'])
dforiglimpo2['Month']=lab.fit_transform(dforiglimpo2['Month'])


# Transformação de campos numéricos em campos Escalonados

Scaler = StandardScaler()

#atribuição dos campos em uma nova variavel
tranformar=dforiglimpo2[['ASKs','RPKs','Revenue','RevenueNSK','RevenueGDS','RevenueTudoAzul','RevenueAzulViagens','DLRev']]

#Transformação ds campos da nova Variavel
transformadas=Scaler.fit_transform(tranformar)

#Substituição dos campos originais pelos transformados
dforiglimpo2['ASKs']=transformadas[:,0]
dforiglimpo2['RPKs']=transformadas[:,1]
dforiglimpo2['Revenue']=transformadas[:,2]
dforiglimpo2['RevenueNSK']=transformadas[:,3]
dforiglimpo2['RevenueGDS']=transformadas[:,4]
dforiglimpo2['RevenueTudoAzul']=transformadas[:,5]
dforiglimpo2['RevenueAzulViagens']=transformadas[:,6]
dforiglimpo2['DLRev']=transformadas[:,7]


# Seleção de um Dia especifico de Venda, representação de rodada do modelo em produção com dados em D-1
dfTeste=dforiglimpo2[dforiglimpo2['Capture']=='2022-07-13']
dfTeste=pd.DataFrame(dfTeste).reset_index()
x=int((dfTeste.shape[0])*0.01)
dfTeste=dfTeste.sample(x,replace=True)


#Atribuição da Data de Capture selecionada na ação anterior a uma nova variavel
data=dfTeste['Capture'].max()

#Identificação de Voos que estejam com data anterior a data selecionada
grava_fim=dforiglimpo2.loc[(dforiglimpo2['Capture']<=data)]

#Seleção de / Atribuição dos voos anteriores a data selecionada e que já tenham sido finalizados (NDO = -1)
grava_fim=grava_fim.query('NDO==-1')

#Criação de um novo campo na nova variavel criada para identificação dos voos já finalizados
grava_fim['Finalizado']=1

#Seleção dos dois campos que serão usados na base final do modelo 
grava_fim=grava_fim[['chave2','Finalizado']]

#União dos dados recem gerados com a base principal do modelo
dforiglimpo2=pd.merge(dforiglimpo2, grava_fim, how='left', on='chave2')


#Variavel Vazia para receber o resultado da Predição
base=[]

#for para percorrer todas as linhas da base selecionada / Definição de Variaveis para Filtros
for i in range(len(dfTeste)):#len(dfTeste)
     
     lin=dfTeste.iloc[i]
     a=lin['NDO']
     b=lin['Dep_Date']
     # c=lin['FlightID']
     c=lin['Segment']
     d=lin['Periodo']
     e=lin['Ocupacao']
     f=lin['Flts']
     g=lin['chave2']
     h=lin['Nest']
     j=lin['Dest']
     k=lin['DOW']
     m=lin['Capture']
     n=lin['Rask']


     print(i)

     try:

          try:
               #Filtro de linhas que possuem a mesma caracteristica da linha selecionada anteriormente
               subdf1=dforiglimpo2.loc[(dforiglimpo2['NDO']==a) & (dforiglimpo2['Dep_Date'].values<=b) & (dforiglimpo2['Segment']==c) &
               (dforiglimpo2['Periodo']==d)&(dforiglimpo2['Nest']==h)&(dforiglimpo2['Capture']<=m)&(dforiglimpo2['Finalizado']==1)]
               
               #Seleção dos dados referente ao voo que será previsto
               subdfpred=dforiglimpo2.loc[(dforiglimpo2['chave2'].values==g)&(dforiglimpo2['Capture']==m)]

               #Concatenação dos dados históricos com os dados que serão previstos
               subdf1=pd.concat([subdfpred,subdf1])

               #Filtro de linhas com as mesma caracteristica da linha selecionada anteriormente, porem com NDO = -1 para definição da Variavel Y
               subdf2=dforiglimpo2.loc[(dforiglimpo2['NDO']==-1) & (dforiglimpo2['Dep_Date'].values<b) & (dforiglimpo2['Segment']==c) & 
                                   (dforiglimpo2['Periodo']==d)&(dforiglimpo2['Nest']==h)]
               
               #Sepação dos dados que serão usados como Variavel Y nas duas Previsões (Ocupação e Rask)
               y=subdf2[['chave2','Ocupacao', 'Rask']]

               #Merge das informações X e Y
               subdf3=pd.merge(subdf1,y,how='left',on='chave2')

               #Definição das Variaveis que serão apagadas do modelo
               subdf_Ocupacao=subdf3[[  'Event_Name', 'EvDef',  'DOW','Month', 'Description', 'chave2', 'Ocupacao_x', 'RPKs', 'Booked','Revenue',  'Finalizado', 'Ocupacao_y', ]]                              


               subdf_Ocupacao=subdf_Ocupacao.set_index('chave2')


               N_Linhas=len(subdf_Ocupacao)
          
               #Definição de Base de Teste
               pred1=subdf_Ocupacao.loc[(subdf_Ocupacao['Finalizado']!=1)]
               
               #Definição de Base de Treino
               subdf_Ocupacao=subdf_Ocupacao.loc[(subdf_Ocupacao['Finalizado']==1)]  
               
               
               #Definição de x e y para o modelo - Previsão Ocupação
               y= np.array(subdf_Ocupacao['Ocupacao_y'])
               x=subdf_Ocupacao.drop(['Ocupacao_y','Finalizado'], axis = 1) 
               pred1=pred1.drop(columns=['Ocupacao_y','Finalizado'], axis = 1)
               x_list = list(x.columns)
               x = np.array(x)
               pred1 = np.array(pred1)

               
               

               #Definição de x e y para o modelo - Previsão Rask
               subdf_Rask=subdf3[[ 'Event_Name', 'EvDef', 'DOW', 'Month', 'Description', 'Nest', 'chave2', 'Rask_x', 'RPKs', 'Booked', 'CNXBooked', 'Revenue', 'Finalizado','Rask_y' ]]


               subdf_Rask=subdf_Rask.set_index('chave2')
               #Definição de Base de Teste
               pred2=subdf_Rask.loc[(subdf_Rask['Finalizado']!=1)]
               
               #Definição de Base de Treino
               subdf_Rask=subdf_Rask.loc[(subdf_Rask['Finalizado']==1)]  


               y2=np.array(subdf_Rask['Rask_y'])
               x2= subdf_Rask.drop(['Finalizado','Rask_y' ], axis = 1) 
               pred2=pred2.drop(columns=['Finalizado','Rask_y',])
               x_list2 = list(x2.columns)
               x2 = np.array(x2)
               pred2 = np.array(pred2)

               #Uso de Try para prevenção de erros de execução devido a falta de dados históricos no voo que causam erro no modelo
                              
               #Modelo Usado - Random Forest Regressor
               model =RandomForestRegressor()

               # Treino e Previsão da Variavel Ocuçpação
               model.fit(x, y)
               model_pred = model.predict(pred1)

               #Definição de métricas para Avaliação do Modelo
               z= model.predict(x)
               r2 = model.score(x, y)
               MAE= metrics.mean_absolute_error(y, z)
               MSE= metrics.mean_squared_error(y, z)
               RMSE=np.sqrt(metrics.mean_squared_error(y, z))   

               # Treino e Previsão da Variavel Rask
               model.fit(x2, y2)
               model_pred2 = model.predict(pred2)

               #Definição de métricas para Avaliação do Modelo
               z2= model.predict(x2)
               r22 = model.score(x2, y2)
               MAE2= metrics.mean_absolute_error(y2, z2)
               MSE2= metrics.mean_squared_error(y2, z2)
               RMSE2=np.sqrt(metrics.mean_squared_error(y2, z2))
               
               base.append([g,b,c,d,h,a,N_Linhas,e,np.round(r2,3),np.round(MAE,5),model_pred[0],n,np.round(r22,3),np.round(MAE2,5),model_pred2[0]])

          except:
               
               #Filtro de linhas que possuem a mesma caracteristica da linha selecionada anteriormente
               subdf1=dforiglimpo2.loc[(dforiglimpo2['NDO']==a) & (dforiglimpo2['Dep_Date'].values<=b) & (dforiglimpo2['Dest']==j) &
               (dforiglimpo2['Periodo']==d)&(dforiglimpo2['Nest']==h)&(dforiglimpo2['Capture']<=m)&(dforiglimpo2['Finalizado']==1)]
               
               #Seleção dos dados referente ao voo que será previsto
               subdfpred=dforiglimpo2.loc[(dforiglimpo2['chave2'].values==g)&(dforiglimpo2['Capture']==m)]

               #Concatenação dos dados históricos com os dados que serão previstos
               subdf1=pd.concat([subdfpred,subdf1])

               #Filtro de linhas com as mesma caracteristica da linha selecionada anteriormente, porem com NDO = -1 para definição da Variavel Y
               subdf2=dforiglimpo2.loc[(dforiglimpo2['NDO']==-1) & (dforiglimpo2['Dep_Date'].values<b) & (dforiglimpo2['Dest']==j) & 
                                   (dforiglimpo2['Periodo']==d)&(dforiglimpo2['Nest']==h)]
               
               #Sepação dos dados que serão usados como Variavel Y nas duas Previsões (Ocupação e Rask)
               y=subdf2[['chave2','Ocupacao', 'Rask']]

               #Merge das informações X e Y
               subdf3=pd.merge(subdf1,y,how='left',on='chave2')

               #Definição das Variaveis que serão apagadas do modelo
               subdf_Ocupacao=subdf3[[  'Event_Name', 'EvDef',  'DOW','Month', 'Description', 'chave2', 'Ocupacao_x', 'RPKs', 'Booked','Revenue',  'Finalizado', 'Ocupacao_y', ]]     

               subdf_Ocupacao=subdf_Ocupacao.set_index('chave2')


               N_Linhas=len(subdf_Ocupacao)
          
               #Definição de Base de Teste
               pred1=subdf_Ocupacao.loc[(subdf_Ocupacao['Finalizado']!=1)]
               
               #Definição de Base de Treino
               subdf_Ocupacao=subdf_Ocupacao.loc[(subdf_Ocupacao['Finalizado']==1)]  
               
               
               #Definição de x e y para o modelo - Previsão Ocupação
               y= np.array(subdf_Ocupacao['Ocupacao_y'])
               x=subdf_Ocupacao.drop(['Ocupacao_y','Finalizado'], axis = 1) 
               pred1=pred1.drop(columns=['Ocupacao_y','Finalizado'], axis = 1)
               x_list = list(x.columns)
               x = np.array(x)
               pred1 = np.array(pred1)

               
               

               #Definição de x e y para o modelo - Previsão Rask
               subdf_Rask=subdf3[[ 'Event_Name', 'EvDef', 'DOW', 'Month', 'Description', 'Nest', 'chave2', 'Rask_x', 'RPKs', 'Booked', 'CNXBooked', 'Revenue', 'Finalizado','Rask_y' ]]

               subdf_Rask=subdf_Rask.set_index('chave2')
               #Definição de Base de Teste
               pred2=subdf_Rask.loc[(subdf_Rask['Finalizado']!=1)]
               
               #Definição de Base de Treino
               subdf_Rask=subdf_Rask.loc[(subdf_Rask['Finalizado']==1)]  


               y2=np.array(subdf_Rask['Rask_y'])
               x2= subdf_Rask.drop(['Finalizado','Rask_y' ], axis = 1) 
               pred2=pred2.drop(columns=['Finalizado','Rask_y'], axis = 1)
               x_list2 = list(x2.columns)
               x2 = np.array(x2)
               pred2 = np.array(pred2)

               #Uso de Try para prevenção de erros de execução devido a falta de dados históricos no voo que causam erro no modelo
                              
               #Modelo Usado - Random Forest Regressor
               model =RandomForestRegressor()

               # Treino e Previsão da Variavel Ocuçpação
               model.fit(x, y)
               model_pred = model.predict(pred1)

               #Definição de métricas para Avaliação do Modelo
               z= model.predict(x)
               r2 = model.score(x, y)
               MAE= metrics.mean_absolute_error(y, z)
               MSE= metrics.mean_squared_error(y, z)
               RMSE=np.sqrt(metrics.mean_squared_error(y, z))   

               # Treino e Previsão da Variavel Rask
               model.fit(x2, y2)
               model_pred2 = model.predict(pred2)

               #Definição de métricas para Avaliação do Modelo
               z2= model.predict(x2)
               r22 = model.score(x2, y2)
               MAE2= metrics.mean_absolute_error(y2, z2)
               MSE2= metrics.mean_squared_error(y2, z2)
               RMSE2=np.sqrt(metrics.mean_squared_error(y2, z2))  

               base.append([g,b,c,d,h,a,N_Linhas,e,np.round(r2,3),np.round(MAE,5),model_pred[0],n,np.round(r22,3),np.round(MAE2,5),model_pred2[0]])
     except:

               base.append([g,b,c,d,e,h,a,N_Linhas,0,0,0,0,0,0,0])


     warnings.simplefilter("ignore")
    


#Construção de Base para Avaliação do Modelo

teste=dforiglimpo2[dforiglimpo2['NDO']==-1]
teste=teste[['chave2','Ocupacao','Rask']]
base2=pd.DataFrame(base, columns=['chave2','Dep_Date','Segment','Periodo','Nest','NDO',
                                  'total_hist','Ocupacao','r2_Ocupação','MAE_Ocupação','Predict_Ocupação',
                                  'Rask','r2_Rask','MAE_Rask','Predict_Rask'])


base2=pd.merge(base2,teste,how = 'left',on='chave2')

erros2=len(base2[base2['Predict_Ocupação']==0])
erros=len(base2[base2['Predict_Ocupação']==0])/len(base2)

base2['Ocupacao_x']=base2['Ocupacao_x']*100
base2['Predict_Ocupação']=base2['Predict_Ocupação']*100
base2['MAE_Ocupação']=base2['MAE_Ocupação']*100
base2['Ocupacao_y']=base2['Ocupacao_y']*100

base2['Assertividade'] = (base2['Predict_Ocupação']-base2['Ocupacao_y'])
# base2['Assertividade'] = (base2['Predict_Ocupação']/base2['Ocupacao_y']-1)*100
base2['Assertividade2'] = (base2['Predict_Rask']/base2['Rask_y']-1)*100

base2['Assertividade']=base2['Assertividade'].fillna(0)
base2['Assertividade2']=base2['Assertividade2'].fillna(0)

base2['X']=''
base2=base2[['chave2','Dep_Date','Segment','Periodo','Nest','NDO',
        'total_hist','Ocupacao_x','r2_Ocupação','MAE_Ocupação',
        'Predict_Ocupação','Ocupacao_y','Assertividade','X',
        'Rask_x','r2_Rask','MAE_Rask','Predict_Rask','Rask_y','Assertividade2']]


#saida da base de dados
base2.head(10) #Defina Aqui o local onde o dataset de Saida deverá ser salvo

