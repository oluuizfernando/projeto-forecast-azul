import streamlit as st

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import date

import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)





st.set_page_config(
     page_title='AZUL & DNC',
     page_icon='Imagens/logo.png'
    
     )



st.header('Forecast Azul Linhas Aereas')
# st.write('Bem vindo a Gerador de Forecast de voos AZUL!')
st.write('Defina os Filtros ao lado para iniciar o Processo:')

st.sidebar.image('Imagens/azul-logo.png')
st.sidebar.header('Filtros')


Data_app= st.sidebar.date_input('Informe a Data de Capture desejada')

Data_Inicio = st.sidebar.date_input('Data Inicio')

Data_Fim = st.sidebar.date_input('Data Final')

Origem_select = st.sidebar.text_input('Aéroporto de Origem')

Destino_select = st.sidebar.text_input('Aéroporto de Destino')

NDO_select = st.sidebar.text_input(label='Especifique um Número de NDO',)

Voo_select = st.sidebar.text_input('Especifique um Voo')

Porcentagem_select = st.sidebar.slider(label='Especifique um Percentual dos dados que deseja Gerar',min_value=0.01, max_value=1.00)




if st.button('Verificar nº de Predições'):
    tempo_inicial=(time.time())
    # Código do Modelo

    #Importando Dataset
    dforiglimpo2=pd.read_csv(r'C:\Users\bruno\OneDrive\Área de Trabalho\Curso Data Expert DNC\AZUL\Project_Azul_DNC\azul_dnc_forecast\data\processed\dforiglimpo2.csv')

    dforiglimpo2['Dep_Date']= dforiglimpo2['Dep_Date'].astype('datetime64')

    # Aplicação de filtros

    Data = str(Data_app.year)+'-'+('0'+str(Data_app.month) if Data_app.month <=9  else str(Data_app.month))+'-'+('0'+str(Data_app.day) if Data_app.day <=9 else str(Data_app.day) )

    Porcentagem = Porcentagem_select

    Voo_Select=Voo_select
    Voo_Geral = dforiglimpo2['FlightID']
    Voo = Voo_Geral if Voo_Select=="" else Voo_Select


    Origem_Select = Origem_select 
    Origem_Geral = dforiglimpo2['Orig'] 
    Origem = Origem_Geral if Origem_Select=="" else Origem_Select

    Destino_Select = Destino_select
    Destino_Geral = dforiglimpo2['Dest'] 
    Destino = Destino_Geral if Destino_Select=="" else Destino_Select


    NDO_Select = NDO_select if NDO_select=="" else int(NDO_select)
    NDO_Geral = dforiglimpo2['NDO'] 
    NDO =  NDO_Geral  if NDO_Select=="" else NDO_Select 


    Data_Inicio = Data_Inicio.strftime('%Y - %m - %d')

    Data_Fim = Data_Fim.strftime('%Y - %m - %d')
    
    # Seleção de um Dia especifico de Venda, representação de rodada do modelo em produção com dados em D-1
    dfTeste=dforiglimpo2.loc[(dforiglimpo2['Capture']==Data)&(
         dforiglimpo2['NDO']==NDO)&(dforiglimpo2['FlightID']==Voo)&
         (dforiglimpo2['Orig']==Origem)&(dforiglimpo2['Dest']==Destino)&
         (dforiglimpo2['Dep_Date']>=Data_Inicio)&
         (dforiglimpo2['Dep_Date']<=Data_Fim)
         ]
    dfTeste=pd.DataFrame(dfTeste).reset_index()
    x=int((dfTeste.shape[0])*Porcentagem)
    dfTeste=dfTeste.sample(x,replace=True)

    st.write(dfTeste.shape[0])
    tempo_final=(time.time())

    
    resumo=dfTeste.groupby(['Orig','Dest'])['FlightID'].count().reset_index().sort_values(by='FlightID', ascending=False)
    resumo=resumo.rename(columns={'FlightID':'Voos'})
    resumo=resumo.set_index('Voos')


    resumo1=dfTeste.groupby(['NDO'])['FlightID'].count().reset_index().sort_values(by='FlightID', ascending=False)
    resumo1=resumo1.rename(columns={'FlightID':'Voos'})
    resumo1=resumo1.set_index('Voos')


    st.write(resumo.head(5), resumo1.head(5))
    

   
        
    
    st.write(f'Tempo para carregar os dados = {(tempo_final-tempo_inicial)/60:,.2f} Minutos')

    
    st.write(f'Tempo estimado Para Predição = {((tempo_final-tempo_inicial)+(8*dfTeste.shape[0]))/60:,.2f} Minutos')
    
    





if st.button('Gerar Forecast!'): 

    tempo_inicial=(time.time())
    #Importando Dataset
    dforiglimpo2=pd.read_csv(r'C:\Users\bruno\OneDrive\Área de Trabalho\Curso Data Expert DNC\AZUL\Project_Azul_DNC\azul_dnc_forecast\data\processed\dforiglimpo2.csv')

    # Aplicação de filtros

    Data = Data_app

    Porcentagem = Porcentagem_select

    Voo_Select=Voo_select
    Voo_Geral = dforiglimpo2['FlightID']
    Voo = Voo_Geral if Voo_Select=="" else Voo_Select


    Origem_Select = Origem_select 
    Origem_Geral = dforiglimpo2['Orig'] 
    Origem = Origem_Geral if Origem_Select=="" else Origem_Select

    Destino_Select = Destino_select
    Destino_Geral = dforiglimpo2['Dest'] 
    Destino = Destino_Geral if Destino_Select=="" else Destino_Select


    NDO_Select = NDO_select if NDO_select=="" else int(NDO_select)
    NDO_Geral = dforiglimpo2['NDO'] 
    NDO =  NDO_Geral  if NDO_Select=="" else NDO_Select 


    
    # Seleção de um Dia especifico de Venda, representação de rodada do modelo em produção com dados em D-1
    dfTeste=dforiglimpo2.loc[(dforiglimpo2['Capture']==Data)&(dforiglimpo2['NDO']==NDO)&(dforiglimpo2['FlightID']==Voo)&(dforiglimpo2['Orig']==Origem)&(dforiglimpo2['Dest']==Destino)]
    dfTeste=pd.DataFrame(dfTeste).reset_index()
    x=int((dfTeste.shape[0])*Porcentagem)
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


#Modelo Final



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
        # f=lin['Flts']
        g=lin['chave2']
        h=lin['Nest']
        j=lin['Dest']
        k=lin['DOW']
        m=lin['Capture']
        n=lin['Rask']


        # print(i)

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
        


    # definição de Base de saida do Modelo

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
    

    

    st.write(base2.head(20))

    st.download_button(
         label='Download CSV', data=base2.to_csv(index=False),
         mime='texto/csv', file_name='forecast_AZUL_linhas_aereas.csv'
        )


    #Metricas Variavel Ocupação
    t_hist = base2['total_hist'].mean()
    st.write(f'Média de Linhas Usadas como Histórico: {t_hist:,.0f}')
    st.write(f'Total de linhas avaliadas: {len(base2)}')
    st.write(f'Total de Linhas com erros: {erros2}')
    st.write(f'% de erros de execução: {erros*100:,.1f}%')
    
    st.write('---------------------------------------------------------------------')


    st.write('**Metricas da Previsão da Variavel Ocupação**')

    R2=base2['r2_Ocupação'].sum()/len(base2)
    MAE=base2['MAE_Ocupação'].sum()/len(base2)
    prev=base2['Predict_Ocupação'].sum()
    real=base2['Ocupacao_y'].sum()
    Assertividade=((prev/real)*100)
    Assertividade1= base2.loc[(base2['Assertividade']>=-10) &(base2['Assertividade']<=10)] 
    Assertividade2=len(Assertividade1)/len(base2)
    st.write(f'R2 médio: {R2:,.2f}')
    st.write(f'MAE Médio: {MAE:,.2f}%')
    st.write(f'% Geral de Assertividade - {Assertividade:,.2f}%')
    st.write( f'% de linhas com erro de até 10% - {Assertividade2*100:,.2f}%') 
    
    
    # plt.figure(figsize=[5,5])
    sns.histplot(base2['Assertividade'],bins=50,kde=True)
    st.pyplot(plt)
    plt.clf()

    st.write('---------------------------------------------------------------------')

    # Metricas Variavel RASK

    st.write('** Metricas da Previsão da Variavel RASK (centavos por KM):')

    R22=base2['r2_Rask'].sum()/len(base2)
    MAE2=base2['MAE_Rask'].sum()/len(base2)
    prev2=base2['Predict_Rask'].sum()
    real2=base2['Rask_y'].sum()
    Assertividade2=((prev2/real2)*100)
    Assertividade1_2= base2.loc[(base2['Assertividade2']>=-10) &(base2['Assertividade2']<=10)] 
    Assertividade2_2=len(Assertividade1_2)/len(base2)
    st.write(f'R2 médio: {R22:,.2f}')
    st.write(f'MAE Médio: {MAE2:,.2f}')
    st.write(f'% Geral de Assertividade - {Assertividade2:,.2f}%')
    st.write( f'% de linhas com erro de até 10% - {Assertividade2_2*100:,.2f}%') 

    # plt.figure(figsize=[5,5])
    sns.histplot(base2['Assertividade2'],bins=50,kde=True, color='green')
    st.pyplot(plt)
    plt.clf()

    st.write('---------------------------------------------------------------------')



    tempo_final=(time.time())


    st.write(f'Tempo Real de Execução = {(tempo_final-tempo_inicial)/60:,.2f} Minutos')
