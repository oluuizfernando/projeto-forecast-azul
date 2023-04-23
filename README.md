Sobre o Projeto

Este projeto foi a partir de uma parceiria educacional entre a Escola Dinâmica Group (DNC) e a Empresa Azul Linhas Aéreas

Este é um projeto de MAchine Learning direcionado para o desenvolvimento de um cistema de Forecast para prever a ocupação e a receita estimada de cada voo futuro.

Para Realizar as previsões foi utilizado a biblioteca Scikit-learn, algoritmo Randomforest, este modelo foi selecionado por ter atingido os melhores métricas nos treinos.

as métricas avaliadas foram:

R2;
MAE;
Assertividade Geral (Soma das previsões / soma dos valores reais);
% de Linhas como erro médio de até 10% (Para mais ou para menos);

O Modelo usa como referencias as variaveis de entrada : 'Event_Name', 'EvDef', 'DOW', 'Month', 'Description', 'Nest', 'chave2'(Desenvolvida no projeto), 'Rask', 'RPKs', 'Booked', 'CNXBooked', 'Revenue', 'Finalizado'(Desenvolvida no projeto), Ocupacao

e Tambem usa as seguintes variaveis como referencia para dataset de saida: 'Dep_Date','Segment','Periodo','Nest','NDO','total_hist' (Desenvolvida no Projeto),'r2_Ocupação' (calculada),'MAE_Ocupação' (calculada),'Predict_Ocupação' (calculada), 'r2_Rask' (calculada),'MAE_Rask' (calculada),'Predict_Rask' (calculada)

No desenvolvimento da etapa de Data Preparation foi fonciderado uma reconstrução do dataset seguindo os seguintes passos:

Seleção uma data Capture
Seleção de uma linha dos voos filtrados a partir da data de Capture selecionado
Seleção de variaveis especificas que serão usadas para filtrage de dados históricos dessa linha selecionada
filtro dos dados históticos
reconstrução do dataset considerando os dados históricos e o o voo selecionado.
separação da Base em Treino e teste, sendo que todos os dados históricos serão considerados como treino e apenas a linha do voo selecionado inicialmente sera usada como teste
aplicação do modelo nos dados de treino
predição dos dados de teste
salvar os dados de saida em uma variavel

Todos esses passos serão utilizados em um laço FOR finalizansdo o processo apenas quando todas as linhas selecionadas no filtro inicial forem finalizadas.

Apos a consluão do laço for uma base de dados será gerada com os resultados das previsões e asmetricas calculadas.
Essa base de dados deverá ser salva no local escolhido pela empresa.

É importante resaltar que o tempo de processamento do modelo não esta satisfatório, podendo demorar algumas horas para rodar todos os dados selecionados.


Além do código do modelo propriamente dito, tambem estamos entregando um app desenvolvido na biblioteca Streamlit que tem como objetivo proporcionar a projeção futura das duas variaveis previstas com mais opções de filtros, reduzindo a quantidade de dados e consequentemente reduzindo o tempo de processamento, o objetivo dessa entrega é proporcionar maior flexibilidade para o dia a dia da Empresa.




abaixo segue a referencia para criação do ambiente virtual para o projeto:


conda create -n Azul_DNC python=3.9.13
conda activate Azul_DNC
pip install -r requirement.txt


   


