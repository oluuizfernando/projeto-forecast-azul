# Elaboração de modelo de forecast para receita e ocupação de voos de companhia aérea.

Sobre o Projeto

Este projeto foi a partir de uma parceiria educacional entre a Escola Dinâmica Group (DNC) e uma companhia aérea parceira.

Este é um projeto de Machine Learning direcionado para o desenvolvimento de um cistema de Forecast para prever a ocupação e a receita estimada de cada voo futuro.

Para realizar as previsões foi utilizado a biblioteca Scikit-learn, algoritmo Randomforest, este modelo foi selecionado por ter atingido os melhores métricas nos treinos.

As métricas avaliadas foram:

R2;
MAE;
Assertividade Geral (Soma das previsões / soma dos valores reais);
% de Linhas como erro médio de até 10% (Para mais ou para menos);

O Modelo usa como referências as variáveis de entrada : 'Event_Name', 'EvDef', 'DOW', 'Month', 'Description', 'Nest', 'chave2'(Desenvolvida no projeto), 'Rask', 'RPKs', 'Booked', 'CNXBooked', 'Revenue', 'Finalizado'(Desenvolvida no projeto), Ocupacao

e também usa as seguintes variáveis como referência para o dataset de saída: 'Dep_Date','Segment','Periodo','Nest','NDO','total_hist' (Desenvolvida no Projeto),'r2_Ocupação' (calculada),'MAE_Ocupação' (calculada),'Predict_Ocupação' (calculada), 'r2_Rask' (calculada),'MAE_Rask' (calculada),'Predict_Rask' (calculada)

No desenvolvimento da etapa de Data Preparation foi considerado uma reconstrução do dataset seguindo os seguintes passos:

- Seleção de uma data Capture.
- Seleção de uma linha dos voos filtrados a partir da data de Capture selecionada.
- Seleção de variáveis específicas que serão usadas para filtragem de dados históricos dessa linha selecionada.
- Filtro dos dados históricos.
- Reconstrução do dataset considerando os dados históricos e o o voo selecionado.
- Separação da base em treino e teste, sendo que todos os dados históricos serão considerados como treino e apenas a linha do voo selecionado inicialmente será usada como teste
- Aplicação do modelo nos dados de treino.
- Predição dos dados de teste.
- Salvar os dados de saida em uma variável.

Todos esses passos serão utilizados em um laço FOR finalizando o processo apenas quando todas as linhas selecionadas no filtro inicial forem finalizadas.

Apos a conclusão do laço FOR, uma base de dados será gerada com os resultados das previsões e as métricas calculadas.
Essa base de dados deverá ser salva no local escolhido pela empresa.

É importante resaltar que o tempo de processamento do modelo não está satisfatório, podendo demorar algumas horas para rodar todos os dados selecionados.

Além do código do modelo propriamente dito, tambem entregamos um app desenvolvido na biblioteca Streamlit que tem como objetivo proporcionar a projeção futura das duas variáveis previstas com mais opções de filtros, reduzindo a quantidade de dados e consequentemente reduzindo o tempo de processamento, o objetivo dessa entrega é proporcionar maior flexibilidade para o dia a dia da empresa.

Abaixo segue a referência para criação do ambiente virtual para o projeto:

conda create -n Azul_DNC python=3.9.13
conda activate Azul_DNC
pip install -r requirement.txt


   


