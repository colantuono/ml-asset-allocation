# Machine Learning Asset Allocation with Agnostic Priors using Brazilian Stocks

> João Pedro Colantuono  

---

Esse trabalhho é sobre _Alocação de Ativos com Aprendizado de Máquina_.  
Alguns dos portfólios que apresentarei são _Paridade de Risco Hierárquico_ e _PCA-EIgen_, pensando como fundo Long Only de Ações.
O objetivo é fornercer opções aos métodos tradicionais de alocação de ativos. 


Função Pipeline: 
    A Função pipeline é o coração desse trabalho.
    Ela recebe todo o histórico de preços, transforma preços em retornos, filtra os ativos que não existiam naquele periodo.
    Após isso, ela entra os retornos filtrados no algoritmo de alocação, calcula o retorno daquela alocação para o retorno do mês _Out-of-Sample_ e itera essa entrada N vezes, até que os dados sejam esgotados.
    Ao fim, ela retorno um dataframe com o retorno global para cada periodo.
    

<!-- Links -->

[LinkedIn](https://www.linkedin.com/in/jpcolantuono/)  
  
[GitHub](https://github.com/colantuono)