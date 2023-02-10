# PROCESSAMENTO DE LINGUAGEM NATURAL (PLN): O QUE É E COMO UTILIZO PARA A TRADUÇÃO DE TEXTOS INGLÊS-PORTUGUÊS?

## Objetivo
  Facilitar a compreensão sobre o Processamento de Linguagem Natural (PLN), como ela pode ser utilizada no cotidiano e demonstrar as etapas necessárias para a construção de um tradutor inglês-português utilizando da linguagem de programação python, modelos pré-treinados por meio do Hugging Face e a comunidade de cientistas de dados, Kaggle.
  
  
## Introdução
  Ao tentar resolver um problema com alguma empresa e se deparou com um assistente virtual, utilizou um tradutor para entender algum texto em uma linguagem que nem mesmo você conseguia reconhecer ou até mesmo usou a predição de textos enquanto teclava em seu celular, saiba que está fazendo uso de aplicações construídas a partir do Processamento de Linguagem Natural (PLN). Mas, o que seria o PLN?
  
  Sendo um subcampo das áreas de inteligência artificial, ciência da computação e linguística, o processamento de linguagem natural (PLN) se deu em meados da Segunda Guerra Mundial com o intuito de interceptar e descriptografar mensagens, é o estudo da problemática sobre a interpretação e manipulação da linguagem humana a partir de uma máquina, com o intuito de entender as emoções humanas e seus recursos linguísticos, como a raiva, sarcasmo e entre outros, utilizando-se da fala e de textos escritos reunindo uma série de técnicas empregadas na compreensão e geração automática de textos. No entanto, raramente a linguagem humana é escrita ou falada de forma correta, o entendimento da linguagem humana não é apenas baseado nas palavras, mas sim em seu real significado e em seus conceitos, mas a capacidade da linguagem possuir mais do que uma interpretação se torna um problema difícil para as máquinas, sendo até hoje um problema não solucionado, principalmente quando estamos a tratar da análise semântica. A priori, as aplicações de PLN eram construídas diretamente no código fonte de um programa sem a geração dos dados em tempo de execução, normalmente extraídos a partir do conhecimento linguístico. No entanto, com a chegada de técnicas mais avançadas sobre machine learning e deep learning, acabou se tornando comum a utilização dessas técnicas em aplicações PLN. O PLN é um método que está sofrendo um alto crescimento à medida que seus métodos são utilizados em novas tecnologias principalmente na área econômica, social e científica a partir da interação humano-computador.
  
  Segundo Searle um computador não é capaz de interpretar ou pensar. E essa afirmação foi feita seguindo o experimento do quarto chinês, onde existe uma pessoa dentro de um quarto, e esse quarto não existe visão para o lado de fora, apenas um dicionário chinês e dois buracos nas paredes, uma para entrada de textos em chinês e outro para a saída do texto em outra língua. O experimento consta que, o homem dentro da sala pode até estar traduzindo as entradas, mas ele não está aprendendo realmente a traduzir do chinês para outra língua.

## Tradução inglês-português
  As arquiteturas de transformadores facilitaram a construção de modelos de maior capacidade e o pré-treinamento tornou possível utilizar efetivamente essa capacidade para uma ampla variedade de tarefas. Transformers é uma biblioteca de código aberto com o objetivo de abrir esses avanços para a comunidade de aprendizado de máquina mais ampla. A biblioteca consiste em arquiteturas Transformer de última geração cuidadosamente projetadas em uma API unificada. A arquitetura do transformers é dimensionada com dados de treinamento e tamanho do modelo que facilitam o treinamento paralelo eficiente, junto ao seu pré-treinamento que permite o treino com longos corpus para facilitar a adaptação em tarefas com forte desempenho.Corpus é ‘Um conjunto de dados lingüísticos (pertencentes ao uso oral ou escrito da língua, ou a ambos), sistematizados segundo determinados critérios, suficientemente extensos em amplitude e profundidade, de maneira que sejam representativos da totalidade do uso lingüístico ou de algum de seus âmbitos, dispostos de tal modo que possam ser processados por computador, com a finalidade de propiciar resultados vários e úteis para a descrição e análise’ (Sanchez, 1995, pp. 8-9)
  
  Em processamento de linguagem natural, a tarefa de traduzir é realizada ao converter uma sequência de texto de um idioma para outro. Tradução é um modelo sequence-to-sequence que utiliza as partes de codificação e decodificação dentro da arquitetura do Transformer. As camadas de atenção do codificador podem acessar todas as palavras da frase inicial, enquanto as camadas de atenção do decodificador podem acessar apenas as palavras posicionadas antes de uma determinada palavra na entrada.
  
  ### Pré-processamento com tokenizer
  Os modelos Transformer não podem processar texto bruto diretamente, portanto, a primeira etapa do nosso pipeline é converter as entradas de texto em números que o modelo possa entender utilizando o tokenizer. O tokenizer irá dividir a entrada em tokens e os mapear para inteiros. Assim que a tokenização esteja completa, receberemos um dicionário pronto para alimentarmos o modelo.
  
  ### Modelos Transformers
  Os modelos Transformer só aceitam tensores como entrada, então é necessário converter a lista de tokens em tensores. Para cada entrada de modelo, recuperaremos um vetor de alta dimensão representando o entendimento contextual dessa entrada pelo modelo Transformer. Se diz um vetor de alta dimensão pois a saída vetorial do módulo Transformer geralmente é grande.

  As cabeças do modelo pegam o vetor de alta dimensão de estados ocultos como entrada e os projetam em uma dimensão diferente. A saída do modelo Transformer é enviada diretamente para o cabeçote do modelo para ser processado.
  
  ![image](https://user-images.githubusercontent.com/113546603/217971793-40026979-7412-419e-8cb4-443e0557a486.png)
  
  A camada de agrupamento do modelo converte cada token de entrada em um vetor que representa o token associado. As camadas subsequentes manipulam esses vetores usando o mecanismo de atenção para produzir a representação final das frases.
  
  ### Pós-processamento
  Normalização das pontuações brutas, emitidas pela última camada do modelo, para a verificação de probabilidades.
  
  ### Pipeline
  Ter que fazer o pré-processamento, passar as entradas pelo modelo e o pós-processamento é algo que demanda muito tempo. No entanto, o pipeline é uma função do Transformer que já engloba todos esses passos.
  
  ![image](https://user-images.githubusercontent.com/113546603/217970972-e6b02359-abe8-4565-8a8e-d192ce6f57f3.png)

    
## Conclusão
  O PLN é uma área que ainda tem muito o que crescer e agregar em nosso cotidiano, mesmo tendendo ao impossível a questão da livre interpretação e manipulação da linguagem a partir de uma máquina. Mas, se demonstrou bem perceptível a facilidade do acesso a aplicações PLN e como as desenvolver de maneira fácil. O uso de modelos pré-treinados facilita o processo de tokenizar, modelo e pós-processamento com a utilização do Transformers e pipeline.
  
## Código
```python
#Importando a função "pipeline" da biblioteca python que servirá para
#construir o pré-processamento, alimentar o modelo e
#finalizar o pós-processamento
from transformers import pipeline


def translator(frase, modelo):
    translate = pipeline("translation", model=modelo)
    return translate(frase)


model_checkpoint = "Helsinki-NLP/opus-mt-tc-big-en-pt"
raw_frase = str(input("Entre com uma frase: "))
translator(frase=raw_frase, modelo=model_checkpoint)
```
 
## Referências
  Transformers: Processamento de linguagem natural de última geração (Wolf et al., EMNLP 2020)
  
  PLN: o que é Processamento de Linguagem Natural?. Alura, 2020. Disponível em: https://www.alura.com.br/artigos/o-que-e-pln
  
  O quarto chinês: o que significa ter uma mente?. Filosofianaescola, 2019. Disponível em: https://filosofianaescola.com/metafisica/quarto-chines/
  
  Translation. Hugging Face. Disponível em: https://huggingface.co/docs/transformers/tasks/translation
  
  The Tatoeba Translation Challenge – Realistic Data Sets for Low Resource and Multilingual MT. ACL Anthology, 2020. Disponível em: https://aclanthology.org/2020.wmt-1.139/
