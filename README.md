# MPI Matrix Multiplication in Blocks
Este projeto implementa a multiplicação de matrizes em blocos, utilizando a biblioteca mpi4py para paralelismo com MPI. O código divide as matrizes em blocos, distribui o trabalho entre processos MPI e realiza a multiplicação de forma paralela, permitindo processamento mais eficiente de matrizes grandes.

## Pré-requisitos
Antes de começar, você precisará dos seguintes pacotes instalados:

- Python 3.x: Certifique-se de ter o Python instalado em seu sistema.
- MPI (Message Passing Interface): Um ambiente MPI como OpenMPI ou MPICH deve estar instalado. (Utilizei https://learn.microsoft.com/pt-br/message-passing-interface/microsoft-mpi, no windows 10)
- mpi4py: A biblioteca Python para MPI.

## Partes do trabalho
O trabalho está dividido em quatro partes principais, sendo que cada uma delas representa uma etapa de evolução na implementação da multiplicação de matrizes distribuída usando MPI. As partes do trabalho são implementadas em arquivos diferentes:

Parte 1.py: Implementação Naive (Broadcast das Matrizes).
Parte 2.py: Primeira Evolução (Broadcast da Matriz B e Particionamento das Linhas de A).
Parte 3.py: Segunda Evolução (Particionamento das Matrizes A e B e Comunicação Assíncrona).
Parte 4 - Extra.py: Particionamento de Matrizes para Dados Grandes (Extra), separando-os em blocos.

## Arquivos
Você precisará de duas matrizes em formato de texto (matrizes A e B) para realizar a multiplicação. Essas matrizes devem estar salvas no diretório Matrizes/.
Exemplo de Estrutura:

```bash
├── Matrizes/
│   ├── Matriz5.txt
│   ├── Matriz6.txt
```

Informe o nome dos arquivos .txt contendo as matrizes logo no começo do código, nas variáveis *matrix_a* e *matrix_b* respectivamente.
Parte-se de premissa que serão informadas matrizes com as dimensões A = [m x n]  e B = [n x p]  para calculas uma matriz C = [m x p].

## Executando o Código
### Comando MPI
Para executar o código, você precisará utilizar o comando *mpiexec*. Este comando distribui o trabalho entre diferentes processos.

Exemplo de execução com 4 processos:
```bash
mpiexec -n 4 'Parte 3.py'
```

#### Output
Após a execução, o processo rank 0 (mestre) imprimirá o resultado da multiplicação da matriz.