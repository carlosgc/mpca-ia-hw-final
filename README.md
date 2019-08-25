# Trabalho Final de IA
Inteligência Artificial do curso de Mestrado Profissional em Computação Aplicada

# *Artigo*
https://www.overleaf.com/read/tznpqrzgfbqt

# *Metas*

Este trabalho irá utilizar uma Rede Convolucional Profunda de uma dimensão (1D CNN) para identificar movimentos de mãos através sinais de sensores inerciais.

Será utilizada a base de dados criada para o manuscrito "Gesture Recognition System for Real-time Mobile Robot Control Based on Inertial Sensors and Motion Strings", publicado em 2017 por Ivo Stančić, Josip Musić e Tamara Grujić. 

Será analisado o desempenho de uma 1D CNN na classificação desses sinais e sua performance será comparada com os resultados de outros métodos de aprendizado de máquinas, utilizados para classificar a mesma base de dados.

# *Abordagem*

A base de dados consiste em dados de 20 voluntários que realizaram uma sequência de nove movimentos com sensores acoplados no dedo e antebraço. Cada voluntário participou de 10 sessões de coleta e, no total, 1800 gestos foram registrados.

As imagens abaixo mostram os movimentos realizados pelos voluntários e o dispositivo utizado para capturar os sinais gerados por eles.

![Gestos realizados pelos voluntários](https://raw.githubusercontent.com/carlosgc/mpca-ia-hw-final/master/images/hand_gestures.png)

![Dispositivo de captura](https://raw.githubusercontent.com/carlosgc/mpca-ia-hw-final/master/images/hand_sensors.png)

Os dados serão dividos em duas partes: 70% em dados de treinamento e 30% em dados de teste. A maior parte será usada para treinar a 1D CNN e a menor para testar sua acurácia.

A rede será implementada utilizando a ferrementa TensorFlow.

Será apresentado uma comparação da acurácia da 1D CNN e outros métodos de aprendizagem de máquina.

![Rede Convolucional Profunda de uma dimensão](https://raw.githubusercontent.com/carlosgc/mpca-ia-hw-final/master/images/1dcnn.png)

# *Bibliografia*

- Gesture recognition system for real-time mobile robot control based on inertial sensors and motion strings (Ivo Stančić, Josip Musić e Tamara Grujić, 2017)

- 1D Convolutional Neural Networks and Applications: A Survey (Serkan Kiranyaz, Onur Avci, Osama Abdeljaber, Turker Ince, Moncef Gabbouj, Daniel J. Inman, 2019)

- Real-Time Patient-Specific ECG Classification by 1D Convolutional Neural Networks (Serkan Kiranyaz, Turker Ince, Moncef Gabbouj, 2015)
