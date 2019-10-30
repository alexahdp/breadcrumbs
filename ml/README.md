# ml
Machine learning notes and examples

## Теория
 - [Основные термины и определения](./glossary/common.md)  
 - [Оптимизаторы](./glossary/optimizers.md)
 - [Классические методы машинного обучения](./glossary/classics.md)  
 - [Нейронные сети](./glossary/neural_networks.md)  
 - [Natural language processing](./glossary/nlp.md)  
 - [Известные архитектуры и их краткое описание](./glossary/architectures.md)  
 - [Планирование и проведение эксперимента](./glossary/experiment_planning.md)
 - [Ссылки](./glossary/links.md)  


## Практика
1. [pandas, numpy](./lesson1/index.md), [google apps datastore analyze](./lesson1/index2.ipynb), [scipy.polyfit](./lesson1/scipy.ipynb)  
2. [pyplot](./lesson2/index.md)  
3. -- [LinearRegression, RandomForestRegression, RandomForestClassifier, GridSearchCV](./lesson3/index.ipynb)--  
4. [KNN, cross_validation](./lesson4/index.ipynb)  
5. [Classification (LogisticRegression, KNN, SVC)](./lesson5/index.ipynb)  
6. [TSNE, KMeans](./lesson6/index.ipynb) - работа про boston dataset, можно объединить  




### Titanic: machine learning for disaster
Стандартный и всеми избитый датасет с данными о выживших на Титанике. Задача классификации, необходимо предсказать кто из пассажиров выживет. Задача интересна тем, что необходимо догадаться извлечь из имени пол. Выживаемость пассажиров оказалась очень сильно скоррелированной с полом. Присутствует чистка данных, никакой особой визуализации.
**Использованные модели**:  
 - DecisionTreeClassifier  
 - RandomForestClassifier  
 - KNeighborsClassifier  
 - SVC  
 - Perceptron  
 - XGBClassifier  

Большая часть моделей была использована просто для эксперимента, лучшие результаты, как и ожидалось, у XGBClassifier. Также в данной работе присутствует использование ансамбля моделей через VotingClassifier.  

**TODO**:  
 - что там делаетLogisticRegression?
 - надо бы хоть какую-то виуализацию данных добавить
 - неплохо бы добавить использование CatBoost
 - подбор гиперпараметров


### Flat prices prediction
Задача регрессии. Присутствует исследование датасета, чистка данных, визуализация.  
**Использованные модели**:  
 - LinearRegression
 - RandomForestRegressor
 - KNeighborsRegressor
 - XGBRegressor  

Лучший результат оказался у XGBRegressor. К сожалению, отсутствует подбор гиперпараметров.  
[Kaggle](https://www.kaggle.com/alexahdp/flat-price-predict)  

**TODO**:  
 - в конце скрипт падает, надо поправить это  


### Boston house prices
Линейная регрессия для предсказания стоимости жилья. Никакого исследования данных, никакой чистки. Прменяется только LinearRegression и RandomForestRegressor. Довольно слабая работа, одна из самых первых.  
[Kaggle](https://www.kaggle.com/alexahdp/boston-house-prices-regression)  
