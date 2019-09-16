# ml
Machine learning notes and examples

## Теория
 - [Основные термины и определения](./glossary/common.md)  
 - [Оптимизаторы](./glossary/optimizers.md)
 - [Методы регрессии](./glossary/regression.md)  
 - [Методы классификации](./glossary/classification.md)  
 - [Методы кластеризации](./glossary/clasterization.md)  
 - [Деревья принятия решений](./glossary/decision_trees.md)  
 - [Ссылки](./glossary/links.md)  


## Практика
1. [pandas, numpy](./lesson1/index.md), [google apps datastore analyze](./lesson1/index2.ipynb), [scipy.polyfit](./lesson1/scipy.ipynb)  
2. [pyplot](./lesson2/index.md)  
3. [LinearRegression, RandomForestRegression, RandomForestClassifier, GridSearchCV](./lesson3/index.ipynb)  
4. [KNN, cross_validation](./lesson4/index.ipynb)  
5. [Classification (LogisticRegression, KNN, SVC)](./lesson5/index.ipynb)  
6. [TSNE, KMeans](./lesson6/index.ipynb)  
7. [Decision Tree vs KNN](./lesson7/index.ipynb)  
8. [Предсказание стоимости квартир (RandomForest, LinearRegression, KNN)](./lesson8/index.ipynb)  


## Исследование данных и построение модели
### Правильный путь
1. Почистить данные от пропусков  
2. Попробовать обучить на данных простейшие модели, обязательно при этом фиксировать результаты  
3. После этого можно начинать исследовать данные с помощью графиков и чистить их, сверяясь с результатами  
4. Не забываем про масштабирование данных  
5. Можно попробовать выбросить некоторые столбцы/данные из датасета  
6. Можно попробовать заняться feature-engineering-ом и достроить новые столбцы с данными  
Иногда можно даже попробовать обучить дополнительную модель для построение дополнительных столбцов  


### Неправильный путь
 - не нужно сначала всесторонне и тщательно строить десятки графиков. Возможно,
они и не понадобятся, но времени отнимут массу  
- всегда нужно ориентироваться на конкретные показатели и оценивать точность модели  


===

