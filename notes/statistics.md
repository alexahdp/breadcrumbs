# Статистика - лекции

Виды формирования выборок из генеральной совокупности: 
 - simple random sample
 - stratified sample
 - cluster sample

Стандартная ошибка выборочного среднего(центральная предельная теорема):
При извлечении некоторой выборки из генеральной совокупности выборочное среднее окажется в той же точке с ошибкой равной:
Se = sigma/sqrt(n)

Z-нормировка:
Z = (xs - m) / se
xs - выборочное среднее
m - среднее генеральной совокупности


Для оценки среднего генеральной совокупности
95% точность: Xs = m +/- 1.96 se 
99% точность: Xs = m +/- 2.58 se
m - среднее по выборке
se - стандартная ошибка

Если число наблюдений невелико и sigma неизвестно, то используется распределение стьюдента
Форма распределения определяется числом степеней свободы
Число степеней свободы - количество элементов, которые могут варьироваться при расчете статистического показателя
df = n - 1
t=(x_ - m) / (sd / sqrt(n))

Sst - sum squares total
Sum(a[i] - m)^2
Ssw- внутригрупповая изменчивость
Sum(a[i] - m)^2
- Вычисляется для подгруппы

Ssb - межгрупповая изменчивость
Количество степеней свободы = n - поправка Бенферони - позволяет скорректировать уровень значимости для большого количества выборок

Критерий Тьюки альтернатива методу стьюдента 

Дисперсионный анализ

Коррекционный анализ Пирсона стоит использовать при нормальном распределении и линейности данных

Коэффициент корреляции Спирмена - менее чувствителен к ненормальности и нелинейность данных 

Регрессионный анализ
 - метод наименьших квадратов

*Коэф детерминации* - доля дисперсии зависимой переменной объясняемая регрессионной моделью

Если две переменные количественные, то стоит использовать корреляционный анализ или линейную регрессию. Если же одна из переменных номинативная, то стоит использовать t-тест или дисперсионный анализ.