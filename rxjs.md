# Работа с rxjs
Проверить код можно на: https://codepen.io
В настройках необходимо подключить: https://unpkg.com/rxjs/bundles/rxjs.umd.min.js

## Пример1
(По клику выводить числа начиная с нуля, каждый клик сбрасывает вывод)
```javascript
const button = document.getElementById('test');
rxjs.fromEvent(button, 'click')
  .pipe(rxjs.operators.switchMap(() => rxjs.interval(1000)))
  .subscribe(t => console.log(t))
```
То же самое, но с использованием switchMapTo:
```javascript
const button = document.getElementById('test');

const interval$ = rxjs.interval(1000);
rxjs.fromEvent(button, 'click')
  .pipe(rxjs.operators.switchMapTo(interval$)
  .subscribe(t => console.log(t))
```

## Пример2
Изначально в консоль выводятся числа, при клике по кнопке stop выводи прекращается
```javascript
const stop = document.getElementById('stop');
const interval$ = rxjs.interval(1000);

const subscription = interval$
   .subscribe(x => console.log(x));

rxjs.fromEvent(stop, 'click')
  .subscribe(() => {
    subscription.unsubscribe();
  });
```

То же самое, но с использованием takeUntil:
```javascript
const stop = document.getElementById('stop');
rxjs.interval(1000)
  .pipe(rxjs.operators.takeUntil(rxjs.fromEvent(stop, 'click')))
  .subscribe(x => console.log(x));
```

## Пример3
При клике по кнопке start начать вывод в консоль, по stop - закончить
```javascript
const start = document.getElementById('start');
const stop = document.getElementById('stop');

const start$ = rxjs.fromEvent(start, 'click');
const stop$ = rxjs.fromEvent(stop, 'click');
const interval$ = rxjs.interval(1000);

const intervalThatStops$ = interval$
  .pipe(rxjs.operators.takeUntil(stop$));

start$
  .pipe(rxjs.operators.switchMapTo(intervalThatStops$))
  .subscribe(t => console.log(t));
```

## Пример4
Добавим сохранение предыдущего состояние с помощью метода .scan(
```javascript
const start = document.getElementById('start');
const stop = document.getElementById('stop');

const start$ = rxjs.fromEvent(start, 'click');
const stop$ = rxjs.fromEvent(stop, 'click');
const interval$ = rxjs.interval(1000);

const intervalThatStops$ = interval$
  .pipe(rxjs.operators.takeUntil(stop$));

start$
  .pipe(rxjs.operators.switchMapTo(intervalThatStops$))
  .pipe(rxjs.operators.scan(acc => ({count: acc.count + 1}), {count: 0}))
  .subscribe(t => console.log(t));
```

## Пример5
Добавим кнопку reset
```javascript
const {startWith, switchMapTo, mapTo, scan, takeUntil} = rxjs.operators;

const start = document.getElementById('start');
const stop = document.getElementById('stop');
const reset = document.getElementById('reset');

const start$ = rxjs.fromEvent(start, 'click');
const stop$ = rxjs.fromEvent(stop, 'click');
const reset$ = rxjs.fromEvent(reset, 'click');

const interval$ = rxjs.interval(1000);

const intervalThatStops$ = interval$
  .pipe(takeUntil(stop$));

start$
  .pipe(switchMapTo(rxjs.merge(
    intervalThatStops$.pipe(mapTo(acc => ({count: acc.count + 1}))),
    reset$.pipe(mapTo(acc => ({count: 0})))
  )))
  .pipe(startWith({count: 0}))
  .pipe(scan((acc, curr) => curr(acc)))
  .subscribe(t => console.log(t));
```

## Пример6
Добавим возможность изменять частоту вывода сообщений
```javascript
const {startWith, switchMap, switchMapTo, mapTo, scan, takeUntil} = rxjs.operators;

const start = document.getElementById('start');
const stop = document.getElementById('stop');
const reset = document.getElementById('reset');
const half = document.getElementById('half');
const quarter = document.getElementById('quarter');

const start$ = rxjs.fromEvent(start, 'click');
const stop$ = rxjs.fromEvent(stop, 'click');
const reset$ = rxjs.fromEvent(reset, 'click');
const half$ = rxjs.fromEvent(half, 'click');
const quarter$ = rxjs.fromEvent(quarter, 'click');

rxjs.merge(
    start$.pipe(mapTo(1000)),
    half$.pipe(mapTo(500)),
    quarter$.pipe(mapTo(250))
  )
  .pipe(switchMap(t => rxjs.merge(
    rxjs.interval(t).pipe(takeUntil(stop$)).pipe(mapTo(acc => ({count: acc.count + 1}))),
    reset$.pipe(mapTo(acc => ({count: 0})))
  )))
  .pipe(startWith({count: 0}))
  .pipe(scan((acc, curr) => curr(acc)))
  .subscribe(t => console.log(t));

```
