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
