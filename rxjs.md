# Работа с rxjs
Проверить код можно на: https://codepen.io
В настройках необходимо подключить: https://unpkg.com/rxjs/bundles/rxjs.umd.min.js

## Пример1
(По клику выводить числа начиная с нуля, каждый клик сбрасывает вывод)
const button = document.getElementById('test');
rxjs.fromEvent(button, 'click')
  .pipe(rxjs.operators.switchMap(() => rxjs.interval(1000)))
  .subscribe(t => console.log(t))
