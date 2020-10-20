# Паттерны проектирования

## Ссылки
[https://refactoring.guru/ru/design-patterns/catalog](https://refactoring.guru/ru/design-patterns/catalog)  
[https://habr.com/post/210288/](https://habr.com/post/210288/)  


## Заметки

### Порождающие
  - fabric  
  - abstract fabric  
  - builder  
  - prototype (rect.clone)  
  - singleton  

### Структурные
  - adapter  
  - bridge (вынесение части свойств в отдельный класс)  
  - composite (rect.getText())  
  - decorator  
  - facade (e.js)  
  - flyweight (page.styles, defaults)  
  - proxy (log-middleware)  

### Поведенческие
  - chain of responsibility (отдаленно похоже на jquery chains)  
  - command (redux)  
  - iterator (rect.deep)  
  - mediator  (ukit-client-side)  
  - memento (снимок состояния объекта)  
  - observer (подписки на события)  
  - state (объект хранит ссылку на один из объектов класса состояний, делегируя часть логики в него)  
  - strategy (в атрибуте - объект одного из классов ответственных за специфическую работу)  
  - template-method (в родительском классе алгоритм, вызывающий методы, в дочерних - переопределяются методы)  
  - visitor (для обработки объектов класса им добавляется метод, который принимает экземпляр visitor-а и вызывает у него нужный метод)  
