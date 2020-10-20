// types space and values space

interface Cat {
  length: number;
  meow: () => void;
}

const Cat = {
  length: 100,
  meow() {
    console.log('give me foood!');
  }
};

// typeof hell

interface Person {
  name: string;
  age: number;
}

const alex: Person = {
  name: 'alex',
  age: 30,
};

type Staff = typeof alex; // - type Person
const jstype = typeof alex; // - 'object'

// class - значение и тип одновременно
// & и | - операторы над множествами
// ! - ненулевое утверждение типа
// const - постоянное значение в категориях типов
// extends расширения множества значений данного типа
// in - отображение типа

// type и interface существуют только в пространстве типов
