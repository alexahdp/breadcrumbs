// сужение типов (стр 123)
// TODO непонятно что здесь имеется ввиду

interface User {
  name: string;
  email: string;
}

function isUser<T>(user: T | null): user is T {
  return !!user;
}

const users = [
  { name: 'alex', email: 'asd'},
  null, 
  { name: 'alex2', email: 'asd1'},
];

const knownUsers = users.filter(isUser);

// добавление свойств в объекты

const pt0 = {};
const pt1 = {...pt0, x: 3};
const pt: Point = {...pt1, y: 4}; // ok
