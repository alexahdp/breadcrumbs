// приведение типов в ts и js
// union-типы

// неверно
function getAge(number: string | number) {
  return  number as number;
}

// верно
function getAge2(number: string | number) {
  return  Number(number);
}