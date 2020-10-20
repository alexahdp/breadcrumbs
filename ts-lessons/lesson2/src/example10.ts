// вывод типов
// typescript имеет неплохую систему выведения типов

interface Product {
  id: string;
  name: string;
  price: number;
}

// не надо так писать
function logProduct(product: Product) {
  const id: string = product.id;
  const name: string = product.name;
  const price: number = product.price;
  console.log(id, name, price);
}

// так лучше
function logProduct2(product: Product) {
  const { id, name, price } = product;
  console.log(id, name, price);
}

// в данном случае нет необходимости писать тип base
function parseString(str: string, base = 10) {}

// в случе, когда мы описываем объектный литерал, который должен реализовывать
// какой-то интерфейм, имеет смысл его аннотировать для того, чтобы выполнялась проверка полей
const person: Person = {
  name: 'alex',
  age: 10,
};