// примитивные типы


// === 1 ===
let name: string = 'Alex';
name = 100; // error

let sum = 100; //  тип выводится автоматически

// простановка автоматически выводимых типов в таких простых случаях -
// признак начинающего разработчика на ts




// === 2 ===
let city = 'new york city';
console.log(city.toUppercase()); // error, toUppercase




// === 3 ===
const countries = [
  {name: 'Russia', capital: 'Moscow'},
  {name: 'Gonduras', capital: 'Tegucigalpa'},
];
for (const country of countries) {
  console.log(country.capitol); // error
}
// TODO: добавить type Country



// === 4 ===
// проверка возвращаемых типов
function isObject(user: { name: string }): boolean {
  return typeof user; // error
} 



// === 5 ===
function getUser(name: string)/*: string*/ {
  return { name };
}
const newUser: string = getUser('alex');



// === 6 ===
const log1 = (msg: string): void => {
  console.log(msg);
}



// === 7 ===
type User = {
  name: string;
}
const user = JSON.parse('{ "name": "alex" }') as User;
// работает только если тип был any



// === 8 ===
// never - пустой тип, которому не может быть присвоено значение и он не содержит значений
// (более подробно будет рассмотрен позднее)
function error(message: string): never {
  throw new Error(message);
}
// бесконечные циклы
// рекурсии
