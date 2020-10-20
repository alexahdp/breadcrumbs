// примитивные типы
// any
// void

let name: string = 'Alex';
name = 100; // error


let sum = 100; //  тип подставляется автоматически

function getName(key: string): string {
  return { name: key }; // error
}


function isValid(email: string): boolean {
  return typeof email; // error
} 


const log1 = (msg: string): void => {
  console.log(msg);
}

let age: number;
age = '12'; // error
// age = '12' as any;

// разница между
let user1 = 'username';
const user2 = 'username';

type User = {
  name: string;
}
const user = JSON.parse('{ "name": "alex" }') as User;

// noImplicitAny
function add(a, b) {
  return a + b; }
}
add(null, 3);


// never - пустой тип, которому не может быть присвоено значение и он не содержит значений
// (более подробно будет рассмотрен позднее)
function error(message: string): never {
  throw new Error(message);
}
// бесконечные циклы
// рекурсии
