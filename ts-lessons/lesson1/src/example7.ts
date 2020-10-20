// отличия интерфейсов от типов
// 1) расширение: интерфейсы - implements, типы - &
// интерфейсы могут расширять типы, но не может расширять сложные типы, например, Union-type
interface IRootState extends State {
  square: number;
}
type TRootState = State & { square: number };

// implements работает как с классами, так и с типами
// type имеет больше возможностей, например:
type NamedVariable = (Input | Output) & { name: string };

// также декларации типов имеют свойство объединяться
interface Person {
  name: string;
}
interface Person {
  email: string;
}