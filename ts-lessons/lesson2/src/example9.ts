// сигнатуры индексов
// парсинг csv-файла

interface Vocabulary {
  [ket: string]: string | number;
}

type Dict = Record<'name' | 'email', string | number>;
const user: Dict = {
  name: 'alex',
  email: 'alex@gmail.com',
};
