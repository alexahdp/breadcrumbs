type Language = 'JavaScript' | 'TypeScript' | 'Python';
function setLanguage(language: Language) { /* ... */ }
setLanguage('JavaScript'); // ok
let language = 'JavaScript';
setLanguage(language); // error

// испрпавления:
// 1) let language: Language = 'JavaScript';
// 2) const language = 'JavaScript';

// еще один пример

// Параметр является парой (latitude, longitude).
function panTo(where: [number, number]) { /* ... */ }
panTo([10, 20]); // ok
const loc = [10, 20];
panTo(loc); // error:  Аргумент типа 'number[]' не может быть назначен // параметру типа '[number, number]'.

// исправления:
// 1) const loc: [number, number] = [10, 20];
// 2) const loc = [10, 20] as const;