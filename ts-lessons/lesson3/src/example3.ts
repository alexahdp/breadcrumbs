type Language = 'JavaScript' | 'TypeScript' | 'Python';

interface GovernedLanguage {
  language: Language;
  organization: string;
}

function complain(language: GovernedLanguage) { /* ... */ }

complain({ language: 'TypeScript', organization: 'Microsoft' }); // ok

const ts = {
  language: 'TypeScript',
  organization: 'Microsoft',
};

complain(ts);

// ~~ Аргумент типа '{ language: string; organization: string; }'
// не может быть назначен параметру типа 'GovernedLanguage'.
// Типы свойства 'language' несовместимы.
// Тип 'string' не может быть назначен для типа 'Language

// решение
// const ts = {
//   language: 'TypeScript' as const,
//   organization: 'Microsoft',
// };