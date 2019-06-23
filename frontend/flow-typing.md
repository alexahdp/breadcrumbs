# Flow-конспект

## Как задать точную форму объекта, чтобы flow ругался на наличие лишних свойств?
```javascript
type user = {|
	name: string,
	age: number
|};

const alex: user = {
	name: 'Alex',
	age: 29,
}
```
Теперь если в alex окажутся лишние поля, flow выдаст ошибку.

## Как описать метод у функции?
```javascript
type funcCall = {
	ok: (cb) => void,
	// $call: () => void, - deprecated
	[[call]]: () => void,
};

function hello() {}
hello.ok = ok;
```

## Как создавать объекты с произвольным набором полей?
```javascript
type objMap = {
	[key: string]: number,
};

const usersMap = {};

usersMap.alex = 21;
usersMap.bob = 12;
```

## Как объединить два типа?
```javascript
type nameType = {
	name: string,
};

type ageType = {
	age: number,
};

type userType = nameType & ageType;
const alex: userType = {
	name: 'alex',
	age: 29,
};
```

## Как объединить два типа с пересекающимися полями?
```javascript
type nameType = {
	name: string,
};

type ageType = {
	age: number,
};

// $Exact предотвращает появление mixed-полей
type userType = {
	...$Exact<nameType>,
	...$Exact<ageType>,
};

const alex: userType = {
	name: 'alex',
	age: 29,
};
```

## Приведение к типу (кастование)
```javascript
(u: user)
((u: empty): user)
```


## Disjoint unions - выведение свойств для объединенных типов
```javascript
type successResponseT = {|
	success: true,
	data: {
		meta: string,
	},
|};

type errorResponseT = {|
	success: false,
	error: string,
|};

type Response = successResponseT | errorResponseT;

function handleResponse(r: Response) {
	if (response.error) {
		console.log(response.data.meta.includes('something'));
	}
}
```

## structured typing и nominal typing
structured typing - можно значению одного составного типа присводить значение другого составного типа при
условии, что у них совпдают вложенные типы  
nominal typing - при присвоении типы должны строго совпадать (как в java)  


## замена типов
 - **Вариантность** — перенос наследования исходных типов на производные от них типы  
 - **ковариантность** - перенос наследования исходных типов на производные от них типы в прямом порядке. Аргументы функции должны быть ковариантны  
 - **контрвариантность** - перенос наследования исходных типов на производные от них типы в обратном порядке. Возвращаемые функцией значение должны быть контрвариантны  
 - **инвариантность** - строгость типа, без возможности замены  


##  Как вывести тип из javascript-переменной?
```javascript
const languages = {
	ru: 'ru.json',
	en: 'en.json',
	de: 'de.json,
};

type Language = $Keys<typeof languages>;
let lang: Language = 'ru';
```
 - тут надо понимать, что typeof выполняется на этапе статического анализа, а не непосредственно js-ом
 Также можно выводить типы из значений:
 ```javascript
 // Object.freeze внутри комментария с двоеточием необходим для того, чтобы freeze происходил на этапе стат анализа
 // чтобы не просаживать производительность js
 const languages = /*::Object.freeze*'/({
	ru: 'ru.json',
	en: 'en.json',
	de: 'de.json,
}/*::*/);

type LanguageFile = $Values<typeof languages>;
let i18n: LanguageFile = 'ru.json';
 ```
 
 ## Как вывести тип с необязательными полями из другого типа?
 ```javascript
 const User = {
 	name: string,
	age: number,
 };
 
 // во втором аргументе теперь все поля из типа User являются опциональными
 function update(u: User, data: $Shape<User>) {...}
 
 update({name: 'alex', age: 29}, {age: 30});
 ```
 
 ## Как получить свойство типа-объекта в виде отдельного типа?
```javascript
type User = {
	name: string,
	contacts: {
		email: string,
		phone: string,
	}
};

type Contact = $PropertyType<User, 'contacts'>;
const contact: Contact = {email: 'aaa@gmail.com', phone: '1234'};
```
$PropertyType принимает в качестве второго параметра только строку, переменную ему передать нельзя.
Эту проблему решает $ElementType


