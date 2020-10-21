// сгенерировать все правильные скобочные последовательности
// со скобками одного вида
// - 

function gen(n, counterOpen, counterClose, str) {
	if (counterOpen + counterClose === 2 * n) {
		return str;
	}
	
	if (counterOpen < n) {
		return gen(n, counterOpen + 1, counterClose, str + '(');
	}

	if (counterOpen > counterClose) {
		return gen(n, counterOpen, counterClose + 1, str + ')');
	}
}

function check(str) {
	let counter = 0;
	str.split('').forEach(item => {
		(item === '(') ? ++counter : --counter;
	});
	if (counter !== 0) throw new Error('Wrong sequence');
}

const N = 3;
const data = new Array(N * 2);
const res = gen(N, 0, 0, '');
check(res);
console.log(res);

