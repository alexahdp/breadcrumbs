/**
 * Написать функцию, которая принимает на вход массив примитивов
 * и возвращает их все возможные перестановки
 * (наличие дублей не рассматривать)
 */

function transpose(arr) {
	const res = [];

	if (arr.length === 1) return arr;

	arr.forEach((item, i) => {
		const slicedArr = arr.filter((a, k) => k !== i);
		transpose(slicedArr).forEach(cc => {
			res.push(Array.isArray(cc) ? [item, ...cc] : [item, cc]);
		});
	});

	return res;
}

function check(arr1, arr2) {
	if (arr1.length !== arr2.length) throw new Error('Wrong length');

	if ( ! arr1.every((transposition1, i) => {
		const transposition2 = arr2[i];
		return transposition1.every((item1, j) => item1 === transposition2[j]);
	})) {
		console.log(arr1);
		console.log(arr2);
		throw new Error('Arrays are not equal')
	}

	return 'ok';
}

const res1 = transpose([1, 2]);
const res1Valid = [[1, 2], [2, 1]];
console.log(check(res1, res1Valid));

const res2 = transpose([1, 2, 3]);
const res2Valid = [
	[1, 2, 3],
	[1, 3, 2],
	[2, 1, 3],
	[2, 3, 1],
	[3, 1, 2],
	[3, 2, 1]
];

console.log(check(res2, res2Valid));

const res3 = transpose([1, 2, 3, 4]);
const res3Valid = [
	[1, 2, 3, 4],
	[1, 2, 4, 3],
	[1, 3, 2, 4],
	[1, 3, 4, 2],
	[1, 4, 2, 3],
	[1, 4, 3, 2],

	[2, 1, 3, 4],
	[2, 1, 4, 3],
	[2, 3, 1, 4],
	[2, 3, 4, 1],
	[2, 4, 1, 3],
	[2, 4, 3, 1],

	[3, 1, 2, 4],
	[3, 1, 4, 2],
	[3, 2, 1, 4],
	[3, 2, 4, 1],
	[3, 4, 1, 2],
	[3, 4, 2, 1],

	[4, 1, 2, 3],
	[4, 1, 3, 2],
	[4, 2, 1, 3],
	[4, 2, 3, 1],
	[4, 3, 1, 2],
	[4, 3, 2, 1],
];

console.log(check(res3, res3Valid));