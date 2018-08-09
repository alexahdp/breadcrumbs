/**
 * На вход функции поступает массив с вложенными массивами.
 * Нужно выполнить слияние всех массивов в один массив,
 * при этом нельзя использовать рекурсию, т.к. вложенные массивы
 * могут быть вложены друг в друга ОЧЕНЬ глубоко
 * @param {Array} arr
 * @return {Array}
 */
function flatten(arr) {
	let i = 0;
	let n = arr.length;
	let curArr = arr;
	
	const result = [];
	const stack = [];
	
	if (n == 0) return [];
	
	do {
		const item = curArr[i];
		
		if (Array.isArray(item)) {
			if (i + 1 < n) stack.push({i, n, curArr});
			curArr = item;
			i = 0;
			n = item.length;
			
		} else {
			result.push(item);
			i++;
		}
		
		if (i == n && stack.length > 0) {
			const state = stack.pop();
			i = state.i + 1;
			n = state.n;
			curArr = state.curArr;
		}
		
	} while (i < n || stack.length > 0);
	
	return result;
}

function test(testData, truthData) {
	console.log(JSON.stringify(testData) == JSON.stringify(truthData) ? 'ok' : 'error');
}

test(flatten([1, 2, 3, [4, 5]]), [ 1, 2, 3, 4, 5 ]);
test(flatten([[], 1, 2, 3, [4, 5]]), [ 1, 2, 3, 4, 5 ]);
test(flatten([[0], 1, 2, 3, [4, 5], 6]), [ 0, 1, 2, 3, 4, 5, 6 ]);
test(flatten([[0], 1, 2, 3, [4, 5, [6, [], 7]], 8]), [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ]);
test(flatten([[0], 1, 2, 3, [4, 5, [6, [[], 7], 8]], 9, []]), [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]);
