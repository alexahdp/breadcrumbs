/**
 * Есть матрица 2n-1 x 2n - 1, заполненная случайными значениями
 * Надо вывести их в ряд, начиная из центра по спирали: влево - вниз, вправо - вверх, ...
 * Решение должно быть общим для любого n
 */

function getFlat(arr) {
	const result = [];
	const c = Math.floor((arr.length) / 2)
	
	const pos = {x: c, y: c};
	const dir = {x: -1, y: 0};
	
	let stepsCount = 1;
	let steps2 = 2;
	let stepsDone = 0;
	
	let angle = Math.PI;
	const stepsAll = arr.length ** 2;
	
	for (i = 0; i < stepsAll; i++) {
		console.log(pos, dir, stepsDone, stepsCount);
		result.push(arr[pos.y][pos.x]);
		
		pos.x += dir.x;
		pos.y += dir.y;
		
		stepsDone++;
		
		if (stepsDone == steps2) {
			stepsCount++;
			steps2 = stepsCount * 2;
			stepsDone = 0;
		}
		else if (stepsDone == stepsCount) {
			angle += Math.PI / 2;
			dir.x = Math.round(Math.cos(angle));
			dir.y = -Math.round(Math.sin(angle));
		}
		
	}
	console.log(result);
}

const matrix = [
	[8, 7, 6],
	[1, 0, 5],
	[2, 3, 4]
];

getFlat(matrix);

//console.log(getFlat(matrix));