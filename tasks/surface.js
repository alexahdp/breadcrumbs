/**
 * На вход функции поступает множество точек.
 * Нужно проверить, существует ли прямая разделающая точки по оси X
 * на два симмметричных множества
 * Каждая точка задана как объект {x, y}
 * @param {Array} dots
 * @return {boolean} True - разделяема
 */
function isDivided(dots) {
	let min, max;
	
	dots.forEach(dot => {
		if (!min || min < dot.x) min = dot.x;
		if (!max || dot > max.x) max = dot.x;
	});
	
	const center = (max + min) / 2;
	
	const left = {};
	const right = {};
	dots.forEach(dot => {
		if (dot.x > center) {
			right[dot.x] = 1;
		}
		else if (dot.x < center) {
			left[dot.x] = 1
		}
	});
	
	return Object.keys(right).every(x => left[center - (x - center)]);
}

//true
console.log(isDivided([
	{x: -2}, {x: -1},
	{x: 2}, {x: 1},
]));

console.log(isDivided([
	{x: 2}, {x: 3},
	{x: 6}, {x: 5},
]));

//false
console.log(isDivided([
	{x: 0}, {x: 3},
	{x: 6}, {x: 5},
]));