/*
Перемножить две строки, представляющие собой ОЧЕНЬ большие числа
в виде строк
*/
function multiply(a, b) {
  const elemsA = a.split('').reverse();
  const elemsB = b.split('').reverse();

  const results = [];
  let shift = 0;
  for (elemB of elemsB) {
    let res = '0'.repeat(shift);
    let mem = 0;

    for (elemA of elemsA) {
      const m = Number(elemA) * Number(elemB) + mem;
      mem = Math.floor(m / 10);
      res = res + (m - mem * 10).toString();
    }
    if (mem > 0) {
      res = res + mem.toString().split('').reverse().join('');
    }

    results.push(res);
    shift++;
  }

  const MAX = Math.max(...results.map(r => r.length));
  let result = '';
  let mem = 0;
  for (let i = MAX - 1; i >= 0; i--) {
    let sum = 0;
    results.reduce((sum, v) => sum += Number(v[i] || 0), 0)
    sum += mem;
    
    mem = Math.floor(sum / 10);
    result = (sum - mem * 10).toString() + result;
  }
  if (mem > 0) {
    result = result + mem.toString().split().reverse().join();
  }
  return result;
}

console.assert(multiply('3', '4'), String(3 * 4));
console.assert(multiply('4', '10'), String(4 * 10));
console.assert(multiply('22', '4'), String(22 * 4));
console.assert(multiply('28', '2'), String(28 * 2));
console.assert(multiply('589', '2'), String(589 * 2));
console.assert(multiply('589', '25'), String(589 * 25));
console.assert(multiply('58923', '320025'), String(58923 * 320025));
