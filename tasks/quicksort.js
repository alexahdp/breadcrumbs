function sort(arr) {
  const baseIndex = Math.round(Math.random() * (arr.length - 1));
  const base = arr[baseIndex];

  let ltBase = arr.filter((v, i) => v <= base && i !== baseIndex);
  let gtBase = arr.filter(v => v > base);

  if (ltBase.length > 1) {
    ltBase = sort(ltBase);
  }

  if (gtBase.length > 1) {
    gtBase = sort(gtBase);
  }

  return [...ltBase, base, ...gtBase];
}

function equalArrays(arr1, arr2) {
  if (arr1.length !== arr2.length) return false;
  return arr1.every((n, i) => arr2[i] === n);
}

const test1 = [4, 2, 6, 6, 7, 10, 1, 23, 14];
const res = sort(test1);
console.log((equalArrays(res, [1, 2, 4, 6, 6, 7, 10, 14, 23])));