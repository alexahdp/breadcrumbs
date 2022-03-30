/**
 * Алгоритм Нарайана - позволяет определить следующую лексикографическу последовательность
 * - найти с конца i-ый элемент такой, что a[i] < a[i+1]
 * - найти с конца j-ый элемент такой, что a[i] < a[j]
 * - обменять местами a[i] и a[j]
 * - начиная с a[i+1] отсортировать элементы в обратном порядке
 * Сложность: O(n) ?
 */

const findI = (seq) => {
  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i + 1] && seq[i] < seq[i+1]) return i;
  }
  return -1;
}

const findJ = (seq, value) => {
  // seq.findIndex((v, i, self) => v > self[i]);
  for (let i = seq.length - 1; i >= 0; i--) {
    if (seq[i] > value) return i;
  }
  return -1;
}

function naraian(strseq) {
  const seq = strseq.split('');
  if (seq.length < 2) return seq.join('');

  const i = findI(seq);
  if (i == -1) return null;

  const j = findJ(seq, seq[i]);
  if (j == -1) return null;

  const box = seq[i];
  seq[i] = seq[j]
  seq[j] = box;

  if (i >= seq.length - 2) return seq.join('');

  return [...seq.slice(0, i+1), ...seq.slice(i+1).reverse()].join('');
}

const assert = ((a, b) => a === b || console.log(`ERROR: ${a} is not equal ${b}`));

// assert(naraian(''), '');
// assert(naraian('1'), '1');
// assert(naraian('12'), '21');
// assert(naraian('123'), '132');
// assert(naraian('132'), '213');
// assert(naraian('213'), '231');
// assert(naraian('231'), '312');
// assert(naraian('312'), '321');
// assert(naraian('12345'), '12354');

assert(naraian('132'), '213');