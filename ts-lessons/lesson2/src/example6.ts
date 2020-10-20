// Exclude

type A = 'a' | 'b';

type B = Exclude<A, 'a'>;

const b: B = 'b';
