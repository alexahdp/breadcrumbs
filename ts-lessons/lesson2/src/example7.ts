// in
type Status = 'loading' | 'success' | 'error';

type State = {
  [k in Status]?: boolean;
}

// Pick
interface User {
  id: string;
  email: string;
  age: number;
}

type partUser = Pick<User, 'id' | 'email'>;

type Pick1<T, P> = {
  [p in P]: T[p]; // error
};
type Pick2<T, P extends keyof T> = {
  [p in P]: T[p];
};

// ===

type Readonly1<T> = {
  readonly [k in keyof T]: T[k];
}

type Partial2<T> = {
  [k in keyof T]?: T[k];
}

