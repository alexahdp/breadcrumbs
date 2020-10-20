// more complex typings

type Exact<T> = {
  [P in keyof T]: T[P];
}

interface User {
  email: string;
}

const log = (u: Exact<User>) => {
  console.log(u);
}

log({ email: "dsfdsf" });
log({ email: "dsfdsf", name: "dsd" });