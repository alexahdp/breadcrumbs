// ReturnType

export const CREATE_USER = 'CREATE_USER';
export const createUser = (username: string) => ({
  action: CREATE_USER,
  payload: { username },
});

export const DELETE_USER = 'DELETE_USER';
export const deleteUser = (username: string) => ({
  action: DELETE_USER,
  payload: { username },
});

export type userActionType =
  ReturnType<typeof createUser> |
  ReturnType<typeof deleteUser>;
