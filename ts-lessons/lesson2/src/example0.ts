// классы

class User1 {
  // по-умолчанию все свойства публичные
  name: string;
  email: string;
  age: number;
}
const user1 = new User1();

class User2 {
  constructor(public name: string, public email: string, public age: number) {}
}
const user2 = new User2('alex', 'alex@gmail.com', 30);

class User3 {
  isActive: boolean = false;
  constructor(public name: string, private email: string, public age: number) {}

  getEmail() {
    this.activate();
    return this.email;
  }

  private activate() {
    this.isActive = true;
  }
}
const user3 = new User3('alex', 'alex@gmail.com', 30);
user3.getEmail();
// user3.activate();

// классы реализуют интерфейсы
// классы остаются на этапе исполнения
