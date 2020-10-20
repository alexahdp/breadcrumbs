"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
require("reflect-metadata");
const class_validator_1 = require("class-validator");
const class_transformer_1 = require("class-transformer");
const express_1 = __importDefault(require("express"));
const body_parser_1 = __importDefault(require("body-parser"));
var Role;
(function (Role) {
    Role["admin"] = "admin";
    Role["user"] = "user";
})(Role || (Role = {}));
class User {
}
__decorate([
    class_validator_1.IsString(),
    class_validator_1.IsNotEmpty(),
    __metadata("design:type", String)
], User.prototype, "email", void 0);
__decorate([
    class_validator_1.IsEnum(Role),
    class_validator_1.IsNotEmpty(),
    __metadata("design:type", String)
], User.prototype, "role", void 0);
const users = [
    { email: "alex@gmail.com", role: Role.admin },
];
const app = express_1.default();
app.use(body_parser_1.default());
app.get('/user', (req, res) => {
    res.json(users);
});
app.post('/user', (req, res) => {
    const user = class_transformer_1.plainToClass(User, req.body);
    console.log(user);
    console.log(Reflect.getMetadataKeys(user));
    // console.log(Reflect.(user, 'email'))
    const err = class_validator_1.validateSync(user);
    if (err.length > 0) {
        res.statusCode = 400;
        return res.json(err);
    }
    users.push(user);
    res.json(users);
});
app.listen(4000, () => {
    console.log('server is listening port 4000');
});
