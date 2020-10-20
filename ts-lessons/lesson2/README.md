# typescript

## Lesson 2

### Создание проекта с поддержкой typescript

```bash
npm init -y
npm install typescript
npx tsc --init
mkdir src
```
поправить tsconfig.json
 - "include": ["src"]
 - "exclude": ["node_modules]
 - "outDir": "./dist"
 - "target": "es2019"
 - "moduleResolution": "node",
 - "module": "commonjs"
 - "strictPropertyInitialization": false,

установить:
```bash
npm i --save @types/node 
```

### План
 - установка typescript
 - конфигурация
 - примеры кода

sourcemaps
debug
TODO: declare

=========

strictNullChecks
noImplicitThis
strictFunctionTypes


