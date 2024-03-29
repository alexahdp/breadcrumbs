# Архитектура приложений

Главная проблема в разработке ПО - борьба со сложностью.


## Типовые архитектуры приложений

### MVC
Этому архитектурному паттерну уже довольно много лет (он был впервые предложен еще в 1979 году). Суть его очень проста: разделение зон ответственности на три типа:
 - **model** - инкапсулирует в себе бизнес-логику, взаимодействие с хранилищем данных, валидацию данных    
 - **view** - ответственен за представление данных (в случае веб-приложений - рендеринг html-страниц)  
 - **controller** - отвечает за общение с реальным миром (в случае веб-приложений - прием запросов, отправка ответов, взаимодействие с транспортным протоколом)  
Важные особенности:
 - каждая view принадлежит только одному контроллеру  
 - у одного контроллера может быть множество view  
 - контроллер обращается к модели для получения данных, после чего передает эти данные view для формирования ответа пользователю  

Проблемные места:
 - MVC-паттерн довольно просто, но в то же время, недостаточно гибок. Скажем, в нем не предусмотрено переиспользование представлений или логики моделей, хотя и можно выносить отдельный код в общие библиотеки. Это послужило предпосылкой создания другого паттерна - HMVC  
 - в этом паттерне очень тяжело организовать взаимодействие между view и model, что является очень необходимым, например, во фронтенде, где view может генерировать действия для изменения модели, а изменение модели, в свою очередь, должно приводить к изменению view  

### HMVC
Hierarchical-model-view-controller. Суть данного паттерна в том, чтобы организовать триплеты MVC в переиспользуемые группы. Это позволяет лучше следовать принципу DRY, но по-прежнему не решает проблемы взаимодействия view и model.  

### MVP - Model-vew-presenter
Этот паттерн получил широкое распространение во фронтенд-разработке, где view способен генерировать действия, эти действия уходят в presenter, он, в свою очередь, на основании действия решает как должны измениться данные и представление. Таким образом, view и model оказываются изолированными друг от друга. 

### MVVM model-view-viewModel
 - один viewModel связан только с одним view  
 - логика, которая могла бы оказаться во view должна быть размещена в viewModel  
 - связывание данных viewModel с данными, попадающими во view (концепция двустороннего data-binding)  

### Model-view-presenter-viewModel
> TODO

### Трехслойная архитектура
 * представление/клиент
  - обладает простыми функциями, можешь заниматься простой валидацией, взаимодействие с пользователем, подготовка данных в соответствующий для пользователя формат  
 * бизнес-логика  
  - на этом слое сосредоточена основная логика работы приложения, обработка данных. Слой бизнес-логики не должен зависеть от слоя представления  
 * слой данных  
  - обеспечение хранения и доступа к данным  

![Трехслойная архитектура](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/CSD_SCHEME.png/1920px-CSD_SCHEME.png)

### Луковичная архитектура
Основные принципы:
 - слои могут зависеть от внутренних, но не могут от внешних  
 - доменная модель всегда находится в центре архитектуры  
 - на внешнем слое располагаются: UI, инфраструктура, тесты  
 - инверсия зависимостей является ключевой особенностью данной архитектуры  
 - слой данных располагается не в центре, а вынесен в инфраструктуру  

![onion architecture](https://lh3.googleusercontent.com/proxy/R4pUOS0BaoYiEpKVTJa-nigNipMhDvBvg_0b-lH7rSXoQudvGJNko2pFeBJQWP4xvF82kQOviQQ81nZz3zXUB9bJaED7vT2wjZuFmJ66Kc0-NRb43qYjcohsjCw1Ke8)

### Гексагональная архитектура
Является логичным продолжением луковичной архитектуры и разделяет ui, инфраструктуру и данные.
Все эти сущности являются внешними по отношению к слою бизнес-логики и их можно разделить на порты и адаптеры.

![Гексагональная архитектура](https://hsto.org/webt/wp/gk/2w/wpgk2wxy5fgyjtrwuzctapvv19y.png)

В ходе своего развития приложения имеют свойство вырастать до значительных размеров. Они аккумулируют в себе большее количество бизнес-логики и сущностей и, в итоге, превращаются в то, что мы называем **монолитом**. Монолитом они становятся потому, что теперь бизнес-логика и/или слой данных стали слишком большими и сильносвязными и любые правки становятся слишком сложными и дорогостоящими. Еще одна проблема монолитов - масштабирование. Монолитные приложения масштабируются целиком, что является довольно дорогостоящей задачей, а иногда и вовсе невыполнимой. Отсюда появился новый виток в разработке ПО - микросервисы.

## Links
 * https://herbertograca.com/2017/08/17/mvc-and-its-variants  
 * https://habr.com/ru/post/500072/  
