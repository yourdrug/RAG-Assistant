---
name: fastapi-ddd-layering
description: >
  Подробный чек-лист и примеры code smells для проверки соблюдения слоёв
  Domain-Driven Design (domain, application, infrastructure,
  interfaces/api) в FastAPI-проектах. Использовать при архитектурном
  ревью: поиск утечек инфраструктуры/фреймворка в домен, анемичной
  модели, нарушений границ агрегатов, неправильного использования
  Repository/Unit of Work, размытых bounded context.
license: MIT
---

# Слои и допустимое направление зависимостей

```
interfaces/api  --->  application  --->  domain  <---  infrastructure
   (FastAPI            (use cases,        (entities,      (реализация
    роутеры,            command/query      value objects,  репозиториев,
    Pydantic            handlers,          domain events,  ORM, HTTP-
    request/response    порты-интерфейсы)  бизнес-правила) клиенты LLM/
    схемы)                                                 vector DB)
```

Правило: стрелка зависимости всегда указывает **внутрь**, к domain.
`infrastructure` зависит от `domain` (реализует его интерфейсы), а не
наоборот. `domain` не зависит ни от чего снаружи.

# Чек-лист

## 1. Домен не знает о фреймворке
- [ ] В `domain/**` нет импортов `fastapi`, `pydantic` (v1/v2 BaseModel),
      `sqlalchemy`, `httpx`, SDK клиентов LLM/векторных БД.
  - Исключение: `pydantic.dataclasses`/`attrs` как замена обычным
    dataclass для value objects — приемлемо, если не тянет FastAPI-специфику.
- [ ] ORM-модели (`Base`, `Column`, `relationship`) не используются как
      domain-сущности напрямую — есть явный маппинг ORM-модель ↔ domain-entity
      в infrastructure-слое (repository).

## 2. Application-слой как оркестратор, а не место для бизнес-правил
- [ ] Use case / application service вызывает методы агрегата
      (`aggregate.do_something()`), а не реализует бизнес-логику сам
      (`if aggregate.status == "x" and aggregate.count > 5: ...` прямо в
      сервисе — признак того, что правило должно жить в агрегате).
- [ ] Application-сервис зависит от **интерфейсов** репозиториев/портов
      (`AbstractDocumentRepository`), а не от конкретных реализаций
      (`PostgresDocumentRepository`) — конкретная реализация внедряется
      через DI в composition root.

## 3. Entity vs Value Object
- [ ] Entity: имеет identity (`id`), может изменяться со временем,
      сравнение по `id`.
- [ ] Value Object: не имеет identity, неизменяем (`frozen=True` /
      `@dataclass(frozen=True)`), сравнение по значению всех полей,
      не имеет "сеттеров".
- Code smell: класс называется `...Value`/`...VO`, но имеет мутирующие
  методы или суррогатный `id` — переосмыслить как Entity либо убрать
  мутируемость.

## 4. Aggregate boundaries
- [ ] Один агрегат = одна транзакционная единица согласованности
      (consistency boundary). Изменение агрегата B как побочный эффект
      изменения агрегата A должно идти через domain event (eventual
      consistency), а не в одной SQL-транзакции "заодно".
- [ ] Репозиторий выдаёт/сохраняет **целый агрегат**, а не отдельные
      части его внутреннего состояния (`repo.update_line_item(...)`
      в обход aggregate root — antipattern: обходит инварианты).
- Специфично для RAG-проекта: если "База знаний" (KnowledgeBase) —
  агрегат, а "Документ"/"Чанк" — его часть, изменения чанков должны идти
  через методы `KnowledgeBase`, а не через прямой `ChunkRepository.save()`
  в обход инвариантов агрегата (например, лимит числа документов).

## 5. Repository и Unit of Work
- [ ] Интерфейс репозитория объявлен в `domain` (или `application`), а
      реализация — в `infrastructure`.
- [ ] Методы репозитория говорят на языке домена
      (`find_active_subscription_for(user)`), а не ORM/SQL
      (`select_where_status_eq_active(user_id)`).
- [ ] Транзакционные границы управляются Unit of Work (context manager),
      а не разбросанными `session.commit()` по всему application-слою.
- [ ] Нет прямого использования `Session`/`AsyncSession` в application-слое
      в обход репозитория "для скорости".

## 6. Domain events
- [ ] События поднимаются из домена (`aggregate.pull_events()`), а не
      конструируются вручную в инфраструктуре "по мотивам" произошедшего.
- [ ] Обработчики событий не создают циклическую зависимость между
      bounded contexts (Context A публикует событие → обработчик в
      Context B → событие обратно в Context A).

## 7. Bounded context и anticorruption layer
- [ ] Если в проекте несколько bounded context (например, `billing`,
      `knowledge_base`, `assistant`/`chat`) — интеграция между ними идёт
      через явные контракты (application service чужого контекста, DTO),
      а не через прямой импорт `from other_context.domain.models import X`.
- [ ] Для интеграции с внешними системами (LLM-провайдер, векторная БД,
      сторонние API) есть anticorruption layer, переводящий внешнюю модель
      данных в термины домена — а не "протекание" внешнего SDK-объекта до
      самого домена.

## 8. Признаки анемичной модели (anemic domain model)
- Класс-агрегат состоит только из полей + `get_*`/`set_*` без поведения.
- Вся логика "что можно, а что нельзя" сосредоточена в application-сервисах
  или, того хуже, в FastAPI-роутерах.
- Валидация инвариантов происходит "снаружи" (в сервисе) после того, как
  объект уже создан в невалидном состоянии.

## 9. God services
- Application-сервис/модуль > ~300-400 строк с более чем 3-4
  несвязанными ответственностями — кандидат на разбиение по
  Command/Query handlers (по одному use case на класс/функцию).

# Как оформлять находки

Каждая находка должна ссылаться на конкретный файл и, по возможности,
короткую цитату проблемного места (не длиннее одной строки кода), плюс
1-3 конкретных шага рефакторинга — не общие рассуждения про DDD.
