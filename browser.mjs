// Сквозная проверка страницы в настоящем браузере.
//
// Зачем отдельно от tests.py: тесты на Python проверяют сервер и структуру
// скрипта, но не могут выполнить страницу. Ошибка, ради которой этот прогон и
// появился, была именно такой — «запуститьСценарий» оказался объявлен внутри
// другой функции. Синтаксис корректен, сервер отвечает, все 150 тестов зелёные,
// а кнопка «Спросить» падает с ReferenceError, и сценарий не запускается вовсе.
//
// Как запускать:
//   1) поднять сервер:   MEMORY_DIR=/tmp/проба PORT=5000 python web.py
//   2) поднять браузер:  google-chrome --headless=new --remote-debugging-port=9222 \
//                          --no-sandbox --disable-gpu about:blank
//   3) сам прогон:       node browser.mjs [адрес] [ключ-модели] [запрос] [режим]
//
// Режимы:
//   сценарий (по умолчанию) — запустить сценарий по триггеру и дождаться конца;
//   пауза                   — включить «подтверждать смену стадии», дождаться
//                             остановки, нажать «Продолжить» и убедиться, что
//                             задача сдвинулась;
//   переходы                — завести задачу, нажать запертую кнопку перехода и
//                             убедиться, что страница показывает разбор отказа,
//                             а после плана и утверждения переход открывается;
//   mcp                     — панель MCP: подключиться к своему серверу, затем ко
//                             всем; проверить карточки (соединение, сбой, пропуск),
//                             пометки доступа, закрытые фильтром инструменты и
//                             JSON-схему. Модель в этом режиме не вызывается.
//   инструменты             — День 17: включить инструменты, задать вопрос про
//                             трекер и увидеть строки вызовов «↳» под ответом;
//                             попросить изменение и убедиться, что появилась
//                             заявка, а вызова не было; подтвердить её и увидеть
//                             результат; вызвать инструмент кнопкой из панели MCP.
//                             Нужен поднятый мок: python tracker_api.py
//   планировщик             — День 18: панель планировщика. Поставить задание
//                             руками, увидеть его в расписании и пульс работника;
//                             собрать сводку кнопкой; попросить модель поставить
//                             напоминание; снять задание. Нужен поднятый мок
//                             трекера и, для «работник жив», запущенный
//                             python worker.py.
//   индекс                  — День 21: страница /rag. Таблица документов, поиск
//                             двумя стратегиями рядом, чанк с границами соседей,
//                             сравнение стратегий кнопкой, отказ на неверные
//                             параметры сборки. Индекс должен быть собран.
//   rag                     — День 22: карточки «Вопрос по документам» и
//                             «Контрольные вопросы» на /rag: два ответа рядом,
//                             источники и фрагменты, ловушка, прогон десяти
//                             вопросов с таблицей. Нужны индекс и ключ модели.
//   реранк                  — День 23: карточки «Поиск в два этапа» и «Подбор
//                             порога и top-K» на /rag. Режимы из rag-settings.json,
//                             отбор по этапам в базе и с фильтром (реранкер
//                             поднимает п. 2.2.7 из-за пределов top-5), rewrite,
//                             порог и K вживую, ответы в четырёх режимах и
//                             контрольные вопросы в пяти. Нужны индекс, подбор
//                             и ключ модели.
//   показ                   — День 23 для показа и записи экрана: семь сцен
//                             DEMO-CHROME.md — режимы, подбор, отбор на промахе
//                             Дня 22, rewrite, порог и K, ловушка, ответы во
//                             всех режимах, контрольные вопросы в пяти режимах.
//   показ-22                — День 22 для показа: шесть сцен — два режима на
//                             одном вопросе, фрагменты промпта, ловушка,
//                             контрольные вопросы.
//   показ-21                — День 21 для показа и записи экрана: семь сцен
//                             DEMO-CHROME.md — документы и OCR, сборка с ходом
//                             работы и кэшем, поиск и паспорт чанка, граница
//                             чанка, поиск по скану, сравнение, агент ищет в
//                             индексе через MCP-сервер docs.
//   показ-20                — День 20 для показа и записи экрана: те же шесть
//                             сцен, что в DEMO-CHROME.md, но медленно, с
//                             подписями внизу страницы и паузами для камеры.
//                             Ставится не вместо проверки, а вместо браузерного
//                             расширения, когда оно не подключается: сценарий
//                             проходит этот скрипт, а с экрана идёт запись.
//                             Отклонения не валят прогон — это показ, а не
//                             проверка; в конце печатается, что разошлось.
//   оркестр                 — День 20: панель «Оркестр». Посчитать маршрут
//                             кнопкой «Куда пойдёт?», пройти длинный флоу
//                             (план → вызовы к разным серверам → сверка), затем
//                             флоу с меняющим шагом: он должен остановиться на
//                             заявке, а после подтверждения — продолжиться
//                             кнопкой «Продолжить флоу».
//                             Нужен поднятый мок: python tracker_api.py
//   конвейер                — День 19: панель конвейеров. Выполнить цепочку
//                             кнопкой и убедиться, что под каждым шагом видно,
//                             ЧТО он получил от предыдущего; затем полная
//                             цепочка с файлом, сбой посреди цепочки и просьба
//                             к модели выполнить конвейер одним вызовом.
//                             Нужен поднятый мок: python tracker_api.py
//
// По умолчанию: http://127.0.0.1:5000, модель ds-flash. Прогон печатает, что
// появляется в чате, и — главное — исключения JS, которых в чате не видно.
//
// Ловушка, которая уже стоила времени: выражения для страницы собираются
// шаблонными строками, а в них «\s» — это не regexp-класс, а просто буква «s»
// (JS молча съедает неизвестный escape). «replace(/\s+/g, ' ')» внутри такого
// шаблона вырезал из текста все «s»: «tracker__list_issues» печатался как
// «tracker__li t_i ue », и проверка честно не находила задание. Внутри
// шаблонов escape удваивается: «/\\s+/g».

const АДРЕС = process.argv[2] || 'http://127.0.0.1:5000';
const МОДЕЛЬ = process.argv[3] || 'ds-flash';
const РЕЖИМ = process.argv[5] || 'сценарий';
const ЗАПРОС = process.argv[4] ||
  'Напиши фичу по получению данных о кадастровых участках с ресурса ' +
  'https://nspd.gov.ru/map и вывода их отдельным слоем на карте ГИС с газопроводами.';
const CDP = process.env.CDP || 'http://127.0.0.1:9222';
const ПРЕДЕЛ_МС = 300000;

const цель = await (await fetch(`${CDP}/json/new?${encodeURIComponent(АДРЕС)}`,
                                {method: 'PUT'})).json();
const ws = new WebSocket(цель.webSocketDebuggerUrl);
const ждущие = new Map();
const ошибки = [];
let счётчик = 0;

ws.addEventListener('message', (событие) => {
  const м = JSON.parse(событие.data);
  if (м.id && ждущие.has(м.id)) { ждущие.get(м.id)(м); ждущие.delete(м.id); }
  if (м.method === 'Runtime.exceptionThrown') {
    const д = м.params.exceptionDetails;
    ошибки.push('исключение: ' + (д.exception?.description || д.text));
  }
  if (м.method === 'Runtime.consoleAPICalled' && м.params.type === 'error') {
    ошибки.push('console.error: ' +
                м.params.args.map(а => а.value ?? а.description).join(' '));
  }
});
await new Promise(р => ws.addEventListener('open', р));

const зов = (метод, параметры = {}) => new Promise(р => {
  const id = ++счётчик;
  ждущие.set(id, р);
  ws.send(JSON.stringify({id, method: метод, params: параметры}));
});

const выполнить = async (код) => {
  const о = await зов('Runtime.evaluate',
                      {expression: код, awaitPromise: true, returnByValue: true});
  if (о.result?.exceptionDetails) ошибки.push('evaluate: ' + о.result.exceptionDetails.text);
  return о.result?.result?.value;
};

await зов('Runtime.enable');
await зов('Log.enable');
await new Promise(р => setTimeout(р, 3000));    // страница подтягивает состояние

console.log('страница:', await выполнить('document.title'));
console.log('модель:', await выполнить(
  `(() => { const с = document.getElementById('модель');
            с.value = ${JSON.stringify(МОДЕЛЬ)};
            с.dispatchEvent(new Event('change'));
            return с.value; })()`));

if (РЕЖИМ === 'пауза') {
  console.log('режим «по шагам»:', await выполнить(
    `(() => { const г = document.getElementById('режим-по-шагам');
              г.checked = true; г.dispatchEvent(new Event('change'));
              return г.checked; })()`));
}

if (РЕЖИМ === 'mcp') {
  const карточки = () => выполнить(`Array.from(document.querySelectorAll('.mcp-сервер')).map(к => ({
    имя: к.dataset['сервер'], класс: к.className.replace('mcp-сервер', '').trim() || '—',
    инструментов: к.querySelectorAll('.mcp-инструмент').length,
    закрыто: к.querySelectorAll('.mcp-инструмент.закрыт').length,
    только_чтение: к.querySelectorAll('.доступ.только-чтение').length,
    текст: к.innerText.split('\\n').slice(0, 4).join(' | ').slice(0, 160) }))`);
  const ждать = async (секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const идёт = await выполнить(`document.querySelector('#mcp-панель').innerText.includes('Подключаюсь')`);
      if (!идёт) return (Date.now() - начало) / 1000;
    }
    ошибки.push('панель MCP не дождалась конца подключения');
    return -1;
  };

  const до = await карточки();
  console.log('серверов на панели:', до.length, до.map(к => к.имя).join(', '));
  if (до.length < 2) ошибки.push('на панели MCP нет серверов');

  await выполнить(`document.querySelector('.mcp-подключить[data-mcp="agent-state"]').click(); 'ок'`);
  console.log('agent-state подключён за', await ждать(60), 'с');
  const свой = (await карточки()).find(к => к.имя === 'agent-state');
  console.log('agent-state:', JSON.stringify(свой));
  if (!свой || свой.класс !== 'ок') ошибки.push('свой сервер не подключился');
  if (свой && свой.инструментов !== 6) ошибки.push(`у своего сервера ${свой?.инструментов} инструментов вместо 6`);
  if (свой && свой.только_чтение !== 6) ошибки.push('не все инструменты своего сервера помечены «только чтение»');

  // Схема раскрывается и содержит JSON Schema.
  // Список инструментов свёрнут: раскрываем его так же, как человек, — щелчком.
  await выполнить(`document.querySelector('.mcp-список[data-список="agent-state"] summary').click(); 'ок'`);
  const раскрыт = await выполнить(`document.querySelector('.mcp-список[data-список="agent-state"]').open`);
  if (!раскрыт) ошибки.push('список инструментов не раскрывается');
  const схема = await выполнить(`(() => {
    const д = document.querySelector('.mcp-сервер[data-сервер="agent-state"] .mcp-инструмент[data-инструмент="get_task"] details');
    if (!д) return 'нет';
    д.querySelector('summary').click(); return д.querySelector('pre').innerText; })()`);
  console.log('схема get_task:', (схема || '').replace(/\s+/g, ' ').slice(0, 140));
  if (!(схема || '').includes('"task_id"')) ошибки.push('JSON-схема get_task не показана');

  await выполнить(`document.getElementById('mcp-все').click(); 'ок'`);
  console.log('все серверы осмотрены за', await ждать(240), 'с');
  const после = await карточки();
  const остался = await выполнить(`document.querySelector('.mcp-список[data-список="agent-state"]').open`);
  console.log('раскрытый список пережил повторное подключение:', остался ? 'да' : 'НЕТ');
  if (!остался) ошибки.push('после «Подключиться ко всем» раскрытый список свернулся');
  for (const к of после) console.log(`  ${к.имя.padEnd(12)} ${к.класс.padEnd(9)} инструментов ${к.инструментов}, закрыто ${к.закрыто} | ${к.текст}`);
  const фс = после.find(к => к.имя === 'filesystem');
  if (фс && фс.класс === 'ок' && фс.закрыто === 0) ошибки.push('фильтр filesystem не отмечен на странице');
  const гх = после.find(к => к.имя === 'github');
  if (гх && !['пропущен', 'ок'].includes(гх.класс)) ошибки.push('github показан не так, как настроен');

  const итог = await выполнить(`Array.from(document.querySelectorAll('#чат .системное'))
    .map(у => у.innerText).filter(т => т.startsWith('MCP:')).pop() || ''`);
  console.log('\nв чате:', итог);
  if (!итог) ошибки.push('в чате нет итога подключения');
  const сбоиMCP = await выполнить(`document.querySelectorAll('#чат .ошибка').length`);
  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:', сбоиMCP);
  console.log('исключений JS и провалов:', ошибки.length ? ошибки : 'нет');
  ws.close();
  process.exit(ошибки.length || сбоиMCP ? 1 : 0);
}

if (РЕЖИМ === 'инструменты') {
  const чат = () => выполнить(`Array.from(document.querySelectorAll('#чат .msg'))
    .map(у => (у.className.replace('msg','').trim() + ': ' + у.innerText).slice(0, 200))`);
  // Заявка появляется не в переписке, а карточкой справа: ждать её надо по
  // узлу, а не по тексту чата.
  const ждатьУзел = async (селектор, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const есть = await выполнить(`!!document.querySelector(${JSON.stringify(селектор)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`не дождались узла: ${селектор}`);
    return -1;
  };
  const ждатьЧат = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const есть = await выполнить(
        `document.getElementById('чат').innerText.includes(${JSON.stringify(признак)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`в чате не дождались: ${признак}`);
    return -1;
  };

  // 1. Включаем инструменты и ждём, пока страница скажет, во что это обходится.
  await выполнить(`(() => {
    document.getElementById('инструменты-серверы').value = 'tracker';
    const г = document.getElementById('инструменты-вкл');
    г.checked = true; г.dispatchEvent(new Event('change')); return 'ок'; })()`);
  const предел = Date.now() + 90000;
  let итогВключения = '';
  while (Date.now() < предел) {
    await new Promise(р => setTimeout(р, 1000));
    итогВключения = await выполнить(`document.getElementById('инструменты-итог').innerText`);
    if (итогВключения && !итогВключения.includes('подключаюсь')) break;
  }
  console.log('инструменты включены:', итогВключения);
  if (!/инстр\./.test(итогВключения)) ошибки.push('страница не показала, что досталось модели');
  if (!/схема ≈ \d+ ток/.test(итогВключения)) ошибки.push('страница не показала цену схемы');

  // 2. Вопрос, на который без трекера не ответить.
  await выполнить(`(() => {
    document.getElementById('ввод').value = 'Посмотри в трекере: какие задачи переноса сейчас в работе?';
    document.getElementById('отправить').click(); return 'ок'; })()`);
  console.log('ответ с вызовами получен за', await ждатьЧат('↳ tracker__', 180), 'с');
  const вызовы = await выполнить(`Array.from(document.querySelectorAll('#чат .вызов'))
    .map(у => у.innerText.slice(0, 120))`);
  console.log('вызовы на странице:');
  for (const в of вызовы || []) console.log('   ', в);
  if (!(вызовы || []).length) ошибки.push('строк вызовов на странице нет');
  const сбойные = await выполнить(`document.querySelectorAll('#чат .вызов.сбой').length`);
  if (сбойные) ошибки.push(`вызовов с ошибкой: ${сбойные}`);
  const ответСДанными = await выполнить(`(() => {
    const у = Array.from(document.querySelectorAll('#чат .от-агента')).pop();
    return у ? у.innerText : ''; })()`);
  console.log('в ответе есть ключи задач:', /MIG-\d/.test(ответСДанными || '') ? 'да' : 'НЕТ');
  if (!/MIG-\d/.test(ответСДанными || '')) ошибки.push('в ответе нет данных из трекера');

  // 3. Просьба изменить данные: вызова быть не должно, должна появиться заявка.
  await выполнить(`(() => {
    document.getElementById('ввод').value =
      'Добавь к задаче MIG-7 в трекере комментарий: «Проверено из браузера».';
    document.getElementById('отправить').click(); return 'ок'; })()`);
  console.log('заявка появилась за', await ждатьУзел('#заявки-панель .заявка', 180), 'с');
  const заявка = await выполнить(`(() => {
    const к = document.querySelector('#заявки-панель .заявка');
    return к ? к.innerText.replace(/\\s+/g, ' ').slice(0, 200) : ''; })()`);
  console.log('заявка:', заявка);
  if (!заявка.includes('add_comment')) ошибки.push('в заявке не тот инструмент');
  if (!заявка.includes('MIG-7')) ошибки.push('в заявке не видно аргументов');
  const самовольные = await выполнить(
    `document.getElementById('чат').innerText.includes('↳ tracker__add_comment')`);
  if (самовольные) ошибки.push('меняющий вызов выполнился без подтверждения');

  // 4. Подтверждаем — вызов происходит, результат виден, заявка исчезает.
  await выполнить(`document.querySelector('#заявки-панель .заявка-да').click(); 'ок'`);
  console.log('вызов исполнен за', await ждатьЧат('Заявка №', 120), 'с');
  const результат = await выполнить(`(() => {
    const у = document.querySelector('#чат .ответ-вызова');
    return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 160) : ''; })()`);
  console.log('ответ инструмента:', результат);
  if (!результат.includes('добавлен')) ошибки.push('результат подтверждённого вызова не показан');
  const осталось = await выполнить(`document.querySelectorAll('#заявки-панель .заявка').length`);
  console.log('заявок осталось:', осталось);
  if (осталось) ошибки.push('заявка не исчезла после исполнения');

  // 5. Кнопка «Вызвать» у инструмента в панели MCP. prompt() в headless-браузере
  // сам по себе не отвечает, поэтому подменяем его — человек в этом месте просто
  // вводит аргументы руками.
  await выполнить(`window.prompt = () => '{"key": "MIG-2"}'; 'ок'`);
  await выполнить(`document.querySelector('.mcp-подключить[data-mcp="tracker"]').click(); 'ок'`);
  await new Promise(р => setTimeout(р, 6000));
  const естьКнопка = await выполнить(`(() => {
    const к = document.querySelector('.mcp-вызвать[data-вызвать="tracker__get_issue"]');
    if (!к) return false; к.click(); return true; })()`);
  if (!естьКнопка) ошибки.push('в панели MCP нет кнопки «Вызвать»');
  else {
    console.log('ручной вызов выполнен за', await ждатьЧат('↳ tracker__get_issue', 60), 'с');
    const вручную = await выполнить(`(() => {
      const у = Array.from(document.querySelectorAll('#чат .ответ-вызова')).pop();
      return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 120) : ''; })()`);
    console.log('ответ на ручной вызов:', вручную);
    if (!вручную.includes('MIG-2')) ошибки.push('ручной вызов не вернул задачу');
  }

  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:',
              await выполнить(`document.querySelectorAll('#чат .ошибка').length`));
  for (const строка of (await чат()).slice(-6)) console.log('  ', строка);
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}


if (РЕЖИМ === 'планировщик') {
  const ждатьУзел = async (селектор, секунд, признак = '') => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const текст = await выполнить(`(() => {
        const у = document.querySelector(${JSON.stringify(селектор)});
        return у ? у.innerText : ''; })()`);
      if (текст && (!признак || текст.includes(признак))) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`не дождались ${селектор}${признак ? ' с «' + признак + '»' : ''}`);
    return -1;
  };
  const ждатьЧат = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const есть = await выполнить(
        `document.getElementById('чат').innerText.includes(${JSON.stringify(признак)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`в чате не дождались: ${признак}`);
    return -1;
  };

  // 1. Панель планировщика рисуется сама, без нажатий: её содержимое приходит
  //    отдельной ручкой и обновляется по таймеру.
  console.log('панель появилась за', await ждатьУзел('#планировщик-панель .карточка', 30), 'с');
  const пульс = await выполнить(`(() => {
    const у = document.querySelector('#планировщик-панель .пульс');
    return у ? у.innerText.replace(/\\s+/g, ' ').trim() : ''; })()`);
  console.log('пульс работника:', пульс);
  if (!пульс) ошибки.push('панель не показывает, жив ли работник');

  // 2. Ставим задание руками — тем же путём, что и модель: через MCP.
  await выполнить(`(() => {
    document.getElementById('задание-инструмент').value = 'tracker__list_issues';
    document.getElementById('задание-когда').value = 'каждые 30м';
    document.getElementById('задание-аргументы').value = '{"queue": "MIG"}';
    document.getElementById('задание-зачем').value = 'проверка из браузера';
    document.getElementById('задание-создать').click(); return 'ок'; })()`);
  console.log('задание поставлено за', await ждатьЧат('в расписании', 60), 'с');
  const задание = await выполнить(`(() => {
    const у = document.querySelector('#планировщик-панель .задание');
    return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 220) : ''; })()`);
  console.log('карточка задания:', задание);
  if (!задание.includes('tracker__list_issues')) ошибки.push('задания нет в панели');
  if (!задание.includes('каждые 30 минут')) ошибки.push('расписание показано не по-русски');
  if (!задание.includes('MIG')) ошибки.push('в карточке не видно аргументов');

  // 3. Непонятное расписание должно объясняться, а не молчать.
  await выполнить(`(() => {
    document.getElementById('задание-инструмент').value = 'tracker__list_queues';
    document.getElementById('задание-когда').value = 'когда-нибудь';
    document.getElementById('задание-аргументы').value = '';
    document.getElementById('задание-создать').click(); return 'ок'; })()`);
  console.log('отказ показан за', await ждатьЧат('Не понял расписание', 60), 'с');

  // 4. Сводка кнопкой: цифры считает тот же инструмент, что и работник.
  await выполнить(`document.querySelector('#сводка-сейчас').click(); 'ок'`);
  console.log('сводка собрана за', await ждатьЧат('Сводка №', 90), 'с');
  const сводка = await ждатьУзел('#планировщик-панель details', 30, 'Сводка');
  console.log('сводка в панели через', сводка, 'с');
  // Свёрнутый <details> в innerText не попадает вовсе — как и у человека,
  // который её не раскрыл. Раскрываем так же, как это делает он.
  await выполнить(`(() => {
    const д = document.querySelector('#планировщик-панель details');
    if (д) д.open = true; return 'ок'; })()`);
  const цифры = await выполнить(`(() => {
    const у = document.querySelector('#планировщик-панель .ответ-вызова');
    return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 160) : ''; })()`);
  console.log('цифры сводки:', цифры);
  if (!/запусков \d+/.test(цифры || '')) ошибки.push('в сводке нет счётчиков');

  // 5. Модель ставит напоминание сама — и это её собственная память, без заявки.
  await выполнить(`(() => {
    document.getElementById('инструменты-серверы').value = 'scheduler,tracker';
    const г = document.getElementById('инструменты-вкл');
    г.checked = true; г.dispatchEvent(new Event('change')); return 'ок'; })()`);
  const предел = Date.now() + 90000;
  let итогВключения = '';
  while (Date.now() < предел) {
    await new Promise(р => setTimeout(р, 1000));
    итогВключения = await выполнить(`document.getElementById('инструменты-итог').innerText`);
    if (итогВключения && !итогВключения.includes('подключаюсь')) break;
  }
  console.log('инструменты включены:', итогВключения);
  await выполнить(`(() => {
    document.getElementById('ввод').value =
      'Поставь напоминание через 2 часа: проверить бэкап PostGIS. Используй планировщик.';
    document.getElementById('отправить').click(); return 'ок'; })()`);
  console.log('модель позвала планировщик за', await ждатьЧат('↳ scheduler__', 180), 'с');
  const вызовы = await выполнить(`Array.from(document.querySelectorAll('#чат .вызов'))
    .map(у => у.innerText.slice(0, 120))`);
  for (const в of вызовы || []) console.log('   ', в);
  if (!(вызовы || []).some(в => в.includes('scheduler__'))) {
    ошибки.push('модель не позвала планировщик');
  }
  const заявки = await выполнить(`document.querySelectorAll('#заявки-панель .заявка').length`);
  if (заявки) ошибки.push('на своё напоминание агент зачем-то завёл заявку');

  // 6. Снятие задания: после него в расписании его быть не должно.
  const снято = await выполнить(`(() => {
    const к = document.querySelector('#планировщик-панель .задание-снять');
    if (!к) return false; к.click(); return true; })()`);
  if (!снято) ошибки.push('в панели нет кнопки «Снять»');
  else console.log('задание снято за', await ждатьЧат('снято с расписания', 60), 'с');

  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:',
              await выполнить(`document.querySelectorAll('#чат .ошибка').length`));
  const панель = await выполнить(`(() => {
    const у = document.getElementById('планировщик-панель');
    return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 400) : ''; })()`);
  console.log('панель на конец прогона:', панель);
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

// --- День 21: страница индекса документов --------------------------------------
// Два режима на одной странице /rag. «индекс» — проверка для чек-листа: всё,
// что страница умеет, нажимается по разу и сверяется с ожиданием, расхождение
// валит прогон. «показ» — то же для записи экрана: медленно, с подписями внизу,
// и в конце агент сам ищет в индексе через MCP-сервер docs.
const наСтраницеИндекса = async (путь = '/rag') => {
  // Пустой путь — переход уже сделан кликом по ссылке, остаётся дождаться страницы.
  if (путь) await выполнить(`(() => { location.href = ${JSON.stringify(путь)}; return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < 30000) {
    await new Promise(р => setTimeout(р, 700));
    const готово = await выполнить(
      `!!document.getElementById('собрать') && document.getElementById('состояние').innerText !== 'Загрузка…'`);
    if (готово) return true;
  }
  ошибки.push('страница индекса не загрузилась');
  return false;
};
// Ждать надо конкретный текст внутри конкретного узла: «готово» уже написано
// от прошлой работы, и общий признак проскакивал бы мимо.
const ждатьТекст = async (ид, признак, секунд) => {
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    const текст = await выполнить(
      `(document.getElementById(${JSON.stringify(ид)}) || {}).innerText || ''`);
    if ((текст || '').includes(признак)) return (Date.now() - начало) / 1000;
    await new Promise(р => setTimeout(р, 1000));
  }
  return -1;
};
const работаИндекса = async (кнопка, секунд) => {
  // Номер работы берём до нажатия: ждём, что он сменится и новая дойдёт до конца.
  const прежняя = await выполнить(`fetch('/api/rag').then(о => о.json()).then(с => с.работа ? с.работа.id : '')`);
  await выполнить(`(() => { document.getElementById(${JSON.stringify(кнопка)}).click(); return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    await new Promise(р => setTimeout(р, 1500));
    const р = await выполнить(`fetch('/api/rag').then(о => о.json()).then(с => с.работа)`);
    if (р && р.id !== прежняя && р.готово) return р;
  }
  return null;
};
const поискНаСтранице = async (запрос, где = 'обе', сколько = 3) => {
  await выполнить(`(() => {
    document.getElementById('запрос').value = ${JSON.stringify(запрос)};
    document.getElementById('где').value = ${JSON.stringify(где)};
    document.getElementById('сколько').value = ${сколько};
    return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 600));
  await выполнить(`(() => { document.getElementById('искать').click(); return 'ок'; })()`);
  await ждатьТекст('находки', '·', 60);
  await new Promise(р => setTimeout(р, 800));
  return выполнить(`(() => {
    const итог = {};
    document.querySelectorAll('#находки [data-стратегия]').forEach(к => {
      итог[к.dataset['стратегия']] = Array.from(к.querySelectorAll('.находка')).map(н => н.innerText.replace(/\\s+/g, ' '));
    });
    return итог; })()`);
};
// Чанк открывается по тексту находки, а не по месту в выдаче: место зависит
// от запроса, и однажды первым оказался чанк с контактными данными из
// приложения договора — в записи для отчёта ему не место.
const открытьЧанк = async (стратегия, место = 0, признак = '') => {
  const ид = await выполнить(`(() => {
    const к = document.querySelector('#находки [data-стратегия="${стратегия}"]');
    if (!к) return '';
    const находки = Array.from(к.querySelectorAll('.находка'));
    // Карточка показывает начало чанка (600 символов): у широкого чанка
    // признак может лежать дальше, и тогда берётся находка по месту.
    const нужная = (${JSON.stringify(признак)}
      && находки.find(н => н.innerText.includes(${JSON.stringify(признак)}))) || находки[${место}];
    const и = нужная && нужная.querySelector('.чанк-ид');
    if (!и) return '';
    и.click(); return и.dataset['ид']; })()`);
  if (ид) await ждатьТекст('чанк', ид, 20);
  return ид;
};

if (РЕЖИМ === 'индекс') {
  await наСтраницеИндекса();
  const состояние = await выполнить(`document.getElementById('состояние').innerText.replace(/\\s+/g, ' ')`);
  console.log('состояние:', состояние);
  if (!состояние.includes('эмбеддер')) ошибки.push('нет плашки эмбеддера');
  if (!/«структура» — чанков/.test(состояние)) ошибки.push('индекс «структура» не показан (соберите индекс до прогона)');
  const документы = await выполнить(`document.getElementById('документы').innerText`);
  const строк = await выполнить(`document.querySelectorAll('#документы tr').length - 1`);
  console.log('документов в таблице:', строк);
  if (строк < 1) ошибки.push('таблица документов пуста');
  if (!документы.includes('pdf')) ошибки.push('в таблице нет PDF');

  const находки = await поискНаСтранице('до какого числа платить за связь по договору VPN');
  console.log('поиск:', Object.fromEntries(Object.entries(находки).map(([к, в]) => [к, в.map(т => т.slice(0, 110))])));
  if (!находки['структура'] || !находки['фикс']) ошибки.push('поиск не показал обе стратегии рядом');
  if (!(находки['структура'] || []).some(т => т.includes('20 числа'))) ошибки.push('«структура» не нашла пункт о 20 числе');
  if (!(находки['структура'] || []).every(т => /стр\. \d/.test(т) && /(struct|fixed)-/.test(т))) {
    ошибки.push('у находки нет страницы или chunk_id');
  }

  const ид = await открытьЧанк('фикс');
  const чанк = await выполнить(`document.getElementById('чанк').innerText`);
  console.log('чанк:', ид, '→', (чанк || '').slice(0, 160).replace(/\n/g, ' '));
  if (!ид || !(чанк || '').includes(ид)) ошибки.push('чанк по chunk_id не открылся');
  if (!/предыдущего|следующего/.test(чанк || '')) ошибки.push('границы с соседями не показаны');

  const одна = await поискНаСтранице('антивирус', 'фикс', 2);
  if (Object.keys(одна).join() !== 'фикс') ошибки.push('выбор «только фикс» не сработал');

  const сравнение = await работаИндекса('сравнить', 300);
  console.log('сравнение:', сравнение && (сравнение.ошибка || JSON.stringify(сравнение.итог)));
  if (!сравнение || сравнение.ошибка) ошибки.push('сравнение стратегий не отработало');
  await ждатьТекст('сравнение', 'Как находится', 20);
  const таблица = await выполнить(`document.getElementById('сравнение').innerText.replace(/\\s+/g, ' ')`);
  for (const слово of ['Как нарезано', 'hit@1', 'mrr', 'По вопросам']) {
    if (!таблица.includes(слово)) ошибки.push(`в сравнении нет «${слово}»`);
  }
  // Ошибка ввода — сообщением на странице, а не молчанием.
  await выполнить(`(() => { document.getElementById('размер').value = 50;
    document.getElementById('перекрытие').value = 60;
    document.getElementById('собрать').click(); return 'ок'; })()`);
  if (await ждатьТекст('работа', 'Перекрытие должно быть меньше', 10) < 0) {
    ошибки.push('неверные параметры сборки не объяснены');
  }
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

// --- День 22: RAG-запрос с документами и без ------------------------------------
// Карточки «Вопрос по документам» и «Контрольные вопросы» на странице /rag.
// Помощники Дня 21 (наСтраницеИндекса, ждатьТекст) объявлены выше и годятся здесь.
const спроситьДокументы = async (вопрос, режим = 'оба', секунд = 120) => {
  await выполнить(`(() => {
    document.getElementById('rag-вопрос').value = ${JSON.stringify(вопрос)};
    document.getElementById('rag-режим').value = ${JSON.stringify(режим)};
    const м = document.getElementById('rag-модель');
    if ([...м.options].some(о => о.value === ${JSON.stringify(МОДЕЛЬ)})) м.value = ${JSON.stringify(МОДЕЛЬ)};
    document.getElementById('rag-ответы').innerHTML = '';
    return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 900));
  await выполнить(`(() => { document.getElementById('rag-спросить').click(); return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    await new Promise(р => setTimeout(р, 1000));
    const готово = await выполнить(`document.querySelectorAll('#rag-ответы .ответ').length > 0
      || !!document.querySelector('#rag-ответы .ошибка')`);
    if (готово) break;
  }
  return выполнить(`(() => {
    const итог = {};
    document.querySelectorAll('#rag-ответы .ответ').forEach(о => {
      итог[о.dataset['режим']] = о.innerText.replace(/\\s+/g, ' '); });
    const о = document.querySelector('#rag-ответы .ошибка');
    if (о) итог['ошибка'] = о.innerText;
    return итог; })()`);
};
const прогнатьКонтроль = async (секунд = 600) => {
  const прежний = await выполнить(`fetch('/api/rag').then(о => о.json()).then(с => с.контроль ? с.контроль.when : '')`);
  await выполнить(`(() => { document.getElementById('прогнать-контроль').click(); return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    await new Promise(р => setTimeout(р, 2000));
    const с = await выполнить(`fetch('/api/rag').then(о => о.json()).then(с => ({
      работа: с.работа, когда: с.контроль ? с.контроль.when : '', итог: с.контроль ? с.контроль.totals : null}))`);
    if (с.работа && с.работа.готово && с.работа.вид === 'контроль' && (с.когда !== прежний || с.работа.ошибка)) return с;
  }
  return null;
};

if (РЕЖИМ === 'rag') {
  await наСтраницеИндекса();
  const заголовок = await выполнить('document.title');
  // С Дня 23 страница называется «RAG: реранкинг…»; карточки Дня 22 те же.
  if (!заголовок.startsWith('RAG')) ошибки.push('заголовок /rag не RAG: ' + заголовок);
  const оба = await спроситьДокументы('Какая неустойка за просрочку оплаты предусмотрена договором с ВымпелКомом?');
  console.log('без RAG:', (оба['без'] || '').slice(0, 220));
  console.log('с RAG:', (оба['rag'] || '').slice(0, 320));
  if (!оба['без'] || !оба['rag']) ошибки.push('режим «оба» не показал два ответа рядом');
  if (!/0,2\s*%|0\.2\s*%/.test(оба['rag'] || '')) ошибки.push('с RAG нет «0,2 %»');
  if (!/Источники ответа: \[1\]|\[\d\] Вымпелком/.test(оба['rag'] || '')) ошибки.push('с RAG не показан источник');
  if (!/Фрагменты в промпте: \d/.test(оба['rag'] || '')) ошибки.push('не показаны фрагменты промпта');
  if (!/без документов/.test(оба['без'] || '')) ошибки.push('без RAG нет пометки об отсутствии источников');
  const выделено = await выполнить(`document.querySelectorAll('#rag-ответы .ссылка-n').length`);
  if (!выделено) ошибки.push('ссылки [n] в тексте не подсвечены');
  const ловушка = await спроситьДокументы('Сколько стоит подписка на Microsoft Office 365 в бюджете ИТ на 2026 год?', 'rag');
  console.log('ловушка с RAG:', (ловушка['rag'] || '').slice(0, 200));
  if (!/В документах этого нет/.test(ловушка['rag'] || '')) ошибки.push('ловушка с RAG не ответила «в документах этого нет»');
  if (ловушка['без']) ошибки.push('режим «только с RAG» показал и ответ без RAG');
  const контроль = await прогнатьКонтроль();
  // Кнопка с Дня 23 прогоняет пять режимов; «база» — это поиск Дня 22.
  console.log('контроль:', контроль && JSON.stringify(контроль.итог && {без: контроль.итог['без'].верно_по_правилам, база: контроль.итог['база'].верно_по_правилам}));
  if (!контроль || контроль.работа.ошибка) ошибки.push('контрольные вопросы не прогнались');
  await ждатьТекст('контроль', 'По вопросам', 20);
  const таблица = await выполнить(`document.getElementById('контроль').innerText.replace(/\\s+/g, ' ')`);
  for (const слово of ['верно по правилам', 'ловушек пройдено', 'судья: верно', 'c09', 'ловушка']) {
    if (!таблица.includes(слово)) ошибки.push(`в таблице контроля нет «${слово}»`);
  }
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

// --- День 23: второй этап поиска — реранкинг, фильтр и rewrite -----------------
// Карточки «Поиск в два этапа» и «Подбор порога и top-K» на /rag. Помощники
// Дней 21–22 (наСтраницеИндекса, ждатьТекст, прогнатьКонтроль) объявлены выше.
const показатьОтборНаСтранице = async (вопрос, режим = 'полный', поправки = {}, секунд = 300) => {
  await выполнить(`(() => {
    const п = ${JSON.stringify(поправки)};
    document.getElementById('отбор-вопрос').value = ${JSON.stringify(вопрос)};
    document.getElementById('отбор-режим').value = ${JSON.stringify(режим)};
    document.getElementById('отбор-реранкер').value = п.реранкер ?? '*';
    document.getElementById('отбор-kдо').value = п.kдо ?? '';
    document.getElementById('отбор-kпосле').value = п.kпосле ?? '';
    document.getElementById('отбор-порог').value = п.порог ?? '';
    document.getElementById('отбор-rewrite').value = п.rewrite ?? '*';
    document.getElementById('отбор').innerHTML = '';
    return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 900));
  await выполнить(`(() => { document.getElementById('показать-отбор').click(); return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    await new Promise(р => setTimeout(р, 1000));
    const готово = await выполнить(`!!document.querySelector('#отбор table') || !!document.querySelector('#отбор .ошибка')`);
    if (готово) break;
  }
  // Строка кандидата — судьба, место «было» и текст: по «было» видно, откуда
  // реранкер поднял фрагмент.
  return выполнить(`(() => ({
    текст: document.getElementById('отбор').innerText.replace(/\\s+/g, ' '),
    строки: Array.from(document.querySelectorAll('#отбор tr[data-судьба]')).map(т => ({
      судьба: т.dataset['судьба'], было: +т.children[1].innerText,
      текст: т.innerText.replace(/\\s+/g, ' ')})),
    ошибка: (document.querySelector('#отбор .ошибка') || {}).innerText || ''
  }))()`);
};
const вПромпт = (о) => (о.строки || []).filter(с => с.судьба === 'в промпт');
const ответитьВсеНаСтранице = async (вопрос, секунд = 400) => {
  await выполнить(`(() => {
    document.getElementById('отбор-вопрос').value = ${JSON.stringify(вопрос)};
    const м = document.getElementById('rag-модель');
    if ([...м.options].some(о => о.value === ${JSON.stringify(МОДЕЛЬ)})) м.value = ${JSON.stringify(МОДЕЛЬ)};
    document.getElementById('ответы-поиск').innerHTML = '';
    return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 700));
  await выполнить(`(() => { document.getElementById('ответить-все').click(); return 'ок'; })()`);
  const начало = Date.now();
  while (Date.now() - начало < секунд * 1000) {
    await new Promise(р => setTimeout(р, 1000));
    const готово = await выполнить(`document.querySelectorAll('#ответы-поиск [data-поиск]').length > 0
      || !!document.querySelector('#ответы-поиск .ошибка')`);
    if (готово) break;
  }
  return выполнить(`(() => {
    const итог = {};
    document.querySelectorAll('#ответы-поиск [data-поиск]').forEach(о => {
      итог[о.dataset['поиск']] = о.innerText.replace(/\\s+/g, ' '); });
    const о = document.querySelector('#ответы-поиск .ошибка');
    if (о) итог['ошибка'] = о.innerText;
    return итог; })()`);
};
// Вопросы показа и проверки. c03 — промах Дня 22: пункт 2.2.7 о тарифах стоит
// в длинном разделе, и косинус ставил его десятым. Ловушка — из стенда подбора.
const ВОПРОС_ТАРИФЫ = 'За сколько дней Ростелеком обязан предупредить об изменении тарифов по договору VPN?';
const ВОПРОС_ЛОВУШКА = 'Сколько точек доступа Wi-Fi закупается по бюджету ИТ на 2026 год?';
// Пункт 2.2.7 стоит в конце чанка «2. Права и обязанности сторон» договора VPN;
// в строке отбора виден раздел, а не номер пункта внутри чанка.
const этоТарифы = (с) => /Ростелеком VPN/.test(с.текст) && /Права и обязанности/.test(с.текст);

if (РЕЖИМ === 'реранк') {
  await наСтраницеИндекса();
  const заголовок = await выполнить('document.title');
  if (!заголовок.startsWith('RAG: реранкинг')) ошибки.push('заголовок /rag не Дня 23: ' + заголовок);
  const режимы = await выполнить(`Array.from(document.querySelectorAll('#режимы-поиска tr[data-режим-поиска]')).map(т => т.getAttribute('data-режим-поиска'))`);
  console.log('режимы поиска:', режимы.join(', '));
  if (режимы.join() !== 'база,фильтр,rewrite,полный') ошибки.push('таблица режимов не та: ' + режимы.join());
  const источник = await выполнить(`document.getElementById('настройки-источник').innerText`);
  console.log('параметры:', источник);
  if (!источник.includes('rag-settings.json')) ошибки.push('режимы не из rag-settings.json (подбора не было?)');
  if (await ждатьТекст('подбор', 'Выбрано: реранкер', 20) < 0) ошибки.push('карточка подбора не показала выбранное');

  // База Дня 22: нужный пункт не в промпте. Фильтр: реранкер поднимает его.
  const база = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'база');
  console.log('база, в промпт:', вПромпт(база).map(с => с.текст.slice(0, 90)));
  if (база.ошибка) ошибки.push('отбор «база»: ' + база.ошибка);
  if (/реранкер «/.test(база.текст)) ошибки.push('в режиме «база» есть этап реранкера');
  const фильтр = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'фильтр');
  console.log('фильтр:', фильтр.текст.slice(0, 400));
  console.log('фильтр, в промпт:', вПромпт(фильтр).map(с => `было ${с.было}: ` + с.текст.slice(0, 90)));
  if (фильтр.ошибка) ошибки.push('отбор «фильтр»: ' + фильтр.ошибка);
  for (const слово of ['2) первый этап', '3) порог косинуса', '4) реранкер «', '5) фильтр']) {
    if (!фильтр.текст.includes(слово)) ошибки.push(`в отборе «фильтр» нет этапа «${слово}»`);
  }
  if (!вПромпт(фильтр).some(с => с.было > 5)) ошибки.push('реранкер не поднял ни одного кандидата из-за пределов top-5');
  if (!вПромпт(фильтр).some(этоТарифы)) ошибки.push('раздел 2 договора VPN (п. 2.2.7) не попал в промпт после фильтра');

  const полный = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'полный');
  console.log('полный:', полный.текст.slice(0, 400));
  if (!/1\) запросы/.test(полный.текст) || !полный.текст.includes('исходный:')) ошибки.push('в режиме «полный» не показаны переписанные запросы');

  // Порог и K вживую. Подобранный порог отсёк часть кандидатов (строки «ниже
  // порога» у фильтра выше), порог 0 пропускает всех до K после. «Порог 0,999
  // режет всё» для LLM-реранкера не годится: прямому ответу он ставит 10 из 10.
  if (!фильтр.строки.some(с => с.судьба === 'ниже порога')) ошибки.push('подобранный порог никого не отсёк');
  const порог = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'фильтр', {порог: '0'});
  console.log('порог 0: в промпт', вПромпт(порог).length, '· ниже порога', порог.строки.filter(с => с.судьба === 'ниже порога').length,
              '; подобранный: в промпт', вПромпт(фильтр).length);
  if (порог.строки.some(с => с.судьба === 'ниже порога')) ошибки.push('порог 0 кого-то отсёк');
  if (вПромпт(порог).length <= вПромпт(фильтр).length) ошибки.push('без порога в промпт ушло не больше, чем с порогом');
  const k2 = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'фильтр', {kпосле: '2', порог: '0'});
  console.log('K после 2: в промпт', вПромпт(k2).length);
  if (вПромпт(k2).length !== 2) ошибки.push('K после = 2 не оставил ровно два фрагмента');
  if (!k2.строки.some(с => с.судьба === 'за пределами K')) ошибки.push('нет строк «за пределами K»');
  const ловушка = await показатьОтборНаСтранице(ВОПРОС_ЛОВУШКА, 'фильтр');
  console.log('ловушка, фильтр: в промпт', вПромпт(ловушка).length, 'из', ловушка.строки.length);

  const ответы = await ответитьВсеНаСтранице(ВОПРОС_ТАРИФЫ);
  for (const [м, т] of Object.entries(ответы)) console.log(`ответ «${м}»:`, т.slice(0, 200));
  if (Object.keys(ответы).join() !== 'база,фильтр,rewrite,полный') ошибки.push('не четыре ответа: ' + Object.keys(ответы).join());
  if (!/\b10\b/.test(ответы['фильтр'] || '')) ошибки.push('с фильтром ответ не назвал 10 дней');

  const контроль = await прогнатьКонтроль(2400);
  const т = контроль && контроль.итог;
  if (т) console.log('верно по правилам:', Object.entries(т).map(([м, x]) => `${м} ${x.верно_по_правилам}`).join(', '));
  if (!контроль || контроль.работа.ошибка) ошибки.push('контрольные вопросы не прогнались');
  await ждатьТекст('контроль', 'По вопросам', 20);
  const таблица = await выполнить(`document.getElementById('контроль').innerText.replace(/\\s+/g, ' ')`);
  for (const слово of ['без RAG', '+ фильтр', '+ rewrite', '+ rewrite и фильтр', 'фрагментов в промпте', 'время поиска']) {
    if (!таблица.includes(слово)) ошибки.push(`в таблице контроля нет «${слово}»`);
  }
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

if (РЕЖИМ === 'показ') {
  const ПАУЗА = Number(process.env.PAUSE || 3000);
  const замечания = [];
  const подождать = мс => new Promise(р => setTimeout(р, мс));
  const должно = (условие, чего) => { if (!условие) замечания.push(чего); };
  const подпись = async (сцена, текст) => {
    await выполнить(`(() => {
      let п = document.getElementById('подпись-показа');
      if (!п) {
        п = document.createElement('div');
        п.id = 'подпись-показа';
        п.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
          'background:#0f172a;color:#f8fafc;padding:10px 18px;font:16px/1.45 ' +
          'system-ui,sans-serif;box-shadow:0 -2px 10px rgba(0,0,0,.25)';
        document.body.appendChild(п);
        document.body.style.paddingBottom = '70px';
      }
      п.innerHTML = '<b>' + ${JSON.stringify(сцена)} + '</b> — ' + ${JSON.stringify(текст)};
      return 'ок'; })()`);
    console.log(`\n=== ${сцена} — ${текст}`);
  };
  const к = async (селектор, блок = 'start') => выполнить(`(() => {
    const у = document.querySelector(${JSON.stringify(селектор)});
    if (у) у.scrollIntoView({behavior: 'smooth', block: ${JSON.stringify(блок)}});
    return !!у; })()`);

  // --- сцена 1 -----------------------------------------------------------------
  await подождать(1500);
  await подпись('Сцена 1', 'День 23: после поиска bge-m3 — второй этап. Реранкер переоценивает кандидатов, порог отсекает лишнее, в промпт идёт top-K');
  await к('#rag-вход', 'center');
  await подождать(ПАУЗА * 1.5);
  await выполнить(`(() => { document.getElementById('rag-ссылка-23').click(); return 'ок'; })()`);
  await наСтраницеИндекса('');
  await к('#отбор-карточка');
  await подпись('Сцена 1', 'Четыре режима поиска: база Дня 22, + фильтр (реранкер и порог), + rewrite (2–3 запроса и слияние RRF), + rewrite и фильтр');
  await подождать(ПАУЗА * 2);

  // --- сцена 2 -----------------------------------------------------------------
  await к('#подбор-карточка');
  const подбор = await выполнить(`fetch('/api/rag').then(о => о.json()).then(с => с.подбор)`);
  должно(!!подбор, 'подбора нет — режимы на значениях по умолчанию');
  if (подбор) {
    const в = подбор.выбрано;
    await подпись('Сцена 2', `Порог и top-K подобраны на 24 вопросах и 5 ловушках Дня 21 (контрольные не участвуют): реранкер «${в.реранкер}», K до ${в.k_до}, K после ${в.k_после}, порог ${в.порог}`);
    await подождать(ПАУЗА * 2.5);
    await подпись('Сцена 2', `Полнота ${в.среднее.полнота.toFixed(2)} и MRR ${в.среднее.mrr.toFixed(2)} против ${подбор.база.полнота.toFixed(2)} / ${подбор.база.mrr.toFixed(2)} у базы. Справа — цена: секунды реранкера на вопрос без кэша`);
    await подождать(ПАУЗА * 3);
  }

  // --- сцена 3 -----------------------------------------------------------------
  await к('#отбор-карточка');
  await подпись('Сцена 3', 'Промах Дня 22: «за сколько дней Ростелеком предупреждает о тарифах». База — top-5 по косинусу, нужного пункта 2.2.7 в промпте нет');
  const база = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'база');
  должно(!вПромпт(база).some(этоТарифы), 'в базе раздел с п. 2.2.7 уже в промпте — сцена 3 теряет смысл');
  await к('#отбор');
  await подождать(ПАУЗА * 3);
  await к('#отбор-карточка');
  await подпись('Сцена 3', 'Режим «+ фильтр»: берём больше кандидатов, реранкер переоценивает пару «вопрос + фрагмент» целиком');
  const фильтр = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'фильтр');
  const поднят = вПромпт(фильтр).find(этоТарифы);
  должно(!!поднят, 'после фильтра раздел с п. 2.2.7 не в промпте');
  await к('#отбор');
  await подпись('Сцена 3', поднят ? `Раздел 2 договора VPN (в его конце — п. 2.2.7) был на ${поднят.было}-м месте по косинусу — после реранкера он в промпте; всё ниже порога помечено «ниже порога»` : 'Судьба каждого кандидата: в промпт, ниже порога, за пределами K');
  await подождать(ПАУЗА * 3.5);

  // --- сцена 4 -----------------------------------------------------------------
  await к('#отбор-карточка');
  await подпись('Сцена 4', 'Режим «+ rewrite и фильтр»: модель переписывает вопрос в запросы на языке документов, поиск идёт по каждому, списки сливаются RRF');
  const полный = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'полный');
  должно(полный.текст.includes('исходный:'), 'не показаны переписанные запросы');
  await к('#отбор');
  await подождать(ПАУЗА * 3.5);

  // --- сцена 5 -----------------------------------------------------------------
  await к('#отбор-карточка');
  await подпись('Сцена 5', 'Порог и K настраиваются вживую: K после = 2 — в промпт идут двое, остальные «за пределами K»');
  const k2 = await показатьОтборНаСтранице(ВОПРОС_ТАРИФЫ, 'фильтр', {kпосле: '2'});
  должно(вПромпт(k2).length <= 2, 'K после = 2 оставил больше двух');
  await к('#отбор');
  await подождать(ПАУЗА * 2.5);
  await к('#отбор-карточка');
  await подпись('Сцена 5', 'Ловушка: точек доступа Wi-Fi в бюджете нет. Порог отсечения не пускает в промпт похожие, но нерелевантные таблицы');
  const ловушка = await показатьОтборНаСтранице(ВОПРОС_ЛОВУШКА, 'фильтр');
  await к('#отбор');
  await подпись('Сцена 5', `Ловушка: из ${ловушка.строки.length} кандидатов в промпт прошло ${вПромпт(ловушка).length}`);
  await подождать(ПАУЗА * 3);

  // --- сцена 6 -----------------------------------------------------------------
  await выполнить(`(() => { document.getElementById('отбор').innerHTML = ''; return 'ок'; })()`);
  await к('#отбор-карточка');
  await подпись('Сцена 6', 'Тот же вопрос модели ds-flash во всех четырёх режимах поиска рядом');
  const ответы = await ответитьВсеНаСтранице(ВОПРОС_ТАРИФЫ);
  должно(Object.keys(ответы).length === 4, 'не четыре ответа');
  должно(/\b10\b/.test(ответы['фильтр'] || ''), 'с фильтром ответ не назвал 10 дней');
  for (const [м, т] of Object.entries(ответы)) console.log(`ответ «${м}»:`, т.slice(0, 160));
  await к('#ответы-поиск');
  await подпись('Сцена 6', 'База: пункта нет в контексте — «в документах этого нет». С фильтром — «не менее чем за 10 дней» со ссылкой на п. 2.2.7');
  await подождать(ПАУЗА * 4);

  // --- сцена 7 -----------------------------------------------------------------
  await к('#контроль-карточка');
  await подпись('Сцена 7', 'Десять контрольных вопросов (8 с ответом + 2 ловушки) в пяти режимах: без RAG, база, + фильтр, + rewrite, + rewrite и фильтр; правила и судья');
  const контроль = await прогнатьКонтроль(2400);
  должно(контроль && !контроль.работа.ошибка, 'контрольные вопросы не прогнались');
  await ждатьТекст('контроль', 'По вопросам', 20);
  await к('#контроль-карточка');
  const т = контроль && контроль.итог;
  if (т) {
    console.log('верно по правилам:', Object.entries(т).map(([м, x]) => `${м} ${x.верно_по_правилам}`).join(', '));
    const в = м => (т[м] ? т[м].верно_по_правилам : '—');
    await подпись('Сцена 7', `Верно по правилам из 10: без RAG ${в('без')}, база ${в('база')}, + фильтр ${в('фильтр')}, + rewrite ${в('rewrite')}, + rewrite и фильтр ${в('полный')}`);
  }
  await подождать(ПАУЗА * 3);
  await выполнить(`(() => {
    const с = document.querySelector('#контроль tr[data-вопрос="c03"]');
    if (с) с.scrollIntoView({behavior: 'smooth', block: 'center'});
    return 'ок'; })()`);
  await подпись('Сцена 7', 'c03 — промах Дня 22: база отвечает «в документах этого нет», режимы с реранкером находят п. 2.2.7');
  await подождать(ПАУЗА * 3);
  await к('#контроль-карточка');
  // Время поиска здесь почти равно: оценки реранкера лежат в кэше индекса.
  // Честная цена второго этапа — замер подбора без кэша (сцена 2).
  if (т) await подпись('Сцена 7', `Цена: фрагментов в промпте ${т['база'].фрагментов} у базы против ${т['фильтр'].фрагментов} с фильтром, входных токенов ${т['база'].токенов_вход} против ${т['фильтр'].токенов_вход}. Оценки реранкера здесь из кэша; без кэша LLM-реранкер добавляет ≈ ${подбор ? подбор.задержка.llm : 3} с на вопрос`);
  await подождать(ПАУЗА * 3);

  console.log('\n=== итог показа ===');
  console.log(замечания.length ? 'РАСХОЖДЕНИЯ:\n  ' + замечания.join('\n  ')
                               : 'показ прошёл без расхождений');
  if (ошибки.length) console.log('ошибки страницы:\n  ' + ошибки.join('\n  '));
  ws.close();
  process.exit(0);          // показ не проверка: код возврата всегда 0
}

if (РЕЖИМ === 'показ-22') {
  const ПАУЗА = Number(process.env.PAUSE || 3000);
  const замечания = [];
  const подождать = мс => new Promise(р => setTimeout(р, мс));
  const должно = (условие, чего) => { if (!условие) замечания.push(чего); };
  const подпись = async (сцена, текст) => {
    await выполнить(`(() => {
      let п = document.getElementById('подпись-показа');
      if (!п) {
        п = document.createElement('div');
        п.id = 'подпись-показа';
        п.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
          'background:#0f172a;color:#f8fafc;padding:10px 18px;font:16px/1.45 ' +
          'system-ui,sans-serif;box-shadow:0 -2px 10px rgba(0,0,0,.25)';
        document.body.appendChild(п);
        document.body.style.paddingBottom = '70px';
      }
      п.innerHTML = '<b>' + ${JSON.stringify(сцена)} + '</b> — ' + ${JSON.stringify(текст)};
      return 'ок'; })()`);
    console.log(`\n=== ${сцена} — ${текст}`);
  };
  const к = async (селектор, блок = 'start') => выполнить(`(() => {
    const у = document.querySelector(${JSON.stringify(селектор)});
    if (у) у.scrollIntoView({behavior: 'smooth', block: ${JSON.stringify(блок)}});
    return !!у; })()`);

  // --- сцена 1 -----------------------------------------------------------------
  await подождать(1500);
  await подпись('Сцена 1', 'День 22: у агента два режима ответа по документам — с RAG и без RAG');
  await к('#rag-вход', 'center');
  await подождать(ПАУЗА * 1.5);
  await выполнить(`(() => { document.getElementById('rag-ссылка-22').click(); return 'ок'; })()`);
  await наСтраницеИндекса('');
  await подпись('Сцена 1', 'Вопрос → поиск чанков в индексе Дня 21 → фрагменты с паспортом в промпт → ответ модели со ссылками [n]');
  await подождать(ПАУЗА * 1.5);

  // --- сцена 2 -----------------------------------------------------------------
  await к('#rag-вопрос-карточка');
  await подпись('Сцена 2', 'Один вопрос, одна модель (ds-flash), два режима рядом: слева без документов, справа с RAG');
  const оба = await спроситьДокументы('Какая неустойка за просрочку оплаты предусмотрена договором с ВымпелКомом?');
  должно(/0,2\s*%|0\.2\s*%/.test(оба['rag'] || ''), 'с RAG нет «0,2 %»');
  должно(!!оба['без'], 'нет ответа без RAG');
  console.log('без RAG:', (оба['без'] || '').slice(0, 200));
  await подождать(ПАУЗА);
  await подпись('Сцена 2', 'Без RAG модель отвечает «по типовому договору» — правдоподобно, но не из нашего договора. С RAG — 0,2 % в день и ссылка [1]');
  await подождать(ПАУЗА * 3);

  // --- сцена 3 -----------------------------------------------------------------
  await выполнить(`(() => {
    const д = document.querySelector('#rag-ответы .с-rag details'); if (д) д.open = true;
    const о = document.querySelector('#rag-ответы .с-rag'); if (о) о.scrollIntoView({behavior: 'smooth', block: 'start'});
    return 'ок'; })()`);
  await подпись('Сцена 3', 'Что ушло в промпт: пять фрагментов со сходством; процитированный [1] — раздел 5 скана договора с ВымпелКомом (OCR)');
  await подождать(ПАУЗА * 3);

  // --- сцена 4 -----------------------------------------------------------------
  await к('#rag-вопрос-карточка');
  await подпись('Сцена 4', 'Ловушка: подписки на Office 365 в документах нет. Правильный ответ — «В документах этого нет»');
  const ловушка = await спроситьДокументы('Сколько стоит подписка на Microsoft Office 365 в бюджете ИТ на 2026 год?');
  должно(/В документах этого нет/.test(ловушка['rag'] || ''), 'ловушка с RAG не ответила «в документах этого нет»');
  console.log('ловушка без RAG:', (ловушка['без'] || '').slice(0, 200));
  await подождать(ПАУЗА);
  await подпись('Сцена 4', 'С RAG модель не угадывает, хотя в промпт попали похожие бюджетные таблицы: правило промпта держит');
  await подождать(ПАУЗА * 3);

  // --- сцена 5 -----------------------------------------------------------------
  await к('#контроль-карточка');
  await подпись('Сцена 5', 'Десять контрольных вопросов (8 с ответом + 2 ловушки) в обоих режимах: правила по фактам и источникам + модель-судья');
  const контроль = await прогнатьКонтроль();
  должно(контроль && !контроль.работа.ошибка, 'контрольные вопросы не прогнались');
  await ждатьТекст('контроль', 'По вопросам', 20);
  await к('#контроль-карточка');
  const т = контроль && контроль.итог;
  if (т) console.log('верно по правилам: без', т['без'].верно_по_правилам, 'с RAG', т['rag'].верно_по_правилам,
                     '· выдумок: без', т['без'].выдумок, 'с RAG', т['rag'].выдумок);
  await подпись('Сцена 5', т ? `Итог: верно по правилам без RAG ${т['без'].верно_по_правилам} из 10, с RAG ${т['rag'].верно_по_правилам} из 10; выдуманной конкретики ${т['без'].выдумок} против ${т['rag'].выдумок}` : 'Итог прогона');
  await подождать(ПАУЗА * 3);

  // --- сцена 6 -----------------------------------------------------------------
  await выполнить(`(() => {
    const с = document.querySelector('#контроль tr[data-вопрос="c02"]');
    if (с) { const д = с.nextElementSibling.querySelector('details'); if (д) д.open = true;
             с.scrollIntoView({behavior: 'smooth', block: 'start'}); }
    return 'ок'; })()`);
  await подпись('Сцена 6', 'По вопросам: c02 — общая стоимость договора VPN. Без RAG модели взять число неоткуда, с RAG — 24 441 696 руб. и ссылка на п. 3.10');
  await подождать(ПАУЗА * 3);
  await выполнить(`(() => {
    const с = document.querySelector('#контроль tr[data-вопрос="c03"]');
    if (с) с.scrollIntoView({behavior: 'smooth', block: 'start'});
    return 'ок'; })()`);
  await подпись('Сцена 6', 'Честный промах: c03 поиск не нашёл пункт в длинном разделе — с RAG «в документах этого нет», а не выдумка. Это работа для реранкинга');
  await подождать(ПАУЗА * 3);

  console.log('\n=== итог показа ===');
  console.log(замечания.length ? 'РАСХОЖДЕНИЯ:\n  ' + замечания.join('\n  ')
                               : 'показ прошёл без расхождений');
  if (ошибки.length) console.log('ошибки страницы:\n  ' + ошибки.join('\n  '));
  ws.close();
  process.exit(0);          // показ не проверка: код возврата всегда 0
}

if (РЕЖИМ === 'показ-21') {
  // PAUSE латиницей: имя приходит из bash, а он кириллицу в именах переменных
  // окружения не принимает.
  const ПАУЗА = Number(process.env.PAUSE || 3000);
  const замечания = [];
  const подождать = мс => new Promise(р => setTimeout(р, мс));
  const должно = (условие, чего) => { if (!условие) замечания.push(чего); };
  // Подпись внизу страницы: в записи нет звука, и без неё непонятно, на что
  // смотреть. После перехода на другую страницу её надо ставить заново.
  const подпись = async (сцена, текст) => {
    await выполнить(`(() => {
      let п = document.getElementById('подпись-показа');
      if (!п) {
        п = document.createElement('div');
        п.id = 'подпись-показа';
        п.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
          'background:#0f172a;color:#f8fafc;padding:10px 18px;font:16px/1.45 ' +
          'system-ui,sans-serif;box-shadow:0 -2px 10px rgba(0,0,0,.25)';
        document.body.appendChild(п);
        document.body.style.paddingBottom = '70px';
      }
      п.innerHTML = '<b>' + ${JSON.stringify(сцена)} + '</b> — ' + ${JSON.stringify(текст)};
      return 'ок'; })()`);
    console.log(`\n=== ${сцена} — ${текст}`);
  };
  const к = async (селектор, блок = 'start') => выполнить(`(() => {
    const у = document.querySelector(${JSON.stringify(селектор)});
    if (у) у.scrollIntoView({behavior: 'smooth', block: ${JSON.stringify(блок)}});
    return !!у; })()`);

  // --- сцена 1: откуда начинается день --------------------------------------
  await подождать(1500);
  await подпись('Сцена 1', 'Агент миграции ГИС: новое в Дне 21 — индекс документов организации');
  await к('#rag-вход', 'center');
  await подождать(ПАУЗА * 1.5);
  await выполнить(`(() => { document.getElementById('rag-ссылка').click(); return 'ок'; })()`);
  await наСтраницеИндекса('');
  await подпись('Сцена 1', 'Документы из DocsSample/: договоры связи (PDF и сканы) и бюджетные таблицы Excel');
  await подождать(ПАУЗА);
  await к('#документы');
  await подождать(ПАУЗА * 2);
  const документы = await выполнить(`document.getElementById('документы-итог').innerText`);
  console.log('документы:', документы);
  должно(/OCR [1-9]/.test(документы || ''), 'в итоге документов не видно страниц, прошедших OCR');
  await подпись('Сцена 1', 'Сканы без текстового слоя распознаны tesseract — колонка «OCR стр.»');
  await подождать(ПАУЗА * 1.5);

  // --- сцена 2: сборка -------------------------------------------------------
  // Первая сборка на CPU — это час работы bge-m3 (сотни тысяч токенов), и в
  // кадр она не помещается. Поэтому показывается пересборка тех же чанков:
  // конвейер проходит целиком (извлечение, резка, запись), а векторы берутся
  // из кэша по хэшу текста — это и видно в итоге «из кэша N».
  await к('#собрать', 'center');
  await подпись('Сцена 2', 'Сборка обеих стратегий: извлечение → чанки → векторы bge-m3 → SQLite. Параметры — размер, перекрытие, предел раздела');
  await выполнить(`(() => {
    document.querySelectorAll('.стратегия').forEach(г => { г.checked = true; });
    document.getElementById('размер').value = 500;
    document.getElementById('перекрытие').value = 75;
    document.getElementById('макс').value = 700;
    return 'ок'; })()`);
  await подождать(ПАУЗА * 1.5);
  const сборка = await работаИндекса('собрать', 1200);
  const изКэша = сборка && сборка.итог ? Object.entries(сборка.итог.стратегии).map(([с, д]) => `${с} ${д.из_кэша}/${д.чанков}`) : [];
  console.log('сборка, из кэша:', изКэша.join(', '));
  должно(сборка && !сборка.ошибка, 'сборка не удалась');
  await подпись('Сцена 2', 'Векторы уже посчитаны при первой сборке (≈ час на CPU) и лежат в кэше по хэшу текста — пересборка идёт секунды');
  await подождать(ПАУЗА * 2);
  await к('#состояние-карточка');
  await подпись('Сцена 2', 'Два индекса в одном файле index/docs.db: у каждого эмбеддер, размерность вектора 1024, число чанков');
  await подождать(ПАУЗА * 2);

  // --- сцена 3: поиск и метаданные -------------------------------------------
  await подпись('Сцена 3', 'Поиск по смыслу: один вектор запроса, две стратегии рядом. У находки — файл, раздел, пункт, страницы, chunk_id');
  await к('#запрос', 'start');
  const оплата = await поискНаСтранице('в какой срок абонент оплачивает услуги связи каждый месяц');
  должно((оплата['структура'] || [])[0]?.includes('20 числа'), '«структура» не поставила пункт о 20 числе первым');
  await подождать(ПАУЗА * 3);

  // --- сцена 4: граница чанка ------------------------------------------------
  await подпись('Сцена 4', 'Чанк «фикс» целиком: серым — конец предыдущего и начало следующего. Граница идёт по счёту токенов, начало повторяет конец соседа');
  const фикс = await открытьЧанк('фикс', 0);
  должно(!!фикс, 'чанк «фикс» не открылся');
  await подождать(ПАУЗА * 3);
  await к('#находки', 'start');
  await подпись('Сцена 4', 'Чанк «структура»: начинается с номера пункта 3.5 и не выходит за раздел «3. Порядок расчетов»');
  const структура = await открытьЧанк('структура', 0, '20 числа');
  должно(!!структура, 'чанк «структура» не открылся');
  await подождать(ПАУЗА * 3);

  // --- сцена 5: поиск по скану -----------------------------------------------
  await к('#запрос', 'start');
  await подпись('Сцена 5', 'Вопрос к договору-скану: «структура» знает, что чанк из договора с ВымпелКомом, а у «фикс» первым идёт похожий пункт Ростелекома');
  const скан = await поискНаСтранице('какой штраф ВымпелКом берёт за просрочку оплаты', 'обе', 3);
  должно(Object.values(скан).flat().some(т => т.includes('OCR')), 'находки из скана не помечены OCR');
  должно((скан['структура'] || [])[0]?.includes('Вымпелком'), '«структура» не поставила скан ВымпелКома первым');
  await подождать(ПАУЗА * 3);

  // --- сцена 6: сравнение ----------------------------------------------------
  await к('#сравнить', 'start');
  await подпись('Сцена 6', 'Сравнение стратегий: как нарезано и как находится на 20+ эталонных вопросах (hit@k, MRR)');
  const сравнение = await работаИндекса('сравнить', 600);
  должно(сравнение && !сравнение.ошибка, 'сравнение не отработало');
  await ждатьТекст('сравнение', 'Как находится', 20);
  await к('#сравнение', 'start');
  const лучшая = сравнение && сравнение.итог ? сравнение.итог.лучшая : '';
  console.log('лучшая по MRR:', лучшая);
  await подождать(ПАУЗА * 3);
  await выполнить(`window.scrollBy({top: 420, behavior: 'smooth'}), 'ок'`);
  await подпись('Сцена 6', `По вопросам: где каждая стратегия поставила опорный фрагмент. По MRR лучше «${лучшая}»`);
  await подождать(ПАУЗА * 3);

  // --- сцена 7: индекс в работе агента ---------------------------------------
  await выполнить(`(() => { location.href = '/'; return 'ок'; })()`);
  await подождать(3500);
  await подпись('Сцена 7', 'Задел на неделю: агент ищет в индексе сам — через MCP-сервер docs, и называет источник');
  await выполнить(`(() => {
    const с = document.getElementById('модель'); с.value = ${JSON.stringify(МОДЕЛЬ)};
    с.dispatchEvent(new Event('change'));
    document.getElementById('инструменты-серверы').value = 'docs';
    const г = document.getElementById('инструменты-вкл');
    г.checked = true; г.dispatchEvent(new Event('change')); return 'ок'; })()`);
  await ждатьТекст('инструменты-итог', 'инстр.', 90);
  await к('#чат', 'start');
  await подождать(ПАУЗА);
  await выполнить(`(() => {
    document.getElementById('ввод').value = 'Найди в документах: до какого числа мы должны оплачивать услуги связи по договору VPN с Ростелекомом? Назови файл, пункт и страницу.';
    return 'ок'; })()`);
  await подождать(1200);
  await выполнить(`(() => { document.getElementById('отправить').click(); return 'ок'; })()`);
  const ответ = await ждатьТекст('чат', 'docs__search_docs', 180);
  должно(ответ >= 0, 'агент не вызвал docs__search_docs');
  await подождать(ПАУЗА * 2);
  const чат = await выполнить(`document.getElementById('чат').innerText`);
  // Строка «↳» стоит ПОД ответом, поэтому ищем по всей переписке, а не в
  // хвосте после имени инструмента: там только сама строка вызова.
  должно((чат || '').includes('20 числа'), 'в ответе нет «20 числа»');
  // Ответ длинный, и чат сам уезжает к строкам «↳» внизу. Показываем сначала
  // начало ответа — там файл, пункт и «20 числа», — потом строку вызова.
  await выполнить(`(() => {
    const ответы = document.querySelectorAll('#чат .от-агента');
    const о = ответы[ответы.length - 1];
    if (о) о.scrollIntoView({behavior: 'smooth', block: 'start'});
    return 'ок'; })()`);
  await подпись('Сцена 7', 'Ответ по найденному фрагменту: файл договора VPN, пункт 3.5, страница 4 — «не позднее 20 числа»');
  await подождать(ПАУЗА * 3);
  await выполнить(`(() => { const ч = document.getElementById('чат'); ч.scrollTop = ч.scrollHeight; return 'ок'; })()`);
  await подпись('Сцена 7', 'Строка «↳» — вызов docs__search_docs: модель сама пошла в индекс через MCP-сервер docs');
  await подождать(ПАУЗА * 3);

  console.log('\n=== итог показа ===');
  console.log(замечания.length ? 'РАСХОЖДЕНИЯ:\n  ' + замечания.join('\n  ')
                               : 'показ прошёл без расхождений');
  if (ошибки.length) console.log('ошибки страницы:\n  ' + ошибки.join('\n  '));
  ws.close();
  process.exit(0);          // показ не проверка: код возврата всегда 0
}

if (РЕЖИМ === 'показ-20') {
  // Темп показа: с ним видно, что происходит, и запись не выглядит нарезкой.
  // PAUSE, а не ПАУЗА: имя приходит из bash, а он кириллицу в именах
  // переменных окружения не принимает.
  const ПАУЗА = Number(process.env.PAUSE || 3000);
  const замечания = [];

  const подождать = мс => new Promise(р => setTimeout(р, мс));
  // «Отклонить» спрашивает причину через prompt(). Модальный диалог держит
  // страницу, и дальше показ идёт вслепую: клики уходят в никуда. Отвечаем на
  // диалог сами — заодно в записи видно, ПОЧЕМУ человек отказал.
  await зов('Page.enable');
  ws.addEventListener('message', (событие) => {
    const м = JSON.parse(событие.data);
    if (м.method === 'Page.javascriptDialogOpening') {
      зов('Page.handleJavaScriptDialog', {
        accept: true,
        promptText: 'Задачу не читали — текст комментария был бы выдумкой',
      });
    }
  });
  const ждатьЧат = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await подождать(1000);
      const есть = await выполнить(
        `document.getElementById('чат').innerText.includes(${JSON.stringify(признак)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    замечания.push(`не дождались в переписке: ${признак}`);
    return -1;
  };
  // Заявка появляется в панели не мгновенно: ответ флоу приходит раньше, чем
  // карточка перерисуется. Без ожидания «первая же кнопка» — это лотерея, и
  // прогон уже уезжал в сторону именно так.
  const ждатьЗаявку = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      const текст = await выполнить(
        `(document.getElementById('заявки-панель') || {}).innerText || ''`);
      if ((текст || '').includes(признак)) return (Date.now() - начало) / 1000;
      await подождать(1000);
    }
    замечания.push(`заявка «${признак}» так и не появилась в панели`);
    return -1;
  };
  // Итог маршрута ждём по тексту, а не по таймеру: пока идёт круг выбора
  // серверов, в том же месте стоит «Считаю маршрут…», и по таймеру показ
  // читал именно заглушку.
  const ждатьМаршрут = async (секунд = 60) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      const текст = await выполнить(
        `(document.getElementById('маршрут-итог') || {}).innerText || ''`);
      if ((текст || '').includes('маршрут (')) return текст;
      await подождать(1000);
    }
    замечания.push('маршрут так и не посчитался');
    return '';
  };
  // Подпись внизу страницы: сцена и одна фраза о том, что показывается. В
  // записи без неё непонятно, на что смотреть, а звука у записи нет.
  const подпись = async (сцена, текст) => {
    await выполнить(`(() => {
      let п = document.getElementById('подпись-показа');
      if (!п) {
        п = document.createElement('div');
        п.id = 'подпись-показа';
        п.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
          'background:#0f172a;color:#f8fafc;padding:10px 18px;font:16px/1.45 ' +
          'system-ui,sans-serif;box-shadow:0 -2px 10px rgba(0,0,0,.25)';
        document.body.appendChild(п);
      }
      п.innerHTML = '<b>' + ${JSON.stringify(сцена)} + '</b> — ' + ${JSON.stringify(текст)};
      return 'ок'; })()`);
    console.log(`\n=== ${сцена} — ${текст}`);
  };
  const кпанели = async () => выполнить(
    `(document.getElementById('оркестр-панель') || {}).scrollIntoView && ` +
    `document.getElementById('оркестр-панель').scrollIntoView({block: 'start'}), 'ок'`);
  const раскрыть = async () => выполнить(`(() => {
    document.querySelectorAll('#оркестр-панель details').forEach(д => { д.open = true; });
    return 'ок'; })()`);
  const панель = async () => {
    await раскрыть();
    return выполнить(`(() => {
      const у = document.getElementById('оркестр-панель');
      return у ? у.innerText.replace(/\\s+/g, ' ') : ''; })()`);
  };
  const спросить = async (текст, кнопка) => {
    // Значение ставится отдельным шагом и с паузой: в записи должно быть видно,
    // что в поле появился запрос, и только потом нажалась кнопка.
    await выполнить(`(() => {
      document.getElementById('флоу-вопрос').value = ${JSON.stringify(текст)};
      return 'ок'; })()`);
    await подождать(800);
    return выполнить(`(() => {
      document.getElementById(${JSON.stringify(кнопка)}).click(); return 'ок'; })()`);
  };
  // Продолжаем именно нужный флоу: карточек в панели несколько, и «первая
  // кнопка» — это лотерея. У кнопки есть data-номер, им и пользуемся.
  const продолжить = async номер => выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#оркестр-панель .флоу-продолжить'))
      .find(к => к.dataset['номер'] === String(${номер}));
    if (к) к.click(); return к ? 'нажал' : 'нет кнопки'; })()`);
  // Решение по заявке ищется по её ТЕКСТУ: заявок в панели может висеть
  // несколько, и «первая кнопка» однажды подтвердит чужую. Признак — кусок
  // текста комментария, он у каждой сцены свой.
  const решить = async (какая, признак = '') => выполнить(`(() => {
    const класс = ${JSON.stringify(какая)} === 'Подтвердить' ? '.заявка-да' : '.заявка-нет';
    const карточки = Array.from(document.querySelectorAll('#заявки-панель .заявка'));
    const нужная = ${JSON.stringify(признак)}
      ? карточки.find(к => к.innerText.includes(${JSON.stringify(признак)}))
      : карточки[0];
    const к = нужная && нужная.querySelector(класс);
    if (к) к.click();
    return к ? 'нажал' : 'нечего решать'; })()`);
  const должно = (условие, чего) => { if (!условие) замечания.push(чего); };

  // --- сцена 1: из чего агент выбирает -------------------------------------
  await подождать(2000);
  await подпись('Сцена 1', 'Девять MCP-серверов и правила порядка — то, из чего агент выбирает');
  await кпанели();
  await раскрыть();
  const сцена1 = await панель();
  console.log(сцена1.slice(0, 420));
  должно(сцена1.includes('только по явной просьбе'),
         'в панели не видно пометки «только по явной просьбе»');
  должно(сцена1.includes('tracker__add_comment'), 'в панели не видно правил порядка');
  await подождать(ПАУЗА * 2);

  // --- сцена 2: маршрут ------------------------------------------------------
  await подпись('Сцена 2', 'Маршрут: под запрос поднимаются не все серверы, и видно почему');
  await спросить('Посмотри в настоящем трекере, что там по задачам', 'маршрут-показать');
  const маршрут1 = await ждатьМаршрут();
  console.log('маршрут:', (маршрут1 || '').replace(/\s+/g, ' ').slice(0, 300));
  должно((маршрут1 || '').includes('tracker-real'),
         'по явной просьбе не выбран боевой Трекер');
  await подождать(ПАУЗА);
  await подпись('Сцена 2', 'А на вопрос без внешних данных не поднимается ни один сервер');
  await спросить('Привет, как дела?', 'маршрут-показать');
  const маршрут2 = await ждатьМаршрут();
  console.log('маршрут:', (маршрут2 || '').replace(/\s+/g, ' ').slice(0, 200));
  должно((маршрут2 || '').includes('ни одного'),
         'на «как дела» всё равно выбраны серверы');
  await подождать(ПАУЗА);

  // --- сцена 3: длинный флоу -------------------------------------------------
  await подпись('Сцена 3', 'Длинный флоу: модель называет план, агент его исполняет');
  await выполнить(`(() => {
    const м = document.getElementById('модель');
    м.value = ${JSON.stringify(МОДЕЛЬ)}; м.dispatchEvent(new Event('change'));
    const п = document.getElementById('инструменты-вкл');
    if (!п.checked) { п.checked = true; п.dispatchEvent(new Event('change')); }
    const поле = document.getElementById('инструменты-серверы');
    поле.value = '*';
    return 'ок'; })()`);
  await подождать(2500);
  await спросить('Разбери задачу MIG-5: прочитай её в трекере, подними, что о ней знает '
                 + 'наша память, и сохрани короткий отчёт файлом', 'флоу-пуск');
  console.log('флоу №1 прошёл за', await ждатьЧат('флоу №1 — готов', 240), 'с');
  await кпанели();
  const сцена3 = await панель();
  console.log(сцена3.slice(0, 700));
  должно(сцена3.includes('план:'), 'в карточке нет плана');
  должно(сцена3.includes('по плану'), 'шаги не помечены «по плану»');
  должно(сцена3.includes('порядок соблюдён'), 'сверка не подтвердила порядок');
  await подпись('Сцена 3', 'Сверка плана с фактом — так проверяются выбор и порядок вызовов');
  await подождать(ПАУЗА * 2);

  // --- сцена 4: порядок и отказ человека -------------------------------------
  await подпись('Сцена 4', 'Порядок вызовов проверяет код: писать в задачу, не прочитав её, нельзя');
  await спросить('Добавь к задаче MIG-5 комментарий «Проверка порядка» — ничего не читая',
                 'флоу-пуск');
  console.log('флоу №2 прошёл за', await ждатьЧат('флоу №2', 240), 'с');
  await подождать(2500);
  const сцена4 = await панель();
  const отклонён = сцена4.includes('отклонён');
  console.log(отклонён ? 'шаг отклонён правилом порядка'
                       : 'модель сама начала с чтения задачи (правило сработало в наставлении)');
  await подождать(ПАУЗА);
  // Этот флоу упёрся в подтверждение — и здесь человек говорит «нет»: текст
  // комментария был бы выдумкой. Заодно сцена не оставляет висящую заявку,
  // иначе следующая сцена подтвердила бы не ту.
  await подпись('Сцена 4', 'А на такой комментарий человек отвечает «нет» — и флоу честно это пишет');
  console.log('заявка появилась за', await ждатьЗаявку('Проверка порядка', 60), 'с');
  const отказ = await решить('Отклонить', 'Проверка порядка');
  console.log('отклонение заявки:', отказ);
  должно(отказ === 'нажал', 'заявку «Проверка порядка» не на чем было отклонить');
  await подождать(2500);
  const после4 = await панель();
  должно(после4.includes('отклонён') || после4.includes('ожидание')
         || после4.includes('отказано'),
         'в шагах не видно ни отклонения по порядку, ни ожидания человека');
  // Флоу №2 дальше не продолжаем: человек сказал «нет», и этого для показа
  // достаточно. Продолжение попросило бы у модели тот же вызов снова, а сцену
  // 6 это оставило бы с двумя ждущими заявками.
  await подождать(ПАУЗА);

  // --- сцена 5: пауза на подтверждении --------------------------------------
  await подпись('Сцена 5', 'Перед изменением чужой системы флоу останавливается и ждёт человека');
  await спросить('Прочитай задачу MIG-5 в трекере и добавь к ней комментарий '
                 + '«Разобрано показом в браузере»', 'флоу-пуск');
  console.log('флоу №3 встал за', await ждатьЧат('флоу №3 — ждёт', 240), 'с');
  await подождать(2500);
  const сцена5 = await панель();
  должно(сцена5.includes('ожидание'), 'шаг комментария не помечен ожиданием');
  await подождать(ПАУЗА);

  // --- сцена 6: подтверждение, продолжение, живой протокол ------------------
  await подпись('Сцена 6', 'Человек подтверждает — и флоу продолжается с того же места');
  console.log('заявка появилась за',
              await ждатьЗаявку('Разобрано показом в браузере', 60), 'с');
  const согласие = await решить('Подтвердить', 'Разобрано показом в браузере');
  console.log('подтверждение заявки:', согласие);
  должно(согласие === 'нажал', 'заявку «Разобрано показом в браузере» не на чем было подтвердить');
  console.log('вызов исполнен за', await ждатьЧат('"добавлен": true', 120), 'с');
  await подождать(2000);
  const нажал = await продолжить(3);
  должно(нажал === 'нажал', 'у остановленного флоу нет кнопки «Продолжить флоу»');
  console.log('флоу продолжен за', await ждатьЧат('флоу №3 — готов', 240), 'с');
  await подождать(2000);
  const сцена6 = await панель();
  должно(сцена6.includes('подтверждена заявка'),
         'в шагах не видно, что вызов записан результатом заявки');
  await подпись('Сцена 6', 'Протокол лежит в базе — перезагрузим страницу и убедимся');
  await подождать(ПАУЗА);
  await зов('Page.reload', {ignoreCache: false});
  await подождать(6000);
  const после = await панель();
  должно(после.includes('сверка'), 'после перезагрузки протокол флоу пропал');
  await подпись('Итог', 'Серверы выбрал агент, порядок проверил код, перед чужой системой он '
                + 'остановился и спросил человека');
  await подождать(ПАУЗА * 2);

  console.log('\n=== итог показа ===');
  console.log('сообщений об ошибке в чате:',
              await выполнить(`document.querySelectorAll('#чат .ошибка').length`));
  console.log(замечания.length ? 'РАСХОЖДЕНИЯ:\n  ' + замечания.join('\n  ')
                               : 'показ прошёл без расхождений');
  if (ошибки.length) console.log('исключения страницы:\n  ' + ошибки.join('\n  '));
  ws.close();
  process.exit(0);          // показ не проверка: код возврата всегда 0
}

if (РЕЖИМ === 'оркестр') {
  const ждатьУзел = async (селектор, секунд, признак = '') => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const текст = await выполнить(`(() => {
        const у = document.querySelector(${JSON.stringify(селектор)});
        return у ? у.innerText : ''; })()`);
      if (текст && (!признак || текст.includes(признак))) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`не дождались ${селектор}${признак ? ' с «' + признак + '»' : ''}`);
    return -1;
  };
  const ждатьЧат = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const есть = await выполнить(
        `document.getElementById('чат').innerText.includes(${JSON.stringify(признак)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`в чате не дождались: ${признак}`);
    return -1;
  };
  // Свёрнутый <details> в innerText не попадает вовсе — раскрываем все карточки
  // флоу так же, как это делает человек, которому нужен протокол.
  const раскрыть = async () => выполнить(`(() => {
    document.querySelectorAll('#оркестр-панель details').forEach(д => { д.open = true; });
    return 'ок'; })()`);
  const панельТекстом = async () => {
    await раскрыть();
    return выполнить(`(() => {
      const у = document.getElementById('оркестр-панель');
      return у ? у.innerText.replace(/\\s+/g, ' ') : ''; })()`);
  };
  const спросить = async (текст, кнопка) => выполнить(`(() => {
    document.getElementById('флоу-вопрос').value = ${JSON.stringify(текст)};
    document.getElementById(${JSON.stringify(кнопка)}).click(); return 'ок'; })()`);

  // 1. Панель рисуется сама: серверы, правила порядка, бюджет.
  console.log('панель появилась за', await ждатьУзел('#оркестр-панель .карточка', 30), 'с');
  const панель = await панельТекстом();
  console.log('панель:', панель.slice(0, 500));
  for (const признак of ['Оркестр', 'бюджет', 'маршрутизация']) {
    if (!панель.includes(признак)) ошибки.push(`в панели нет «${признак}»`);
  }
  for (const сервер of ['tracker', 'pipeline', 'deepwiki']) {
    if (!панель.includes(сервер)) ошибки.push(`в списке серверов нет «${сервер}»`);
  }
  if (!панель.includes('только по явной просьбе')) {
    ошибки.push('не видно пометки «только по явной просьбе» у боевого Трекера');
  }
  if (!панель.includes('tracker__add_comment')) {
    ошибки.push('в панели нет правил порядка');
  }

  // 2. Маршрут: кнопка считает его, ничего не вызывая.
  await спросить('Посмотри в настоящем трекере, что там по задачам', 'маршрут-показать');
  // Признак — «маршрут (», а не «маршрут»: пока идёт запрос, в этом же месте
  // стоит «Считаю маршрут…», и по короткому признаку проверка читала заглушку.
  console.log('маршрут посчитан за', await ждатьУзел('#маршрут-итог', 60, 'маршрут ('), 'с');
  const маршрут = await выполнить(
    `(document.getElementById('маршрут-итог') || {}).innerText || ''`);
  console.log('маршрут:', (маршрут || '').replace(/\s+/g, ' ').slice(0, 300));
  if (!(маршрут || '').includes('tracker-real')) {
    ошибки.push('по явной просьбе не выбран боевой Трекер');
  }
  if (!(маршрут || '').includes('заменён сервером')) {
    ошибки.push('мок не вытеснен: не видно причины «заменён сервером»');
  }

  // 3. Включаем инструменты и ведём длинный флоу.
  await выполнить(`(() => {
    const п = document.getElementById('инструменты-вкл');
    if (п && !п.checked) { п.checked = true; п.dispatchEvent(new Event('change')); }
    const поле = document.getElementById('инструменты-серверы');
    if (поле) { поле.value = 'tracker,pipeline'; поле.dispatchEvent(new Event('change')); }
    return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 1500));
  await спросить('Разбери задачу MIG-5: прочитай её в трекере, подними, что о ней '
                 + 'знает наша память, и сохрани короткий отчёт файлом', 'флоу-пуск');
  // Ждём именно «флоу №1 — готов»: строка «↳ флоу» появляется и у сбоя, и
  // второй клик по «Запустить флоу» поверх незаконченного первого — это два
  // флоу на одном агенте разом, чего страница делать не должна.
  console.log('флоу прошёл за', await ждатьЧат('флоу №1 — готов', 300), 'с');
  const послеФлоу = await панельТекстом();
  console.log('карточка флоу:', послеФлоу.slice(0, 700));
  for (const признак of ['план:', 'по плану', 'сверка:']) {
    if (!послеФлоу.includes(признак)) ошибки.push(`в карточке флоу нет «${признак}»`);
  }
  if (!послеФлоу.includes('tracker__get_issue')) {
    ошибки.push('флоу не читал задачу в трекере');
  }
  if (!послеФлоу.includes('порядок соблюдён')) {
    ошибки.push('сверка не подтвердила порядок вызовов');
  }

  // 4. Флоу с меняющим шагом: пауза, подтверждение, продолжение.
  await спросить('Прочитай задачу MIG-5 в трекере и добавь к ней комментарий '
                 + '«Разобрано браузерной проверкой»', 'флоу-пуск');
  console.log('флоу остановился за', await ждатьЧат('флоу №2 — ждёт', 300), 'с');
  await new Promise(р => setTimeout(р, 1500));
  const заявки = await ждатьУзел('#заявки-панель', 30, 'tracker__add_comment');
  console.log('заявка появилась за', заявки, 'с');
  await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#заявки-панель button'))
      .find(к => к.textContent.includes('Подтвердить'));
    if (к) к.click(); return к ? 'нажал' : 'нет кнопки'; })()`);
  console.log('вызов исполнен за', await ждатьЧат('"добавлен": true', 120), 'с');
  await new Promise(р => setTimeout(р, 2000));
  const нажал = await выполнить(`(() => {
    const к = document.querySelector('#оркестр-панель .флоу-продолжить');
    if (к) к.click(); return к ? 'нажал' : 'нет кнопки'; })()`);
  console.log('кнопка «Продолжить флоу»:', нажал);
  if (нажал !== 'нажал') ошибки.push('у остановленного флоу нет кнопки «Продолжить флоу»');
  // Признак с номером: «— готов» уже сказано про первый флоу, и короткое
  // ожидание срабатывало на чужой строке, не дождавшись продолжения второго.
  console.log('флоу продолжен за', await ждатьЧат('флоу №2 — готов', 300), 'с');
  const итог = await панельТекстом();
  if (!итог.includes('подтверждена заявка')) {
    ошибки.push('в шагах не видно, что вызов записан результатом заявки');
  }
  if (итог.includes('ждёт подтверждения') && !итог.includes('готов')) {
    ошибки.push('флоу так и остался в ожидании');
  }

  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:',
              await выполнить(`document.querySelectorAll('#чат .ошибка').length`));
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

if (РЕЖИМ === 'конвейер') {
  const ждатьУзел = async (селектор, секунд, признак = '') => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const текст = await выполнить(`(() => {
        const у = document.querySelector(${JSON.stringify(селектор)});
        return у ? у.innerText : ''; })()`);
      if (текст && (!признак || текст.includes(признак))) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`не дождались ${селектор}${признак ? ' с «' + признак + '»' : ''}`);
    return -1;
  };
  const ждатьЧат = async (признак, секунд) => {
    const начало = Date.now();
    while (Date.now() - начало < секунд * 1000) {
      await new Promise(р => setTimeout(р, 1000));
      const есть = await выполнить(
        `document.getElementById('чат').innerText.includes(${JSON.stringify(признак)})`);
      if (есть) return (Date.now() - начало) / 1000;
    }
    ошибки.push(`в чате не дождались: ${признак}`);
    return -1;
  };
  // Свёрнутый <details> в innerText не попадает вовсе — раскрываем так же, как
  // это делает человек, которому нужен протокол.
  const раскрыть = async () => выполнить(`(() => {
    const д = document.querySelector('#конвейеры-панель details');
    if (д) д.open = true; return 'ок'; })()`);
  const протокол = async () => {
    await раскрыть();
    return выполнить(`(() => {
      const у = document.querySelector('#конвейеры-панель details');
      return у ? у.innerText.replace(/\\s+/g, ' ') : ''; })()`);
  };
  const запустить = async (имя, вход) => выполнить(`(() => {
    document.getElementById('конвейер-имя').value = ${JSON.stringify(имя)};
    document.getElementById('конвейер-вход').value = ${JSON.stringify(вход)};
    document.getElementById('конвейер-пуск').click(); return 'ок'; })()`);

  // 1. Панель рисуется сама и показывает описанные цепочки.
  console.log('панель появилась за', await ждатьУзел('#конвейеры-панель .карточка', 30), 'с');
  const описания = await выполнить(`(() => {
    const у = document.getElementById('конвейеры-панель');
    return у ? у.innerText.replace(/\\s+/g, ' ').slice(0, 1400) : ''; })()`);
  console.log('панель:', описания.slice(0, 600));
  for (const имя of ['отчёт', 'память', 'в-трекер']) {
    if (!описания.includes(имя)) ошибки.push(`в панели нет конвейера «${имя}»`);
  }
  if (!описания.includes('tracker__list_issues')) {
    ошибки.push('в панели не видно инструментов цепочки');
  }

  // 2. Короткая цепочка: главное — строки «получил …» под вторым шагом.
  await запустить('память', '{"запрос": "PostGIS индексы"}');
  console.log('цепочка «память» выполнена за', await ждатьЧат('конвейер «память»', 120), 'с');
  const короткий = await протокол();
  console.log('протокол:', короткий.slice(0, 500));
  if (!короткий.includes('готов')) ошибки.push('прогон «память» не готов');
  if (!короткий.includes('получил вход.запрос')) {
    ошибки.push('не видно, что первый шаг получил вход конвейера');
  }
  if (!короткий.includes('получил находки.данные.находки')) {
    ошибки.push('не видно передачи данных между шагами');
  }

  // 3. Полная цепочка: четыре шага, три сервера, файл на диске.
  await запустить('отчёт', '');
  console.log('цепочка «отчёт» выполнена за', await ждатьЧат('конвейер «отчёт»', 240), 'с');
  const полный = await протокол();
  console.log('протокол:', полный.slice(0, 700));
  for (const кусок of ['tracker__list_issues', 'pipeline__search',
                       'pipeline__summarize', 'pipeline__save_to_file']) {
    if (!полный.includes(кусок)) ошибки.push(`в протоколе нет шага ${кусок}`);
  }
  if (!полный.includes('выжимка.данные.текст')) {
    ошибки.push('не видно, что выжимка доехала до шага сохранения');
  }
  if (!/итог: .*\.md/.test(полный)) ошибки.push('в протоколе нет пути файла отчёта');

  // 4. Сбой посреди цепочки: она останавливается, и видно, на чём.
  await запустить('в-трекер', '{"задача": "MIG-999"}');
  console.log('сбой показан за', await ждатьЧат('сбой', 180), 'с');
  const сбойный = await протокол();
  console.log('протокол сбоя:', сбойный.slice(0, 400));
  if (!сбойный.includes('2/3') && !сбойный.includes('2 / 3')) {
    ошибки.push('цепочка не остановилась на упавшем шаге');
  }
  if (!сбойный.includes('ошибка')) ошибки.push('не видно причины сбоя');

  // 5. Модель делает то же самое одним вызовом.
  await выполнить(`(() => {
    document.getElementById('инструменты-серверы').value = 'pipeline,tracker';
    const г = document.getElementById('инструменты-вкл');
    г.checked = true; г.dispatchEvent(new Event('change')); return 'ок'; })()`);
  const предел = Date.now() + 120000;
  let итогВключения = '';
  while (Date.now() < предел) {
    await new Promise(р => setTimeout(р, 1000));
    итогВключения = await выполнить(`document.getElementById('инструменты-итог').innerText`);
    if (итогВключения && !итогВключения.includes('подключаюсь')) break;
  }
  console.log('инструменты включены:', итогВключения);
  await выполнить(`(() => {
    document.getElementById('ввод').value =
      'Собери отчёт о состоянии миграции и сохрани его файлом. ' +
      'Посмотри, какие есть конвейеры, и выполни подходящий.';
    document.getElementById('отправить').click(); return 'ок'; })()`);
  // Ждём именно имя инструмента: строка «↳ конвейер …» уже есть в чате после
  // запусков кнопкой, и по ней проверка сработала бы раньше ответа модели.
  console.log('модель запустила цепочку за',
              await ждатьЧат('pipeline__run_pipeline', 240), 'с');
  const вызовы = await выполнить(`Array.from(document.querySelectorAll('#чат .вызов'))
    .map(у => у.innerText.slice(0, 140))`);
  for (const в of вызовы || []) console.log('   ', в);
  if (!(вызовы || []).some(в => в.includes('run_pipeline'))) {
    ошибки.push('модель не запустила конвейер одним вызовом');
  }

  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:',
              await выполнить(`document.querySelectorAll('#чат .ошибка').length`));
  console.log(ошибки.length ? 'ОШИБКИ:\n  ' + ошибки.join('\n  ') : 'ошибок нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

if (РЕЖИМ === 'переходы') {
  // Отдельный ход: здесь проверяется жизненный цикл задачи, а не сценарий.
  // Модель зовётся один раз — на составление плана, без которого нечего
  // утверждать, а значит, и нечего открывать.
  const имя = 'ворота-' + Date.now().toString().slice(-6);
  await выполнить(`(() => {
    document.getElementById('новая-задача').value = ${JSON.stringify(имя)};
    document.getElementById('название').value = 'проверка ворот из браузера';
    document.getElementById('создать').click(); return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 2500));

  const кнопки = await выполнить(`Array.from(
    document.querySelectorAll('#переходы button')).map(к => к.textContent.trim())`);
  console.log('кнопки переходов:', кнопки);
  const запертых = (кнопки || []).filter(т => т.includes('🔒')).length;
  console.log('запертых кнопок:', запертых);
  if (!кнопки || кнопки.length < 3) ошибки.push('на странице нет кнопок переходов');
  if (!запертых) ошибки.push('все переходы показаны открытыми — ворота не видны');

  // Нажимаем запертый переход: страница должна показать разбор отказа.
  await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#переходы button'))
      .find(к => к.textContent.includes('🔒'));
    if (к) к.click(); return к ? к.textContent : 'нет'; })()`);
  await new Promise(р => setTimeout(р, 2000));
  const отказов = await выполнить(`document.querySelectorAll('#чат .отказ').length`);
  const текстОтказа = await выполнить(
    `(document.querySelector('#чат .отказ') || {}).innerText || ''`);
  console.log('отказов в чате:', отказов);
  console.log('первые строки отказа:\n' + (текстОтказа || '(пусто)').split('\n').slice(0, 6).join('\n'));
  if (!отказов) ошибки.push('нажатие запертого перехода не дало отказа');

  console.log('\nсоставляю план (один вызов модели)…');
  await выполнить(`document.getElementById('план').click(); 'ок'`);
  const началоПлана = Date.now();
  while (Date.now() - началоПлана < 180000) {
    await new Promise(р => setTimeout(р, 3000));
    const занято = await выполнить(`document.getElementById('отправить').disabled`);
    if (!занято && Date.now() - началоПлана > 6000) break;
  }
  const доУтверждения = await выполнить(`document.getElementById('ворота').innerText`);
  console.log('\n=== ворота до утверждения ===\n' + (доУтверждения || '(пусто)'));

  await выполнить(`document.getElementById('утвердить-план').click(); 'ок'`);
  await new Promise(р => setTimeout(р, 2500));
  const послеУтверждения = await выполнить(`document.getElementById('ворота').innerText`);
  console.log('\n=== ворота после утверждения ===\n' + (послеУтверждения || '(пусто)'));

  const открыт = await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#переходы button'))
      .find(к => к.textContent.includes('исполнение'));
    return к ? к.textContent.trim() : 'нет кнопки'; })()`);
  console.log('кнопка исполнения после утверждения:', открыт);
  if (открыт.includes('🔒')) ошибки.push('после утверждения плана переход остался закрытым');

  // Шаги задачи, заведённой руками, закрывает кнопка «Шаг сделан»: без неё
  // условие «шаги доведены» из браузера не выполнить вовсе.
  await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#переходы button'))
      .find(к => к.textContent.includes('исполнение'));
    if (к) к.click(); return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 2000));
  const шаговДо = await выполнить(
    `document.querySelectorAll('.лента-шагов .готов').length`);
  await выполнить(`document.getElementById('шаг-значение').value = 'сделано из браузера';
                   document.getElementById('закрыть-шаг').click(); 'ок'`);
  await new Promise(р => setTimeout(р, 2000));
  const шаговПосле = await выполнить(
    `document.querySelectorAll('.лента-шагов .готов').length`);
  console.log(`шагов закрыто: было ${шаговДо}, стало ${шаговПосле}`);
  if (шаговПосле <= шаговДо) ошибки.push('кнопка «Шаг сделан» не закрыла шаг');

  const сбоиПереходов = await выполнить(`document.querySelectorAll('#чат .ошибка').length`);
  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:', сбоиПереходов);
  console.log('исключений JS:', ошибки.length ? ошибки : 'нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

await выполнить(`document.getElementById('ввод').value = ${JSON.stringify(ЗАПРОС)};
                 document.getElementById('отправить').click(); 'пуск'`);

const начало = Date.now();
let прошлое = '';
while (Date.now() - начало < ПРЕДЕЛ_МС) {
  await new Promise(р => setTimeout(р, 3000));
  const чат = await выполнить(`document.getElementById('чат').innerText`);
  const новое = (чат || '').slice(прошлое.length);
  if (новое.trim()) {
    for (const строка of новое.split('\n').filter(с => с.trim()).slice(0, 3)) {
      console.log(`[${((Date.now() - начало) / 1000).toFixed(0)} с] ${строка.slice(0, 110)}`);
    }
    прошлое = чат;
  }
  const занято = await выполнить(`document.getElementById('отправить').disabled`);
  if (!занято && Date.now() - начало > 8000) break;
}

if (РЕЖИМ === 'пауза') {
  const состояние = await выполнить(
    `document.getElementById('состояние-задачи').innerText`);
  console.log('\n=== состояние задачи на паузе ===\n' + (состояние || '(пусто)'));
  const наПаузе = (состояние || '').includes('на паузе');
  console.log('карточка показывает паузу:', наПаузе ? 'да' : 'НЕТ');

  console.log('\nжму «Продолжить»…');
  await выполнить(`document.getElementById('продолжить-задачу').click(); 'ок'`);
  const начало2 = Date.now();
  while (Date.now() - начало2 < 120000) {
    await new Promise(р => setTimeout(р, 3000));
    const занято = await выполнить(`document.getElementById('отправить').disabled`);
    if (!занято && Date.now() - начало2 > 6000) break;
  }
  const после = await выполнить(`document.getElementById('состояние-задачи').innerText`);
  console.log('\n=== состояние после «Продолжить» ===\n' + (после || '(пусто)'));
  console.log('состояние сдвинулось:', после !== состояние ? 'да' : 'НЕТ');
  if (!наПаузе || после === состояние) ошибки.push('пауза или продолжение не сработали');
}

const чат = await выполнить(`document.getElementById('чат').innerText`);
const отвечено = await выполнить(`document.querySelectorAll('#чат .от-агента').length`);
const сбои = await выполнить(`document.querySelectorAll('#чат .ошибка').length`);

console.log('\n=== итог ===');
console.log('символов в чате:', (чат || '').length);
console.log('ответов агента:', отвечено);
console.log('сообщений об ошибке в чате:', сбои);
console.log('исключений JS:', ошибки.length ? ошибки : 'нет');

ws.close();
// Ненулевой код — чтобы прогон годился и для проверки перед сдачей.
process.exit(ошибки.length || сбои || !отвечено ? 1 : 0);
