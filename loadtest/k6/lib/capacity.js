import http from 'k6/http';
import { check, fail, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

export const baseUrl = (__ENV.BASE_URL || 'http://server:8001').replace(/\/$/, '');
const chatErrors = new Rate('chat_errors');
const chatLatency = new Trend('chat_latency', true);

export function positiveInt(name, fallback) {
  const value = Number(__ENV[name] || fallback);
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  return value;
}

export function loginUsers(users, count) {
  const selected = users.slice(0, count);
  if (selected.length < count || new Set(selected.map((u) => u.email.toLowerCase())).size !== count) {
    fail(`Need ${count} distinct users in USERS_FILE; run loadtest:setup-users first`);
  }
  const tokens = selected.map((user) => {
    const response = http.post(`${baseUrl}/auth/login`, JSON.stringify({ email: user.email, password: user.password }), {
      headers: { 'Content-Type': 'application/json' },
      timeout: '30s', tags: { name: 'auth_login' },
    });
    if (response.status !== 200) fail(`Login failed: HTTP ${response.status}`);
    const token = response.json('access_token');
    if (!token) fail('Login returned no access token');
    return token;
  });
  return { tokens, runId: String(Date.now()) };
}

export const chatThresholds = {
  chat_latency: [`p(95)<${positiveInt('CHAT_P95_MS', 120000)}`],
  chat_errors: ['rate<0.01'],
  'checks{kind:chat}': ['rate>0.99'],
};

export function chatScenarios(count, exec, duration) {
  const scenarios = {};
  for (let index = 0; index < count; index++) {
    scenarios[`chat_user_${index}`] = {
      executor: 'constant-vus', exec, vus: 1, duration, gracefulStop: '130s',
      env: { CHAT_USER_INDEX: String(index) },
    };
  }
  return scenarios;
}

export function chat(token, requestId) {
  let question = __ENV.CHAT_QUESTION || 'Какие документы есть в базе знаний и что в них описано?';
  // Exercise retrieval and generation instead of measuring repeated answer-cache hits.
  if (__ENV.BYPASS_ANSWER_CACHE !== 'false') question += `\nИдентификатор запроса: ${requestId}`;
  const response = http.post(`${baseUrl}/chat/sync`, JSON.stringify({ question }), {
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    timeout: '120s', tags: { name: 'chat_sync', kind: 'chat' },
  });
  let answer = '';
  if (response.status === 200) {
    try { answer = response.json('answer'); } catch { /* Invalid JSON fails the check below. */ }
  }
  const ok = check(response, {
    'chat returns a nonempty answer': (r) => r.status === 200 && typeof answer === 'string' && answer.trim().length > 0,
  }, { kind: 'chat' });
  chatErrors.add(!ok);
  chatLatency.add(response.timings.duration);
  sleep(Number(__ENV.CHAT_PAUSE_SECONDS || 1));
}
