import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8001';

const users = new SharedArray('users', function () {
  try {
    return JSON.parse(open('../data/test_users.json'));
  } catch (e) {
    return [{ email: 'admin', password: 'admin' }];
  }
});

const questions = new SharedArray('questions', function () {
  try {
    return JSON.parse(open('../data/test_questions.json'));
  } catch (e) {
    return ['Что такое RAG?', 'Объясни архитектуру системы', 'Какие документы загружены?'];
  }
});

// Pre-login all users during setup; store as SharedArray of [email, token] pairs
let _tokensByEmail = {};

export function setupTokens() {
  const batchSize = 25;
  const results = [];

  for (let i = 0; i < users.length; i += batchSize) {
    const batch = users.slice(i, i + batchSize);
    const batchResults = http.batch(
      batch.map((u) => ({
        method: 'POST',
        url: `${BASE_URL}/auth/login`,
        body: JSON.stringify({ email: u.email, password: u.password }),
        params: { headers: { 'Content-Type': 'application/json' }, tags: { name: 'auth_login' } },
      }))
    );

    batch.forEach((u, idx) => {
      const res = batchResults[idx];
      if (res.status === 200) {
        try {
          const token = res.json('access_token');
          results.push({ email: u.email, token });
        } catch {}
      }
    });
  }

  const successCount = results.length;
  console.log(`Setup: pre-logged-in ${successCount}/${users.length} users`);
  return results;
}

export function initTokens(setupResult) {
  // Call from default/exec function: initTokens(setupData)
  // Build lookup from setup result
  _tokensByEmail = {};
  if (setupResult && setupResult.tokens) {
    for (const t of setupResult.tokens) {
      _tokensByEmail[t.email] = t.token;
    }
  }
}

export function getToken(vuIndex) {
  const user = users[vuIndex % users.length];
  return _tokensByEmail[user.email] || null;
}

export function getRandomQuestion() {
  return questions[Math.floor(Math.random() * questions.length)];
}

export function getRandomUser(vuIndex) {
  return users[vuIndex % users.length];
}

export function chatSync(token, question) {
  const headers = {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
  const start = Date.now();
  const res = http.post(
    `${BASE_URL}/chat/sync`,
    JSON.stringify({ question }),
    { headers, timeout: '120s', tags: { name: 'chat_sync' } }
  );
  const latency = Date.now() - start;

  check(res, {
    'chat sync 200': (r) => r.status === 200,
    'chat sync has answer': (r) => {
      try {
        return r.json('answer') && r.json('answer').length > 0;
      } catch {
        return false;
      }
    },
  });

  return { status: res.status, latency };
}

export function listDocuments(token) {
  const headers = { Authorization: `Bearer ${token}` };
  const res = http.get(`${BASE_URL}/documents`, {
    headers,
    tags: { name: 'documents_list' },
  });
  check(res, { 'documents list 200': (r) => r.status === 200 });
  return { status: res.status };
}
