import http from 'k6/http';
import execution from 'k6/execution';
import { fail } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { baseUrl, chat, chatScenarios, chatThresholds, loginUsers, positiveInt } from './lib/capacity.js';

const users = JSON.parse(open(__ENV.USERS_FILE || '../data/test_users.json'));
const sample = open('../data/sample.txt');
const uploadCount = positiveInt('UPLOAD_COUNT', 1000);
const uploadVus = positiveInt('UPLOAD_VUS', 20);
const chatVus = positiveInt('CHAT_VUS', 20);
const duration = __ENV.DURATION || '5m';
const attempts = new Counter('upload_attempts');
const accepted = new Counter('uploads_accepted');
const principalBacklog = new Counter('uploads_principal_backlog');
const globalBacklog = new Counter('uploads_global_backlog');
const unexpected = new Rate('upload_unexpected_errors');
const admissionLatency = new Trend('upload_admission_latency', true);

export const options = {
  setupTimeout: '120s',
  scenarios: {
    uploads: { executor: 'shared-iterations', exec: 'upload', vus: uploadVus,
      iterations: uploadCount, maxDuration: duration, gracefulStop: '130s' },
    ...chatScenarios(chatVus, 'otherUsersChat', duration),
  },
  thresholds: {
    ...chatThresholds,
    upload_attempts: [`count==${uploadCount}`],
    uploads_accepted: ['count>0'],
    upload_unexpected_errors: ['rate<0.01'],
  },
};

export function setup() {
  if (!['admin', 'curator'].includes(users[0].role)) fail('The first user must be an admin or curator uploader');
  return loginUsers(users, chatVus + 1);
}

export function upload(data) {
  const id = `${data.runId}-${execution.scenario.iterationInTest}`;
  const response = http.post(`${baseUrl}/documents`, {
    file: http.file(`${sample}\nLoad test document ${id}\n`, `loadtest-${id}.txt`, 'text/plain'),
    visibility: 'internal_private', doc_domain: 'general',
  }, {
    headers: { Authorization: `Bearer ${data.tokens[0]}`, 'Idempotency-Key': `loadtest-${id}` },
    timeout: '120s', tags: { name: 'upload_document', kind: 'upload' },
    responseCallback: http.expectedStatuses(200, 429, 503),
  });
  let body = {};
  try { body = response.json(); } catch { /* Non-JSON errors are unexpected. */ }
  const admitted = response.status === 200 && body.status === 'processing' && Number.isInteger(body.document_id);
  // Do not mistake nginx/rate-limit 429s or unrelated 503s for a working backlog guard.
  const backlog = [429, 503].includes(response.status)
    && body.message === 'Background job capacity reached; retry later'
    && Number(response.headers['Retry-After']) > 0;
  attempts.add(1);
  accepted.add(admitted ? 1 : 0);
  principalBacklog.add(backlog && response.status === 429 ? 1 : 0);
  globalBacklog.add(backlog && response.status === 503 ? 1 : 0);
  unexpected.add(!admitted && !backlog);
  admissionLatency.add(response.timings.duration);
}

export function otherUsersChat(data) {
  // Each participant keeps one account; token zero is reserved for uploads.
  const index = Number(__ENV.CHAT_USER_INDEX);
  chat(data.tokens[index + 1], `${data.runId}-${execution.scenario.name}-${execution.scenario.iterationInTest}`);
}
