import { Counter, Rate, Trend } from 'k6/metrics';
import { sleep } from 'k6';
import { setupTokens, initTokens, getToken, getRandomQuestion, chatSync, listDocuments } from './lib/helpers.js';

const rateLimited = new Counter('rate_limited_429');
const chatLatency = new Trend('chat_sync_latency', true);
const failedRequests = new Rate('failed_requests');

export const options = {
  setupTimeout: '120s',
  scenarios: {
    soak: {
      executor: 'constant-vus',
      exec: 'chatScenario',
      vus: 200,
      duration: '1h',
      gracefulStop: '2m',
    },
    doc_list_soak: {
      executor: 'constant-arrival-rate',
      exec: 'listDocsScenario',
      rate: 5,
      timeUnit: '1s',
      duration: '1h',
      preAllocatedVUs: 10,
      maxVUs: 30,
    },
  },
  thresholds: {
    'chat_sync_latency': ['p(95)<120000'],
    'http_req_failed': ['rate<0.05'],
  },
};

export function setup() {
  return { tokens: setupTokens() };
}

export function chatScenario(data) {
  initTokens(data);
  const vu = __VU - 1;
  const token = getToken(vu);
  if (!token) return;

  const q = getRandomQuestion();
  const result = chatSync(token, q);
  chatLatency.add(result.latency);
  if (result.status === 429) rateLimited.add(1);
  if (result.status !== 200 && result.status !== 429) failedRequests.add(1);

  sleep(Math.random() * 5 + 2);
}

export function listDocsScenario(data) {
  initTokens(data);
  const token = getToken(0);
  if (!token) return;

  const result = listDocuments(token);
  if (result.status === 429) rateLimited.add(1);
  if (result.status !== 200 && result.status !== 429) failedRequests.add(1);

  sleep(1);
}
