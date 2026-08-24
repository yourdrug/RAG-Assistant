import { Counter, Rate, Trend } from 'k6/metrics';
import { group, sleep } from 'k6';
import { setupTokens, initTokens, getToken, getRandomQuestion, chatSync, listDocuments } from './lib/helpers.js';

const rateLimited = new Counter('rate_limited_429');
const chatLatency = new Trend('chat_sync_latency', true);
const failedRequests = new Rate('failed_requests');

export const options = {
  setupTimeout: '120s',
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: 3,
      duration: '2m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.10'],
    chat_sync_latency: ['p(95)<120000'],
  },
};

export function setup() {
  return { tokens: setupTokens() };
}

export default function (data) {
  initTokens(data);
  const vu = __VU - 1;
  const token = getToken(vu);
  if (!token) return;

  group('Smoke: chat sync', () => {
    const q = getRandomQuestion();
    const result = chatSync(token, q);
    chatLatency.add(result.latency);
    if (result.status === 429) rateLimited.add(1);
    if (result.status !== 200 && result.status !== 429) failedRequests.add(1);
  });

  sleep(2);

  group('Smoke: list documents', () => {
    const result = listDocuments(token);
    if (result.status !== 200) failedRequests.add(1);
  });

  sleep(1);
}
