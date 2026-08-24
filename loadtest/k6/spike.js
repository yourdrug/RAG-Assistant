import { Counter, Rate, Trend } from 'k6/metrics';
import { sleep } from 'k6';
import { setupTokens, initTokens, getToken, getRandomQuestion, chatSync } from './lib/helpers.js';

const rateLimited = new Counter('rate_limited_429');
const chatLatency = new Trend('chat_sync_latency', true);
const failedRequests = new Rate('failed_requests');

export const options = {
  setupTimeout: '120s',
  scenarios: {
    spike: {
      executor: 'ramping-vus',
      exec: 'chatScenario',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '30s', target: 500 },
        { duration: '5m', target: 500 },
        { duration: '30s', target: 50 },
        { duration: '2m', target: 50 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    'chat_sync_latency': ['p(95)<120000'],
    'http_req_failed': ['rate<0.05'],
    'rate_limited_429': ['count<3000'],
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

  sleep(Math.random() * 2 + 0.5);
}
