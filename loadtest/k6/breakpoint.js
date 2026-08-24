import { Counter, Rate, Trend } from 'k6/metrics';
import { sleep } from 'k6';
import { setupTokens, initTokens, getToken, getRandomQuestion, chatSync } from './lib/helpers.js';

const rateLimited = new Counter('rate_limited_429');
const chatLatency = new Trend('chat_sync_latency', true);
const failedRequests = new Rate('failed_requests');

export const options = {
  setupTimeout: '120s',
  scenarios: {
    breakpoint: {
      executor: 'ramping-arrival-rate',
      exec: 'chatScenario',
      startRate: 1,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 1000,
      stages: [
        { target: 5, duration: '2m' },
        { target: 10, duration: '2m' },
        { target: 20, duration: '2m' },
        { target: 40, duration: '2m' },
        { target: 60, duration: '2m' },
        { target: 80, duration: '2m' },
        { target: 100, duration: '2m' },
      ],
    },
  },
  thresholds: {
    'chat_sync_latency': ['p(95)<120000'],
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

  sleep(0.1);
}
