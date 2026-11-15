import execution from 'k6/execution';
import { chat, chatScenarios, chatThresholds, loginUsers, positiveInt } from './lib/capacity.js';

const users = JSON.parse(open(__ENV.USERS_FILE || '../data/test_users.json'));
const replicas = positiveInt('API_REPLICAS', 1);
const chatVus = replicas * positiveInt('CHAT_VUS_PER_REPLICA', 20);

export const options = {
  setupTimeout: '120s',
  tags: { api_replicas: String(replicas) },
  // Resolve the Compose service to all replicas and rotate connections between them.
  dns: { ttl: '2s', select: 'roundRobin', policy: 'onlyIPv4' },
  noConnectionReuse: true,
  scenarios: chatScenarios(chatVus, 'replicaChat', __ENV.DURATION || '5m'),
  thresholds: chatThresholds,
};

export function setup() {
  return loginUsers(users, chatVus);
}

export function replicaChat(data) {
  chat(data.tokens[Number(__ENV.CHAT_USER_INDEX)],
    `${data.runId}-${execution.scenario.name}-${execution.scenario.iterationInTest}`);
}
