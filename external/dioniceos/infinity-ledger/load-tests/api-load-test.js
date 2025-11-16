// Load testing script for MEF API using k6
// Run with: k6 run --vus 10 --duration 30s load-tests/api-load-test.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchLatency = new Trend('search_latency');
const upsertLatency = new Trend('upsert_latency');

// Configuration
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8080';

// Test options
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Ramp up to 10 users
    { duration: '1m', target: 10 },    // Stay at 10 users
    { duration: '30s', target: 50 },   // Ramp up to 50 users
    { duration: '2m', target: 50 },    // Stay at 50 users
    { duration: '30s', target: 100 },  // Spike to 100 users
    { duration: '1m', target: 100 },   // Stay at 100 users
    { duration: '30s', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95% of requests should be below 500ms
    'errors': ['rate<0.1'],                            // Error rate should be below 10%
    'http_req_failed': ['rate<0.1'],                   // Failed requests should be below 10%
  },
};

// Generate random vector
function generateRandomVector(dim = 768) {
  const vector = [];
  for (let i = 0; i < dim; i++) {
    vector.push(Math.random());
  }
  return vector;
}

// Generate random ID
function generateId() {
  return `vec_${Date.now()}_${Math.random().toString(36).substring(7)}`;
}

// Test health endpoint
export function testHealth() {
  const res = http.get(`${BASE_URL}/healthz`);
  
  check(res, {
    'health check status is 200': (r) => r.status === 200,
    'health check has ok status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'ok';
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(res.status !== 200);
}

// Test upsert endpoint
export function testUpsert() {
  const payload = JSON.stringify({
    vectors: [
      {
        id: generateId(),
        vector: generateRandomVector(),
        metadata: {
          timestamp: new Date().toISOString(),
          type: 'load_test',
        },
      },
    ],
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const res = http.post(`${BASE_URL}/upsert`, payload, params);
  
  check(res, {
    'upsert status is 200': (r) => r.status === 200,
    'upsert has valid response': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.upserted_count > 0;
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(res.status !== 200);
  upsertLatency.add(res.timings.duration);
}

// Test search endpoint
export function testSearch() {
  const payload = JSON.stringify({
    query_vector: generateRandomVector(),
    top_k: 10,
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const res = http.post(`${BASE_URL}/search`, payload, params);
  
  check(res, {
    'search status is 200': (r) => r.status === 200,
    'search has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.results);
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(res.status !== 200);
  searchLatency.add(res.timings.duration);
}

// Test spiral snapshot endpoint
export function testSnapshot() {
  const payload = JSON.stringify({
    data: {
      type: 'load_test',
      value: Math.random(),
      timestamp: new Date().toISOString(),
    },
    seed: `LOAD_TEST_${generateId()}`,
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const res = http.post(`${BASE_URL}/spiral/snapshot`, payload, params);
  
  check(res, {
    'snapshot status is 200': (r) => r.status === 200,
    'snapshot has id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.snapshot_id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(res.status !== 200);
}

// Main scenario
export default function () {
  // Test mix: 70% search, 20% upsert, 10% other operations
  const rand = Math.random();
  
  if (rand < 0.7) {
    // 70% search operations
    testSearch();
  } else if (rand < 0.9) {
    // 20% upsert operations
    testUpsert();
  } else {
    // 10% snapshot operations
    testSnapshot();
  }
  
  // Periodically check health
  if (Math.random() < 0.1) {
    testHealth();
  }
  
  // Think time between requests
  sleep(0.1 + Math.random() * 0.5);
}

// Setup: Run before the test starts
export function setup() {
  console.log('Starting load test...');
  console.log(`Base URL: ${BASE_URL}`);
  
  // Verify API is accessible
  const res = http.get(`${BASE_URL}/healthz`);
  if (res.status !== 200) {
    throw new Error('API is not accessible');
  }
  
  return { startTime: Date.now() };
}

// Teardown: Run after the test completes
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration} seconds`);
}
