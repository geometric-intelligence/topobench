const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const check = require('./check-contributor-agreement.cjs');

const SHA = 'a'.repeat(40);
const OLD_SHA = 'b'.repeat(40);
const AGREEMENT_URL = 'https://github.com/geometric-intelligence/topobench/blob/7258fccaa938ed35cbf84f8a88f705996afde511/CLA.md';
const STATEMENT = `I have read and accept the [TopoBench Contributor Agreement 1.0](${AGREEMENT_URL}) for my contributions to this pull request. I confirm that I control the rights granted or am authorised to grant them.`;
const CHECKED = `- [x] ${STATEMENT}`;

function pull(number, sha = SHA, extra = {}) {
  return { number, state: 'open', body: CHECKED, head: { sha }, base: { ref: 'main' }, ...extra };
}

function fixture(options = {}) {
  const pr = pull(429, SHA, options.pr);
  const statuses = [];
  const failures = [];
  const storedStatuses = options.storedStatuses || new Map();
  let lists = 0;
  const github = {
    rest: {
      pulls: { list() {}, get: async () => {
        if (options.pullError) throw options.pullError;
        return { data: options.prs?.find(({ number }) => number === pr.number) || pr };
      } },
      repos: {
        createCommitStatus: async (status) => {
          statuses.push(status);
          if (options.statusError?.(status)) throw new Error('private status API details');
          storedStatuses.set(status.sha, { ...status, creator: { login: 'github-actions[bot]' } });
        },
        listCommitStatusesForRef() {},
      },
    },
    paginate: async (method, args, map) => {
      assert.equal(args.per_page, 100);
      if (method === github.rest.repos.listCommitStatusesForRef) {
        return map({ data: storedStatuses.has(args.ref) ? [storedStatuses.get(args.ref)] : [] }, () => {});
      }
      assert.equal(method, github.rest.pulls.list, 'No comment or private-record API should be used');
      assert.equal(args.state, 'open');
      assert.equal(args.base, 'main');
      lists += 1;
      if (options.listErrorAt === lists) throw new Error('private PR discovery details');
      if (options.prSnapshots) return options.prSnapshots[Math.min(lists - 1, options.prSnapshots.length - 1)];
      const current = lists > 1 && options.changedPr ? { ...pr, ...options.changedPr } : pr;
      return (options.prs || [current]).filter(({ state, base }) => state === 'open' && base.ref === 'main');
    },
  };
  const context = { repo: { owner: 'example', repo: 'topobench' }, eventName: 'pull_request_target', payload: { pull_request: { number: 429 } }, runId: 123, ...options.context };
  const core = { setFailed: (message) => failures.push(message), info() {} };
  return { github, context, core, statuses, failures };
}

async function state(options = {}) {
  const f = fixture(options);
  await check(f);
  return f;
}

test('one exact checked acknowledgement succeeds without a separate record or maintainer comment', async () => {
  const f = await state();
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'success']);
  assert.equal(f.statuses[1].sha, SHA);
  assert.equal(f.statuses[1].context, 'Contributor agreement');
  assert.equal(f.failures.length, 0);
});

test('the PR template has one unchecked acknowledgement linking the exact agreement revision', async () => {
  const template = readFileSync(path.join(__dirname, '../PULL_REQUEST_TEMPLATE.md'), 'utf8');
  assert.match(AGREEMENT_URL, /\/blob\/[a-f0-9]{40}\/CLA\.md$/);
  assert.equal(template.split(`- [ ] ${STATEMENT}`).length - 1, 1);
  assert.equal((await state({ pr: { body: template } })).statuses.at(-1)?.state, 'failure');
  assert.equal((await state({ pr: { body: template.replace(`- [ ] ${STATEMENT}`, CHECKED) } })).statuses.at(-1)?.state, 'success');
});

test('a checked PR cannot overwrite an unchecked PR on the same commit', async () => {
  for (const body of ['', `- [ ] ${STATEMENT}`]) {
    const f = await state({ prs: [pull(429), pull(430, SHA, { body })] });
    assert.equal(f.statuses.at(-1)?.state, 'failure');
    assert.match(f.statuses.at(-1)?.description, /PR #430/);
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
  }
});

test('workflow reconciles PR edits and head changes without requiring comment events or permission lookups', () => {
  const workflow = readFileSync(path.join(__dirname, '../workflows/contributor-agreement.yml'), 'utf8');
  for (const event of ['opened', 'reopened', 'closed', 'edited', 'synchronize']) {
    assert.match(workflow, new RegExp(`types: \\[[^\\]]*\\b${event}\\b`));
  }
  assert.doesNotMatch(workflow, /^\s+(?:branches|issue_comment|issues):/m);
  assert.match(workflow, /^    concurrency:\n      group: [\w-]+\n      cancel-in-progress: false$/m);
  assert.doesNotMatch(workflow, /^concurrency:/m);
  assert.match(workflow, /ref: \$\{\{ github\.event\.repository\.default_branch \}\}/);
  assert.match(workflow, /persist-credentials: false/);
});

test('a status publication error fails its commit without aborting other commits', async () => {
  const otherSha = 'c'.repeat(40);
  const f = await state({
    pr: { state: 'closed' },
    prs: [SHA, OLD_SHA, otherSha].map((sha, index) => pull(430 + index, sha)),
    statusError: ({ sha, state }) => sha === SHA && state === 'success',
  });
  assert.deepEqual(new Map(f.statuses.map(({ sha, state }) => [sha, state])), new Map([[SHA, 'error'], [OLD_SHA, 'success'], [otherSha, 'success']]));
  assert.doesNotMatch(JSON.stringify(f.failures), /private status/);
});

test('manual reconciliation invalidates an orphaned commit after its last PR closes', async () => {
  const f = await state({
    pr: { state: 'closed' },
    context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429' } } },
  });
  assert.equal(f.statuses.at(-1)?.sha, SHA);
  assert.equal(f.statuses.at(-1)?.state, 'failure');
  assert.match(f.statuses.at(-1)?.description, /No open pull request/);
});

test('unrelated events do not consume more status slots on an unchanged commit', async () => {
  const storedStatuses = new Map();
  const options = { storedStatuses, prs: [pull(429), pull(430, OLD_SHA)] };
  await state(options);
  const second = await state(options);
  assert.equal(second.statuses.filter(({ sha }) => sha === OLD_SHA).length, 0);
  assert.equal(storedStatuses.get(OLD_SHA)?.state, 'success');
});

test('one surviving queued event reconciles other commits and keeps their outcomes separate', async () => {
  const f = await state({
    prs: [pull(429), pull(430, OLD_SHA, { body: '' })],
  });
  assert.deepEqual(new Map(f.statuses.map(({ sha, state }) => [sha, state])), new Map([[SHA, 'success'], [OLD_SHA, 'failure']]));
});

test('closing or retargeting an unchecked PR releases its checked sibling on the same commit', async () => {
  for (const departing of [{ state: 'closed' }, { base: { ref: 'other' } }]) {
    const f = await state({
      pr: { ...departing, body: '' }, prs: [pull(430)],
      context: { payload: { pull_request: { number: 429, head: { sha: SHA } } } },
    });
    assert.equal(f.statuses.at(-1)?.state, 'success');
  }
});

test('new commits in the same PR do not require another acceptance or a verification marker', async () => {
  const f = await state({ prs: [pull(429, OLD_SHA), pull(430)] });
  assert.deepEqual(new Map(f.statuses.map(({ sha, state }) => [sha, state])), new Map([[OLD_SHA, 'success'], [SHA, 'success']]));
});

test('success requires the acknowledgement in every PR sharing a commit', async () => {
  const f = await state({ prs: [pull(429), pull(430)] });
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'success']);
});

test('new, removed, retargeted, moved, or edited group members prevent stale success', async () => {
  const original = [pull(429), pull(430)];
  for (const changed of [
    [...original, pull(431)], [original[0]],
    [original[0], pull(430, SHA, { base: { ref: 'other' } })],
    [original[0], pull(430, OLD_SHA)], [original[0], pull(430, SHA, { body: '' })],
  ]) {
    const f = await state({ prSnapshots: [original, changed] });
    assert.ok(f.statuses.some(({ sha, state }) => sha === SHA && state === 'failure'));
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
  }
});

test('discovery errors fail every known head without leaking API details', async () => {
  for (const listErrorAt of [1, 2]) {
    const f = await state({ prs: [pull(429), pull(430, OLD_SHA)], listErrorAt });
    const finalStates = new Map(f.statuses.map(({ sha, state }) => [sha, state]));
    assert.equal(finalStates.get(SHA), 'error');
    if (listErrorAt === 2) assert.equal(finalStates.get(OLD_SHA), 'error');
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
    assert.doesNotMatch(JSON.stringify(f.failures), /private PR discovery/);
  }
});

test('missing or unchecked acknowledgement fails', async () => {
  for (const body of [null, '', `- [ ] ${STATEMENT}`]) {
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure');
  }
});

test('ignores checkbox text hidden in HTML comments, fenced code or quoted/indented examples', async () => {
  for (const body of [`<!--\n${CHECKED}\n-->`, `<!--\n${CHECKED}`, `\`\`\`markdown\n${CHECKED}\n\`\`\``, `~~~\n${CHECKED}\n~~~`, `> ${CHECKED}`, `    ${CHECKED}`]) {
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure', body);
  }
  assert.equal((await state({ pr: { body: `~~~\nexample\n~~~\n${CHECKED.replace('[x]', '[X]')}` } })).statuses.at(-1)?.state, 'success');
});

test('rejects changed wording, agreement versions, destinations and duplicate acknowledgements', async () => {
  for (const body of [
    CHECKED.replace('I confirm', 'I do not confirm'), CHECKED.replace('Agreement 1.0', 'Agreement 2.0'),
    CHECKED.replace(AGREEMENT_URL, 'https://example.com/CLA.md'),
    CHECKED.replace(/\/blob\/[^/]+\//, '/blob/main/'),
    `${CHECKED}\n- [ ] ${STATEMENT}`, `${CHECKED}\n${CHECKED}`,
  ]) {
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure');
  }
});

test('raw HTML code examples cannot acknowledge the contributor agreement', async () => {
  for (const tag of ['pre', 'code', 'script', 'style', 'textarea']) {
    const body = `<${tag}>\n${CHECKED}\n</${tag}>`;
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure', tag);
    assert.equal((await state({ pr: { body: `${body}\n${CHECKED}` } })).statuses.at(-1)?.state, 'success');
  }
});

test('multiline and consecutive raw HTML opening tags remain excluded', async () => {
  for (const opening of ['<pre\nclass="example">', '<pre>example</pre><pre>']) {
    assert.equal((await state({ pr: { body: `${opening}\n${CHECKED}\n</pre>` } })).statuses.at(-1)?.state, 'failure');
  }
});

test('initial API failure invalidates the known event head rather than preserving an old success', async () => {
  const f = await state({ pullError: new Error('private API details'), context: { payload: { pull_request: { number: 429, head: { sha: SHA } } } } });
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'error']);
  assert.equal(f.statuses.at(-1).sha, SHA);
  assert.doesNotMatch(JSON.stringify(f.failures), /private API details/);
});

test('head or acknowledgement changes during checking cannot leave success', async () => {
  for (const changedPr of [{ head: { sha: OLD_SHA }, body: CHECKED }, { head: { sha: SHA }, body: '' }]) {
    assert.equal((await state({ changedPr })).statuses.at(-1)?.state, 'failure');
  }
});

test('dispatch validates the PR number as data and does not approve unrelated target branches', async () => {
  assert.equal((await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429' } } } })).statuses.at(-1)?.state, 'success');
  const invalid = await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429; echo secret' } } } });
  assert.equal(invalid.statuses.length, 0);
  assert.equal(invalid.failures.length, 1);
  assert.equal((await state({ pr: { base: { ref: 'other' } } })).statuses.at(-1)?.state, 'failure');
});
