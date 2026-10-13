const AGREEMENT_URL = 'https://github.com/geometric-intelligence/topobench/blob/7258fccaa938ed35cbf84f8a88f705996afde511/CLA.md';
const STATEMENT = `I have read and accept the [TopoBench Contributor Agreement 1.0](${AGREEMENT_URL}) for my contributions to this pull request. I confirm that I control the rights granted or am authorised to grant them.`;
const STATUS = 'Contributor agreement';

function hasAcceptance(body) {
  let fence = null;
  let htmlCode = null;
  const statements = [];
  // Examples and HTML comments are not a contributor acknowledgement.
  for (const line of (body || '').replace(/<!--[\s\S]*?(?:-->|$)/g, '').split(/\r?\n/)) {
    if (!fence) {
      const wasHtmlCode = Boolean(htmlCode);
      const tags = [...line.matchAll(/<(\/?)(pre|code|script|style|textarea)(?=[\s>]|$)/gi)];
      for (const tag of tags) {
        if (!htmlCode && !tag[1]) htmlCode = tag[2].toLowerCase();
        else if (tag[1] && htmlCode === tag[2].toLowerCase()) htmlCode = null;
      }
      if (wasHtmlCode || tags.length) continue;
    }
    const boundary = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (boundary) {
      if (!fence) fence = boundary[1];
      else if (boundary[1][0] === fence[0] && boundary[1].length >= fence.length && !boundary[2].trim()) fence = null;
      continue;
    }
    if (fence) continue;
    const checkbox = line.trimEnd().match(/^- \[([ xX])\] (.+)$/);
    if (checkbox && checkbox[2] === STATEMENT) statements.push(checkbox[1].toLowerCase() === 'x');
  }
  return statements.length === 1 && statements[0];
}

module.exports = async function checkContributorAgreement({ github, context, core }) {
  const repository = context.repo;
  const knownHeads = new Set();
  const pendingHeads = new Set();
  const unavailable = 'Unable to check the contributor acknowledgement; a maintainer must rerun the check.';
  const status = async (sha, state, description) => {
    knownHeads.add(sha);
    if (state !== 'pending') {
      try {
        const [previous] = await github.paginate(github.rest.repos.listCommitStatusesForRef,
          { ...repository, ref: sha, per_page: 100 }, ({ data }, done) => {
            const latest = data.find((item) => item.context.toLowerCase() === STATUS.toLowerCase());
            if (latest) done();
            return latest ? [latest] : [];
          });
        // GitHub caps a SHA/context at 1000 statuses. Unrelated PR events must
        // not spend that quota by repeating an unchanged result.
        if (previous?.creator?.login === 'github-actions[bot]' && previous.state === state && previous.description === description) {
          if (state === 'failure' || state === 'error') core.setFailed(description);
          return state;
        }
      } catch { /* A failed deduplication read must not prevent publication. */ }
    }
    const request = {
      ...repository, sha, state, description, context: STATUS,
      target_url: `${process.env.GITHUB_SERVER_URL || 'https://github.com'}/${repository.owner}/${repository.repo}/actions/runs/${context.runId}`,
    };
    try {
      await github.rest.repos.createCommitStatus(request);
    } catch {
      core.setFailed(unavailable);
      // An exhausted or unavailable SHA must not stop all other PRs being checked.
      if (state !== 'error') {
        try { await github.rest.repos.createCommitStatus({ ...request, state: 'error', description: unavailable }); } catch { /* Keep the run failed. */ }
      }
      return 'error';
    }
    if (state === 'failure' || state === 'error') core.setFailed(description);
    return state;
  };
  const pending = async (sha) => {
    if (!pendingHeads.has(sha)) {
      await status(sha, 'pending', 'Checking the contributor agreement acknowledgement.');
      pendingHeads.add(sha);
    }
  };
  const openPulls = () => github.paginate(github.rest.pulls.list, {
    ...repository, state: 'open', base: 'main', per_page: 100,
  });
  const byHead = (prs) => {
    const groups = new Map();
    for (const pr of prs) {
      if (pr.state !== 'open' || pr.base.ref !== 'main') continue;
      knownHeads.add(pr.head.sha);
      const group = groups.get(pr.head.sha) || [];
      group.push(pr);
      groups.set(pr.head.sha, group);
    }
    return groups;
  };
  const snapshot = (prs = []) => JSON.stringify(prs
    .map(({ number, body }) => [number, body])
    .sort((a, b) => a[0] - b[0]));

  try {
    const payload = context.payload;
    const value = context.eventName === 'workflow_dispatch' ? payload.inputs?.pull_request_number
      : payload.pull_request?.number;
    if (!/^[1-9]\d*$/.test(String(value)) || !Number.isSafeInteger(Number(value))) throw new Error('Invalid PR number');
    const pull_number = Number(value);
    const eventSha = context.eventName === 'pull_request_target' && payload.pull_request?.head?.sha;
    if (/^[a-fA-F0-9]{40}$/.test(eventSha || '')) await pending(eventSha);
    const { data: pr } = await github.rest.pulls.get({ ...repository, pull_number });
    knownHeads.add(pr.head.sha);
    if (pr.base.ref === 'main' && pr.state === 'open') await pending(pr.head.sha);

    // Commit statuses belong to a SHA, not a PR. Reconcile the whole repository:
    // GitHub may replace queued concurrency jobs for unrelated PR events.
    const groups = byHead(await openPulls());
    const results = new Map();
    for (const [sha, prs] of groups) {
      for (const candidate of prs) {
        if (!hasAcceptance(candidate.body)) {
          results.set(sha, `PR #${candidate.number}: Check the exact contributor agreement acknowledgement in the PR description.`);
          break;
        }
      }
    }

    const current = byHead(await openPulls());
    for (const sha of knownHeads) {
      if (snapshot(groups.get(sha)) !== snapshot(current.get(sha))) {
        await status(sha, 'failure', 'The PRs sharing this commit changed during verification; rerun the check.');
      } else if (!current.has(sha)) {
        await status(sha, 'failure', 'No open pull request to main is associated with this commit.');
      } else if (results.has(sha)) {
        await status(sha, 'failure', results.get(sha));
      } else {
        // This checks presence only; it does not establish identity or authority.
        await status(sha, 'success', 'The exact contributor acknowledgement is present in every open PR to main on this commit.');
      }
    }
  } catch {
    for (const sha of knownHeads) {
      await status(sha, 'error', unavailable);
    }
    core.setFailed(unavailable);
  }
};
