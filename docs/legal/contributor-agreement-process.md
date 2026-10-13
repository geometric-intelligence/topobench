# Contributor agreement: acknowledgement and merge check

The [TopoBench Contributor Agreement 1.0](https://github.com/geometric-intelligence/topobench/blob/7258fccaa938ed35cbf84f8a88f705996afde511/CLA.md)
is the agreement. This page explains the PR workflow and its limitations.

## Contributor experience

Contributors read the complete agreement linked from the PR template and
personally select its single acceptance checkbox. The agreement identifies the
recipients and the commercial grant. No separate acceptance email, form,
signature service or maintainer verification marker is required.

The acceptance covers the person's authorised contributions to that PR,
including subsequent updates they intentionally submit under the agreement.
Pushing another commit does not require another acceptance while the same
acknowledgement remains selected. Each separate PR requires its own acceptance.
Other rights holders' material requires their permission or an acceptance by
someone authorised to grant those rights; one submitter cannot automatically
bind every coauthor.

## Agreement version and acceptance record

The checkbox links to a specific commit containing the complete agreement.
Retain that commit and the PR with its acknowledgement, available history and
contribution commits. The PR serves as the acceptance record; this workflow
does not create a separate acceptance database or collect private forms.

Maintainers must not select the checkbox for contributors without authority.
A passing check only confirms the acknowledgement's presence. It does not prove
who selected it, authenticate a legal identity, determine ownership or establish
an employer's or university's authorisation. Resolve disclosed rights issues
through ordinary contribution review before relying on the grant.

GitHub permits removal of revision contents and limits edit history. An
independent export or backup can help preserve evidence, but is not an additional
contributor step or a condition of this CLA. Retain the PR and referenced
agreement for as long as the project relies on that grant. A Git clone alone
does not include PR descriptions or their edit history.

When changing agreement terms or recipients, issue a new version, retain the
accepted revision, and update the template, check and documentation together.
The new version requires fresh express acceptance; a changed file on `main`
does not alter an earlier accepted agreement. Editing or removing a checkbox
does not itself undo rights already validly granted.

## What the automated check does

`Contributor agreement` passes when exactly one visible, checked acknowledgement
matches the required wording and pinned agreement link in the PR description.
Missing, unchecked, altered or duplicate acknowledgements fail. Text hidden in
comments or code examples is not an acknowledgement.

GitHub commit statuses belong to a SHA, so every open PR targeting `main` with
that head commit must satisfy the check before the shared commit passes. A
failure identifies the blocking PR. Every run reconciles all open PRs to `main`;
closing, retargeting or updating a PR refreshes those groups. Runs are serialized
and membership and descriptions are checked again before publishing results.
Unchanged results for unrelated commits are not republished.

The workflow runs trusted default-branch code and never executes PR code with
its status token. It reads PRs and statuses and publishes a status; it does not
send messages or inspect acceptance records, identities, comments or collaborator
permissions. Manual dispatch accepts a PR number for rechecking.

Checks are asynchronous. An edit after the final read is handled by the next
run. API or status-publication failures can leave an older result visible;
inspect workflow errors and rerun before relying on it. One reconciliation run
can report failure for one commit and success for another; inspect the current
PR head's status and the run's errors.

## Enabling protection after deployment

1. Confirm the reviewed workflow and checker are on `main`, and that repository
   Actions policy permits the triggers and token permissions. A workflow file
   alone does not make its status a merge requirement.
2. Dispatch `contributor-agreement.yml` on `main` with an open PR number. Confirm
   a fresh `Contributor agreement` failure for a missing acknowledgement on the
   expected head commit. Do not check acceptance on someone else's behalf merely
   to test a successful result.
3. Re-read branch protection. Add `Contributor agreement` as a required check
   bound to the GitHub Actions app, resolving its ID through
   `GET /apps/github-actions`. Preserve existing required checks and the current
   strict/up-to-date setting.
4. Use `PATCH /repos/{owner}/{repo}/branches/main/protection/required_status_checks`
   with an authorised administrator's credentials. Read back protection and
   confirm review requirements, restrictions and bypass settings are preserved.
   Existing administrator or role bypasses remain unless separately changed.
5. Inspect the required-check display on an unacknowledged PR without attempting
   a test merge. With an actual contributor's acceptance, verify success and that
   a subsequent commit in the same PR passes without another acceptance. The
   unit tests cover checkbox removal and other negative cases without creating
   real acceptance.

Until protection is enabled, maintainers must check for acknowledgement before
merging. Earlier contributions retain their existing rights basis; the workflow
does not manufacture consent or fill missing commercial grants.

References: [GitHub permanent file links](https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files),
[GitHub edit history](https://docs.github.com/en/communities/moderating-comments-and-conversations/tracking-changes-in-a-comment),
[status-check protection API](https://docs.github.com/en/rest/branches/branch-protection#update-status-check-protection).
