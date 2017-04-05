---
layout: single
title: "Git-Branching Tutorial Notes"
tags: [tutorial,basics,git]
category: notes
excerpt: "useful git commands to do what you need to do"
---
I tried this awesome tutorial [`Learn Git Branching`](http://learngitbranching.js.org/) and wanted to write down this cheat sheet. Git has a lot of commands and one can easily forget some of them. So lets write them down! 

- `git branch <b_name> [ref]`: creates branch at [ref]. [ref] is optional and can be commit,hash, branch name or tag. If ommited it is __HEAD__
- `git checkout <commit_hash>` : moves _HEAD_, may cause _detached HEAD_
- `git checkout -b <branch_name`: `git branch <branch_name>` + `git checkout <branch_name>`
- `git rebase <branch_name> [current_branch]`: changes parent of *current_branch* to *branch_name*. *current_branch* is __HEAD__ by deafult. While doing that applies all commits in the *current_branch* to the target_branch. 
- `git checkout <b_name>^^`: 2 commit above *b_name*. You can use these to remove uncommitted changes.
- `git checkout <b_name>~x`: x commit above *b_name*. (x being int)
- `git branch -f master HEAD~3`: moves the branch to a commit(3 commit above HEAD) by force! 
- `git reset HEAD^`: resets the current branch to previous commit. Basically removes the last commit. Use this if your commits are local and haven't shared with people.
- `git revert HEAD~4`: undos the changes and brings back(4 commit above) previous commit as a new commit. Use this if you previous commits are shared(pushed)
- `git cherry-pick [<c_hash>]+`: takes the commit/s and applies them to the current branch. You can use them to pick the commits, necessarry for the task (like only the bug fix commit)
- `git rebase -i HEAD~4`: rebase by reordering, squasing or excluding commits. You can use them to pick the commits, necessarry for the task (like only the bug fix commit)
- `git commit --amend`: updates the current commit with staged differences. 
- `git tag <t_name> <c_name>`: tags a specific commit in the history permanently. If `<c_name>` ommited tags __HEAD__. 
- `git describe <ref>`: returns how far you are from the closest tag. `<ref>`can be commit hash, branch name or tag

