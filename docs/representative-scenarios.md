# Representative Scenarios

These scenarios are durable examples of the behavior self-genesis is intended to study.
They are not fixed expected outputs and should not be treated as hard-coded policies for the agents.

## 1. 他者からのみLifeを回復できる

Agent AのLifeが減少している。
A自身がPointを持っていても、自分のLifeには使用できない。

AがAgent BとEncounterし、BがAへGIVEした場合のみ、AのLifeが増える。

## 2. 過去に会った他者との再Encounter

Agent Aが、以前会ったAgent Bと再びEncounterする。

Bの明示的なIDは入力されず、固定Appearanceと過去の内部状態・記憶を手掛かりに、
以前の相互作用を現在の判断へ利用できる余地がある。

## 3. GIVEは直接Rewardされない

Agent AがBへGIVEしても、「協力した」こと自体には追加Rewardを与えない。

その行動が将来のAの生存に有利なら、その結果を通じて学習される。

## 4. Communicationの意味は事前定義しない

Agent AとBは制約された離散token列を交換できる。

人間はtokenに意味を割り当てない。
学習後に特定tokenやtoken列が特定の状況・行動と対応していても、
それは相互作用から形成されたものとして扱う。

## 5. Working Memoryと感性は次の判断へ戻る

Encounterや観測によって更新されたWorking Memoryと感性latentは、
その場の出力だけで破棄せず、その後の思考・判断へ入力される。

同じ外部Observationでも、過去の内部状態が異なれば異なる判断が可能である。

## 6. 自己表現を正解として与えない

Agentに「これは自分」「これは他者」といった内部表現の正解ベクトルを教師として与えない。

自分の状態・自分が受けた行動・自分が行った行動と、
他者の状態・行動を区別することが生存に有利になる環境を用意し、
その結果としてどのような内部表現が形成されるかを観察する。
