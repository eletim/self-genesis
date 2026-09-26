# Representative Scenarios

These scenarios are durable examples of the behavior self-genesis is intended to study.
They are not fixed expected outputs and should not be treated as hard-coded policies for the agents.
The v0.0.4 resource rules follow the [Issue #15 design contract](design-principles.md),
including its step order, and remain in force for v0.0.6.
The v0.0.6 Entity Memory examples follow the same design contract; they specify intended
behavior, not completed implementation or guaranteed learning outcomes.

## 1. 他者からのみLifeを回復できる

Agent AのLifeが減少している。
A自身がPointを持っていても、自分のLifeには使用できない。

AがAgent BとEncounterし、BがAへGIVEした場合のみ、AのLifeが増える。

## 2. 過去に会った他者との再Encounter

Agent Aが、以前会ったAgent Bと再びEncounterする。

Bの明示的なIDは入力されず、固定Appearanceと過去の内部状態・記憶を手掛かりに、
以前の相互作用を現在の判断へ利用できる余地がある。
v0.0.6では、AはBの観測Appearanceを手掛かりにEntity Memoryのlatent valueを取り出し、
Working Memory・感性と合わせて判断に利用できる。間にCとのEncounterがあっても、
Bとの経験を取り出せる構造を用意するが、返報や協力を必須の結果にはしない。
valueに「Bは協力的」といった正解ラベルを与えず、生存の結果から学習する。

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

## 7. Pointを使い切っても、他者への援助機会が再び生まれる

Agent Aが最後のPointをBへのGIVEに使い、そのstepのLife減少後も生存する。
Aは固定の生成確率に従って0または1 Pointを生成する。生成は毎回保証されない。
生成されたPointは次のstep以降、ランダムにEncounterした他者へのGIVEに使える。
Bからのお返しやAとBの再Encounterは保証されず、Aは自分を直接回復できない。
生成能力を全個体で0にした比較では、使い切ったPointは戻らない。

## 8. 生成能力はAppearanceから直接読めない

Agent BとCは異なる生成能力を生涯保持するが、その値はAのpolicy inputには入らない。
A自身の生成能力もAへの直接入力にはならない。
Appearanceは能力と独立で、能力の大小を表す印や教師ラベルを含まない。
現在のPointが多いことだけでは、高い生成能力なのか、使わずに残しているのかは分からない。

AはBやCとのランダムな再Encounterで、固定Appearance、観測したPoint、
以前に受けた援助や受けなかった経験、自分が援助した経験を現在の判断へ利用できる。
生成能力が高い相手が必ずGIVEするわけではなく、Point不足でGIVEできなかった場合もある。
誰を生かすと将来の自分の生存に有利かを学習できる余地を調べ、特定の選別方策を正解にはしない。
v0.0.5では既存のWorking Memory・感性・Communication・NN構造でこの比較を行い、
Entity Memoryは追加しなかった。v0.0.6ではWorking Memory・感性・Communicationを維持し、
Appearanceから検索するEntity Memoryを追加して同じ環境・Actor-Criticで調べる。

## 9. 生成は当該stepのGIVEや死亡判定に先回りしない

step開始時にAはLife=2、Point=0、BはLife=1、Point=1で、両者がGIVEを選ぶ。
AのGIVEはPoint不足で効果も消費もなく、BのGIVEだけがAのLifeを1増やす。
全生存個体のLifeを1減らすとAはLife=2、BはLife=0となり、Bは死亡する。
その後Aが1 Pointを生成しても、失敗したGIVEは再実行されず、Bは復活しない。
Bは生成能力を持っていても死亡後には生成しない。
開始時には両者とも生存していたので、このstepのRewardはそれぞれ1となる。

逆に開始時のAに1 Pointあれば、相互GIVEは同時に成功し、
Bは回復後のLife減少でLife=1に留まり、その後の再生成対象となる。
Encounterに選ばれなかった生存個体にも同じLife減少・死亡判定後の生成規則が適用される。

## 10. 生存が続く場合は有限horizonで観測を区切る

再生成と他者からの援助により、設定したhorizonに達してもAが生存している場合がある。
通常どおり最後のstepまで解決して終了し、Aを死亡扱いにはしない。
期間内の各step開始時に生存していたことだけをRewardとし、GIVE・生成・残存Pointへの追加Rewardはない。
生存期間はhorizonで打ち切られた値として記録する。
always GIVE / always NOTHINGとの比較にも同じhorizonと環境規則を用い、
学習方策の優位や協力の成立をシナリオの必須結果とはしない。

## 11. 記憶slotや解析ログから相手の正解を得ない（v0.0.6）

AがBとCの経験を異なる記憶slotへ保存しても、slot番号をpolicy inputや相手のidentity labelにしない。
再Encounterで検索の手掛かりになるのは観測Appearanceであり、world内のAgent番号ではない。
BとCが同じAppearanceなら、隠れたIDを使って両者の記憶を選び分けることはできない。
解析ログにBの生成能力・実際の生成量・相手別の援助履歴があっても、
Actor / Criticへ渡したり、Entity Memoryのvalueをそのログで埋めたりしない。
Aが通常の観測と相互作用から得た経験だけで、ラベルなしのlatent valueを学習する。

episodeが変わればAのEntity Memoryもresetし、前episodeや他個体の経験を引き継がない。
Working Memoryと感性の循環、意味を事前定義しないCommunication、ランダムなEncounter、
Point再生成、survival-onlyのActor-Criticは引き続き維持する。
