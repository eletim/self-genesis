# Design Principles

## 目的

self-genesis は、複数の学習Agentが他者との相互作用を通じて生存する環境で、
自己・他者・感性・コミュニケーションに関する内部表現がどのように形成されるかを調べる研究プロジェクトである。

最初から「自己」を表す正解や専用ラベルを与えるのではなく、
自己と他者を区別して扱うことが生存上有利になる環境とNN構造を用意し、
内部表現として何が形成されるかを観察する。

## v0.0.4 の環境契約（Issue #15）

本書と[代表シナリオ](representative-scenarios.md)は、
[Issue #15](https://github.com/eletim/self-genesis/issues/15)で導入する環境の契約を定める。
再生成・有限horizon・解析ログを含むこの環境契約はv0.0.5 / v0.0.6 / v0.0.7 / v0.0.8でも維持する。
v0.0.3の初期Pointを使い切る環境と比較し、他者を区別して過去の関係を利用することが
生存上有利になり得る環境圧を調べる。援助相手の選別や協力が必ず学習されるとは仮定しない。

## v0.0.5 の学習契約（Issue #24）

[Issue #24](https://github.com/eletim/self-genesis/issues/24)では、v0.0.4のproducer oracleが
always GIVEを上回る一方、学習済みNNがalmost-always-GIVE / almost-always-NOTHINGへ
偏った結果を受け、学習則だけを変えて比較する。
Value baseline・Advantage・Entropyにより早期collapseを抑え、条件依存の行動と
held-out survivalが改善するかを検証する。改善や協力の獲得を前提にはしない。
以下は最小限のActor-Criticへ拡張するための契約であり、この文書変更自体は実装完了を意味しない。

## v0.0.6 のEntity Memory契約

v0.0.6では、固定Appearanceを手掛かりに過去の相互作用を取り出す最小限のEntity Memoryを追加する。
Working Memoryを置き換えず、感性latentの循環、Communication、ランダムなEncounter、
renewable resource worldと有限horizon、survival-onlyのActor-Criticを維持する。
変更対象は記憶構造であり、援助・返報・協力の獲得や生存性能の改善を保証しない。
以下は設計契約であり、この文書変更自体はEntity Memoryの実装完了を意味しない。

## v0.0.7 のbatching・capacity契約（Issue #43）

[Issue #43](https://github.com/eletim/self-genesis/issues/43)では、capacity不足とthroughput不足の
影響を切り分けるため、独立した複数worldのbatched実行と数百万parameter級のpolicyを導入する。
変更対象は実行方式とNN容量であり、以下は実装・性能達成の報告ではなく設計契約である。
partner-specificな行動やEntity Memory利用、生存性能の改善を保証しない。

### 独立worldのbatched実行

- world stateと個体の内部状態をTensor中心に管理し、同じ処理段階にある複数worldの
  Agent / Encounter forwardをbatch化する。まず64〜256 worldsを目標とし、
  512〜1024 worldsへの拡張を妨げない構成にする。world数は環境規則を変える設定ではない。
- 各worldは独立した乱数系列・資源・死亡判定・episode境界を持つ。
  Encounterと先手・後手は各worldの生存集合からランダムに選び、world間で相手を組ませない。
  Communicationと先手・後手の観測・行動の依存順序を維持し、未来の選択結果を入力へ漏らさない。
- 第4節のResource timingを各worldでそのまま守る。開始時の資源による観測・選択、
  GIVEの同時解決、Life減少と死亡、残存個体のPoint再生成、開始時生存個体へのRewardの順とする。
  新しいPointは次stepから利用し、非参加・単独生存stepも従来どおり進める。
- padding、死亡個体、終了済みworldをmaskし、架空のdecision・Reward・記憶更新を作らない。
  worldごとの終了とresetを分離し、他worldの未完episodeや記憶をresetしない。

### Tensor化しても保つ記憶と情報境界

Working Memoryと感性latentの循環を維持し、Encounterをまたぐ個体固有の記憶を保持する。
Encounter-local Working Memoryへ置き換えない。Entity Memoryは
world / observer / entity-slot / entity-dimを扱えるTensor表現にしてよいが、
検索keyは観測Appearanceとし、valueは経験から学習されるラベルなしのlatentとする。
同じAppearanceを隠れたIDで区別せず、第8節の検索・書き込みの情報境界を維持する。
共有するのはNN重みであり、Working Memory・感性・Entity Memoryをworld間・個体間で混同しない。
これらのstateは各episode境界でresetする。

world番号・Agent番号・slot番号は内部のroutingに使えても、Actor / Criticの入力や
Appearanceに代わるidentity keyにはしない。hidden generation ability、実際の生成量、
解析専用の相手別履歴、oracle情報も直接・記憶経由のいずれでも入力へ追加しない。
Communication token、自己・他者、trust、reciprocityなどに教師ラベルを与えない。

### 容量と数値精度

Small / Medium / Largeなど複数容量を比較可能にし、層数・次元と実parameter数を記録する。
初期目安はObservation embedding 256、Thought hidden 512、Working Memory 512、
感性128、Entity Memory value 256とし、必要に応じてMLPを拡張してよい。
これは固定architectureの指定ではなく、数百万parameter級を含む比較の出発点である。
容量を増やしても通常の観測、Communication、recurrentな記憶と感性、Appearance-keyed
Entity Memory、scalar Value headという役割と情報境界は変えない。
BF16 / autocastは数値安定性を確認して利用し、return集計・Advantage・Value lossなどは
必要に応じてFP32を維持する。精度・容量・world数・seedを実験条件に記録する。

### 完全episodeのActor-Critic

第11節のsurvival-only Actor-Criticを維持する。各個体の非割引survival return、
detachしたAdvantage、Value回帰、action / message entropyを使い、PPOのclipping・
旧policy比率による更新、social reward、補助的な教師信号は導入しない。
batch内の各episodeは同じ更新前のpolicyでstep zeroから目的の終了まで収集する。
episodeごとのlossを開始時Agent数Nで正規化し、更新対象の完全episodeについて平均する。
world数やepisode長によりdecision数で再正規化した別の目的関数へ変えない。

死亡または明示的なsurvival horizonまでの完全な経験で更新し、Value bootstrapは使わない。
renewable worldでは有限horizonを必須とし、horizonで生存中の個体は死亡ではなく打ち切りとする。
collection budgetやbatch境界をterminalとして扱わず、未完episodeのstateとgraphを保持する。
同じ更新に必要なepisodeが完了するまで重みを更新せず、loss処理前のdetachや
truncated BPTTでWorking Memory・感性・Entity Memoryを通るrecurrent勾配を切らない。

### 正しさと観察可能性

小さな決定的条件で乱数の選択結果を揃え、逐次referenceとbatched実行の資源更新、
死亡 / horizon、Encounter routing、GIVE / NOTHING、Communication、Reward集計を比較する。
world間・個体間のrecurrent state / Entity Memory分離とtraining gradient flowも検証する。
浮動小数点の比較は精度に応じた許容誤差を明示する。
trainingではloss内訳・entropy・survival・GIVE率・Value error・gradient normを軽量に記録し、
詳細trace・latent・Entity Memoryは代表worldのsamplingやevaluationで解析可能にする。
world数・容量ごとにsteps / encounters / episodes per second、wall-clock、VRAM、
GPU utilization・powerを実測し、速度と学習結果を分けて評価する。
解析ログや介入結果は引き続きReward・教師信号・policy inputへ戻さない。

## v0.0.8 のEncounter density契約（Issue #58）

[Issue #58](https://github.com/eletim/self-genesis/issues/58)では、1 world step内の
Encounter数だけを変更し、反復Encounterが関係記憶の学習圧を強めるかを調べる。
以下は設計契約であり、この文書変更自体は複数Encounterの実装完了を意味しない。
密度による生存時間の増加だけでは、相手選別やEntity Memory利用の獲得を示さない。

### 密度設定とランダムmatching

- step開始時の各worldの生存個体数をLとし、可能なpair数の上限をM = floor(L / 2)とする。
  設定はpair数のcountか、生存個体の参加割合fractionのどちらか一方とする。
  両方の明示指定はエラーとし、未指定ならcount = 1（従来の1 pair / step）とする。
- countは0以上の整数とし、実際のpair数K = min(count, M)とする。
  fractionは有限の0以上1以下の数とし、K = floor(fraction × L / 2)とする。
  小数部分は切り捨て、最低1 pairへの切り上げはしない。fractionは初期個体数ではなく、
  毎stepの生存個体数に適用する。負数、非整数count、範囲外・非有限fractionはエラーとする。
- 生存集合を一様ランダムに並べ替え、先頭2K体を隣接する2体ずつのpairにするなど、
  重複なしのランダムmatchingを行う。各pairの先手・後手もランダムとする。
  個体番号、Appearance、生成能力、行動履歴による相手選択や役割固定は行わない。
  各個体の参加は1 stepに最大1 Encounterで、worldをまたぐpairは作らない。
- 奇数の生存集合では少なくとも1体が非参加となる。Lが0または1ならK = 0とする。
  count = 0やfractionの切り捨てでK = 0になる場合も、未終了worldの時間は1 step進む。
  次stepでは新しい生存集合から再抽選し、同じ相手との再Encounterを禁止も保証もしない。
  seedと密度設定から各実行方式で再現できるようにする。

### 複数pairでもworldの時計は1つ

全pairが同じstep開始時のLife / Pointを観測する。各pair内では既存の
message → reply → first action → second actionを維持し、後手のみが選択時に
先手の行動を観測する。他pairのmessage・行動・結果は入力へ追加しない。
全pairの行動が揃ってから、第4節のGIVEをworld全体で同時に解決する。
pairごとにLife減少やPoint生成を挟まず、pairの処理順序で資源結果を変えない。

各未終了world stepにつき、Life減少・死亡判定、残存個体のPoint生成抽選、
開始時生存個体への1のsurvival Reward、step / horizon時計の進行はそれぞれ1回だけとする。
非参加個体にも同じ規則を適用し、Encounter数倍の生成・Rewardや時間経過を作らない。
非参加個体に架空のCommunication・decision・Encounter完了の記憶書き込みを作らない。
終了済みworldは従来どおり停止し、資源・Reward・時計を進めない。

### 変更しない学習・情報境界

v0.0.7のbatched GPU-native実行、Small / Medium / Large容量、NN構造、
Actor-Critic係数と完全episodeのsurvival-only目的関数を維持する。
decision数が増えてもlossの開始時Agent数による正規化とepisode平均を変更せず、
PPO、social reward、補助教師信号、Value bootstrap、truncated BPTTを導入しない。
renewable Point、hidden generation ability、有限horizonと死亡の区別も維持する。

Working Memory・感性はEncounterをまたぐ個体固有の循環状態とし、Encounter-local化しない。
Appearance-keyed Entity Memoryの検索・完了書き込みは第8節のparticipant-visibleな
経験だけを使い、他pairや他worldの経験を混ぜない。episode境界のresetを維持する。
world / Agent / slot番号、hidden trait、生成量、解析履歴、oracle情報をActor / Criticへ
直接にも記憶経由にも渡さず、Communicationやidentityの意味を教師として与えない。

逐次referenceとbatched実行でmatching・役割・選択結果・生成抽選を揃え、
重複参加がないこと、同時GIVE、死亡・horizon、Rewardと記憶の情報境界を比較する。
既定の1 pair、複数pair、奇数人口、人口減少、0 pair・単独生存を検証対象とする。

## 1. シンプルさを優先する

最初の実験では、必要最小限の世界・Agent・学習系だけを実装する。

高度な社会制度、性格、所属集団、繁殖、自然言語などを先回りして追加しない。
研究上必要になった要素を、観察結果に基づいて段階的に追加する。

## 2. 複数Agentを同じ世界で学習させる

1体のAgentだけを孤立して学習させる構成にはしない。

複数Agentが存在し、他者を観測し、他者へ働きかけ、
その結果が将来の自分の生存へ返ってくる環境を基本とする。

各Agentの内部状態と経験は独立して保持する。
NNの重みをAgent間で共有することは許容するが、個体ごとのWorking Memory等を混同しない。

## 3. 生存を基本Rewardとする

Rewardの中心は「どれだけ長く生きたか」とする。

GIVE、会話、協力、返報などの特定行動に、
望ましい社会行動としての追加Rewardを直接与えない。

社会的な振る舞いに価値があるなら、
それが生存へ寄与することを通して学習される構造にする。

## 4. LifeとPoint

各AgentはLifeとPointを持つ。

- Lifeは時間経過で減少する。
- Lifeが0になると死亡する。
- Pointは自分自身には使用できない。
- Pointを他者に使用すると、その相手のLifeが増える。
- Pointは初期値を使い切るだけでなく、生存中に再生成される希少な資源とする。
- 各Agentは出生時に決まり、生涯固定されるPoint生成能力を持つ。
  episodeの初期化時に設定とseedに基づいて個体差を再現可能にサンプリングする。
- 生成能力は1 world stepあたり1 Pointを生成する確率（0以上1以下）とする。
  生存個体ごとに各stepで抽選し、生成量は0または1 Pointとする。
  生成確率の分布と初期Pointは設定可能とし、毎回のGIVEを無条件に賄える供給を標準としない。
  全個体の生成確率を0にする比較設定では、初期Pointだけの環境になる。
- 生成したPointも自分のLifeには変換できない。GIVEはPointの譲渡ではなく、
  送り手の1 Pointを消費して生存中の相手のLifeを1増やす。

したがって、自分の生命を自分だけで直接延長することはできず、
他者との関係が生存に本質的な意味を持つ。

### Resource timing

1 world stepの順序は次のとおりとする。

1. step開始時の生存集合とLife / Pointを確定する。
   その集合から密度設定に従う重複なしのランダムなpair群と先手・後手を選び、
   各Encounterで観測・Communication・行動選択を行う（既定は最大1 pair）。
   資源値はこの間更新しない。後手が先手の選択行動を観測する既存手順は維持する。
2. 全EncounterのGIVEを同時に解決する。開始時に送り手と受け手が生存し、送り手に1 Point以上あれば、
   1 Pointを消費して相手のLifeを1回復する。Point不足のGIVEは効果も消費もない。
   Encounterに選ばれなかった個体はNOTHINGとなる。
3. 開始時に生存していた全個体のLifeを1減らし、0になった個体の死亡を確定する。
   GIVEによる回復はこの減少より先なので、Lifeが1の受け手をそのstepの死亡から救える。
4. この減少後も生存している全個体についてPointを再生成する。
   Encounter参加の有無に依存せず、死亡個体は生成せず復活もしない。
   新しいPointが観測・使用できるのは次のstepからで、そのstepのPoint不足を遡って補わない。
5. 開始時に生存していた個体へ1のsurvival Rewardを与える（死亡した最後のstepも含む）。
   全員死亡で終了する。再生成により全員死亡が保証されないため、明示的な有限horizonでも
   episodeを終了できる契約とする。horizon到達は死亡と区別し、生存中の個体の寿命は打ち切りとして扱う。
   horizon自体や残ったPointへの追加Rewardは与えない。

未終了worldでは、密度設定や生存個体が2体未満のためEncounterがなくても、
Life減少・死亡判定・再生成・survival Rewardは1 stepにつき1回進む。
各個体のstep後Pointは「step開始時Point − 成功したGIVEの消費 + 実際の生成量」となる。

## 5. 他者には固定Appearanceを持たせる

各Agentは出生時に決まり、生涯固定されるN次元Appearanceを持つ。

他者を観測したとき、そのAppearanceが入力される。
人間が明示的な個体IDや「このAgentは誰か」という意味ラベルを与えることを基本としない。

同じ他者との過去の関係を利用する必要があるなら、
Agent自身がAppearanceと経験を結び付ける。

生成能力はAppearanceと独立にサンプリングし、Appearanceへ符号化しない。
自分・相手の生成能力そのものや個体IDをpolicy inputへ追加しない。
現在の自分・相手のLife / Pointという既存の観測は維持するが、
現在のPoint残高は生成能力のラベルではない。
能力や過去の行動を推測するには、固定AppearanceとEncounterで得た資源状態・行動の経験を利用する。

## 6. Encounterを基本的な相互作用単位とする

Agent同士のEncounterは環境によってランダムに発生する。

Encounterには先手・後手があり、これもランダムに決まる。
Encounter中にAgentは相手を観測し、Communicationを行い、
最終的に相手へPointを使う GIVE または何もしない NOTHING を選択できる。

具体的な手順は、研究目的を壊さない範囲で単純に保つ。

## 7. Communicationの意味を人間が決めない

Agent間には小さな離散Communicationチャネルを持たせる。

使用可能なtoken数やmessage長には制約を設けるが、
各tokenの意味は教師として与えない。

Communicationが有用なら、
その意味や使い方がAgent間の学習から形成されることを狙う。

初期段階では自然言語を直接使用しない。

## 8. Working Memoryを明示的に持つ

NNには、時間をまたいで更新されるWorking Memoryを持たせる。

現在のObservationだけで次の行動を決める反射的な構造ではなく、
過去の観測・相互作用・思考を現在の判断へ利用できる構造とする。

Working Memoryの具体的な次元数や実装方式は固定せず、実験対象とする。

### Entity Memory（v0.0.6）

Entity Memoryは、個体ごとの経験をAppearanceに結び付けて保持する補助的な記憶とする。
現在の相手の観測Appearanceを検索の手掛かりにし、取り出したlatent valueを
Working Memory・感性とともに次の判断と記憶更新へ利用する。
記憶するvalueは、観測と相互作用からsurvival-onlyのActor-Criticで学習するラベルなしの内部表現であり、
「協力的」「信用できる」「高い生成能力」などの正解値や、手書きの相手評価を与えない。
個体ごとのEntity Memoryを混同せず、episode境界でresetする。

検索を明示的な個体IDや自己・他者のidentity labelで行わない。
同じAppearanceを持つ相手を、隠れたIDで区別して検索してはならない。
実装上の記憶slot番号やworld内のAgent番号を相手の識別子として使わず、
これらの番号・identity labelをActor / Criticのpolicy inputへ渡さない。
生成能力、実際の生成量、解析専用の相手別履歴やoracle情報などの特権情報も、
直接入力にも記憶の検索・書き込みを経由する入力にも使わない。
利用できる経験は、その個体が通常のObservation・Communication・行動を通して得たものに限る。
検索・保存方式や容量の詳細はここでは固定せず、この情報境界を守る最小構成を実装対象とする。

## 9. 感性を循環する内部状態として扱う

NN内部には「感性」に相当するlatent stateを持たせる。

感性は単なる最終出力ではなく、
現在の入力やWorking Memoryから生成され、
その結果が次の思考・Working Memory更新の入力へ戻る循環構造とする。

Observation、Working Memory、感性、思考は一方向のpipelineではなく、
相互に影響しながら更新されることを重視する。

感性latentの各次元に、人間が事前に「恐怖」「好意」などの意味を割り当てる必要はない。

## 10. Selfを直接教え込まない

「自分」を表す専用の正解ラベルや固定的な自己記述を学習目標として与えない。

自分のLife、自分のPoint、自分が行った行動、自分が受けた行動、
他者の状態や行動を区別する必要がある環境を通して、
自己と他者に対応する内部表現が形成されるかを観察する。

自己表現の実装方法を最初から決め打ちするのではなく、
必要であればNN形状を変えながら検証する。

## 11. NN形状と学習環境を主要な研究対象とする

このプロジェクトで主に変更・比較するものは次の2つである。

1. NNの形状
2. 学習環境

特にNNでは、Working Memoryと感性の循環構造を核にしつつ、
具体的な層、次元、attention、再帰回数などは実験によって比較する。

学習アルゴリズムも固定的な思想として扱わず、
この環境でend-to-endに学習できる妥当な方法を選ぶ。

### v0.0.5での変更範囲（歴史的な制約）

v0.0.4のREINFORCEを比較元とし、v0.0.5では各個体のsurvival-only reward-to-goを
維持した最小限のActor-Criticへ拡張する。以下のValue head以外は、既存のNN形状・次元、
Observation、Working Memory、感性latentの循環、Communication構造を維持する。
renewable resource worldの規則、Encounterの順序、固定Appearanceも変更しない。
Working Memoryのentity / associative memory化、NNの大型化や別architecture化、
world ruleの再設計、自己・他者のidentity label追加、supervised auxiliary taskは行わない。
PPOのclippingや旧policyとの比率を用いる更新など、大規模な学習方式変更も対象外とする。

上記のNN形状維持とentity / associative memory化の禁止は、学習則だけを比較するv0.0.5の制約である。
v0.0.6では第8節のEntity Memory追加を許容するが、identity labelやsupervised auxiliary taskは
引き続き導入しない。以下のActor-Criticのreturn・loss・収集境界の契約は維持し、
Actor / Criticは通常の観測と許可された個体固有の記憶表現のみを利用する。

### Survival returnとValue baseline

各Agent iのworld step tに対して、G_i,tを、そのstepから自身の死亡または設定した
survival horizonまでのsurvival Rewardの非割引和（gamma = 1）とする。
死亡する最後のstep、Encounterに選ばれないstep、単独生存中のstepのRewardも含める。
他Agentのreturnを混ぜたり、集団のreturnで置き換えたりしない。

既存の共有recurrent networkにscalar Value headを追加する。
各message / actionのサンプリング直前に、そのcallbackのObservationと個体固有の
Working Memory・感性から得た既存のrecurrent表現h_i,dを使い、
V_i,d = V(h_i,d)で、その個体のG_i,tを予測する（dはstep t内のdecision）。
同じstepのmessageとactionには同じG_i,tを用いるが、各callback時点の情報に応じて
Valueは異なってよい。Valueはそのdecisionのsample結果や未来の観測を入力に使わない。
生成能力、個体ID、解析専用ログなどの特権情報もActor / Criticへ追加しない。
共有するのは重みであり、個体間のrecurrent stateは混同しない。

### Advantage・Value・Entropyの目的関数

実際にサンプリングしたdecisionについて、次のlossを用いる。
各項はdecisionについて和を取り、既存と同じくepisode開始時のAgent数Nで割る。

- Advantage: A_i,d = G_i,t − V_i,d。
- Actor loss: L_actor = −sum(log pi_i,d(choice) * stop_gradient(A_i,d)) / N。
  Advantageはpolicy更新時にdetachし、Actor lossからValue予測へは勾配を流さない。
- Critic loss: L_value = sum((V_i,d − stop_gradient(G_i,t))^2) / N。
  Value headと共有recurrent表現を自身のsurvival returnの回帰で学習する。
- Action entropy: H_actionはGIVE / NOTHINGのcategorical分布のentropyの和 / N。
- Message entropy: H_messageは各messageのtoken分布のentropyをslot間で足し、
  message decisionについて和を取った値 / N。既存の独立token samplingでは
  messageのlog probabilityもtokenごとのlog probabilityの和とする。
- 最小化する総loss: L = L_actor + c_value * L_value
  − beta_action * H_action − beta_message * H_message。
  c_valueは正、entropy係数は非負の設定値として実験条件に記録する。
  entropy bonusを使う比較では対応する係数を正にし、ゼロによる無効化も可能とする。

Entropyは探索のための正則化であり、world Rewardやsurvival returnへ足さない。
GIVE、協力、会話、返報へのsocial / cooperation / communication rewardは導入しない。
Value回帰以外に補助的な教師信号を追加せず、tokenや感性latentに意味の正解を与えない。
message長が0ならmessageのActor・Value・Entropy項は作らず、内部状態の更新は維持する。
非参加stepの自動NOTHINGにもdecision lossを作らないが、そのstepのRewardはreturnへ含める。
離散sampleにはscore-function勾配を使い、Working Memory・感性へのrecurrent勾配を維持する。

### 有限horizonと収集境界

更新にはstep zeroから死亡・全員死亡または明示的なsurvival horizonまでの完全な経験を使う。
死亡後と目的として定めたhorizonの後にはreturnを加えず、Value bootstrapもterminal bonusも使わない。
horizon到達は死亡ではなく、生存個体の寿命は引き続き打ち切りとして記録する。
単なるcollection budgetの終了は未完のtruncationであり、完全なepisodeとして更新しない。
継続収集ではrecurrent stateとgraphを保持し、完全な目的区間を揃える。
再生成ありの学習では明示的な有限horizonを必須とし、再生成なしでは従来どおり全員死亡までの
学習も許容する。episode境界でstateをresetし、更新に使うlossの処理前にはgraphをdetachしない。

## 12. 観察可能性を保つ

性能だけでなく、何が学習されたかを後から調べられることを重視する。

少なくとも、生存時間、死亡、GIVE / NOTHING、Life / Point、
Communication、Agent間相互作用に加え、
Working Memoryや感性latentを後から解析できるようにする。

生成能力と実際のstepごとの生成量は解析専用ログに記録してよいが、policy inputへ流さない。
GIVEの試行と成功した有向の援助を区別し、Appearanceによる同一相手との再Encounter、
過去に受けたGIVE / 非GIVE、自分から行ったGIVEと後続の行動との関係を解析できるようにする。
既存メトリクスを維持し、v0.0.4と同じ資源・時間・Rewardの規則で学習済みpolicy、
always GIVE、always NOTHING、producer oracleを比較する。oracleの生成能力へのアクセスは
比較用に限定し、学習済みpolicyへは渡さない。可能なら従来REINFORCEとも環境条件・NN構造・
学習予算・seedを揃えたmatched comparisonを行う。
seedごとのGIVE率collapse、held-out survival、partner-history依存の行動、
Appearance shuffle / Working Memory resetによる性能差、Communication利用の変化を確認する。
これらの解析・介入結果を追加Rewardや教師信号にはしない。

ただし解析機能のために学習系を過度に複雑化しない。
