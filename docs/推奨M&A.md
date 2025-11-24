推奨M&A v0.0.2
Recommended Buy-Side Filter for Owner SME Acquisition
(Discount-aware, leverage-aware revision)
	0.	Scope

本仕様は、後継者不足のオーナー系中小企業を対象とした「勝ち寄りM&A案件」の判定ルールを定義する。
財務・顧客構造・許認可・価格・レバレッジ条件を定量化し、買収可否・優先度ラベル・交渉ディスカウント目標を決定する。

本仕様は「買い手側の一貫した判定軸」を目的とし、案件ごとの例外判断は別レイヤーで管理する。
	1.	記号

S          : 年間売上高
GM         : 粗利率 (gross margin, %)
E          : EBITDA（ノーマライズ後）
NI         : 税引後利益（ノーマライズ後）
C          : 内部留保のうち自由に使える実質キャッシュ相当額
D          : 既存有利子負債残高
P          : 最終的な買収対価（成約価格, equity value）
P_ask      : 売り手希望価格（equity value）
P_cap      : 買い手側が許容する理論上限価格
k          : NI 何年分を支払うかを表す係数
r          : 借入金利（実効金利）
D_new      : 新規に調達する有利子負債
D_total    : 買収後の有利子負債合計 = D + D_new
Interest   : 利払費用 ≒ r × D_total
DSCR       : Debt Service Coverage Ratio ≒ E / Interest

DSCR_target_hard     : 安全側に見る DSCR 目標値（推奨 3.0）
Discount(P)          : 1 - P / P_ask
Discount_target_min  : 目標ディスカウント下限（0.10 = 10%）
Discount_target_max  : 目標ディスカウント上限（0.30 = 30%）
	2.	対象企業の前提イメージ

・オーナー系中小企業
・後継者不足による売却意向が明確
・事業としての継続可能性があり、許認可ビジネス（または同等の参入障壁）を優先
・地域制約（例: 首都圏限定）は別途フラグ管理
	3.	ハード条件 (必須フィルタ)

以下を全て満たさない場合、価格検討前に原則見送りとする。

H1. 黒字継続
過去10年以上、経常利益ベースで黒字継続していること。

H2. 粗利率
GM >= 40%

H3. 長期B2B比率
売上のうち、1年以上継続見込みの長期B2B契約売上比率 >= 60%

H4. 有利子負債
D == 0 または D <= E × 1.0
（既存有利子負債は EBITDA の1年分未満）

H5. 許認可・免許
許認可・免許が「オーナー個人資格」ではなく「法人または継続可能体制」に紐づいていること。
オーナー退任後も、有資格者数などの要件を満たせる見込みがあること。

H6. 顧客集中
任意の単一取引先の売上比率 < 40%
（上位1社が40%を超える場合は原則NG。例外扱いする場合は別レイヤーで明示管理する。）
	4.	価格条件 (バリュエーションルール)

4.1 ノーマライズ利益の定義

ノーマライズ後の税引後利益 NI を以下の方針で算定する。

・オーナーの過大役員報酬、親族給与、私的経費を加算し直す
・本来必要な設備更新コストや修繕費を控除する
・一過性利益 / 損失を除外し、平常ベースの NI を保守的に見積もる

4.2 理論上限価格 P_cap

買い手側の理論上限価格を以下で定義する。

P_cap = C + k × NI

ここで k は「何年分の利益を前払いとみなすか」を表す。
推奨レンジ:

・超勝ちレンジ用 k_super = 1.5
・勝ちレンジ用   k_win   = 2.0

定義:

P_max_super = C + 1.5 × NI
P_max_win   = C + 2.0 × NI

解釈:

・P <= P_max_super
→ 内部留保 + 1.5年分の利益以内で買えており、極めて有利。
・P_max_super < P <= P_max_win
→ 内部留保 + 1.5〜2.0年分の利益の範囲であり、十分に魅力的。
・P > P_max_win
→ 成長確信やシナジーの特殊事情がない限り、原則として高すぎる。

4.3 ディスカウントターゲット

売り手希望価格 P_ask に対して、買い手は原則として 10〜30% のディスカウントを目標とする。

Discount(P) = 1 - P / P_ask

パラメータ:

Discount_target_min = 0.10
Discount_target_max = 0.30

売り手希望価格に対するディスカウント要求率:

Discount_req_cap = 1 - P_cap / P_ask
（P_cap として通常 P_max_win を用いる）

ルール:

D1. P_ask > P_max_win かつ Discount_req_cap > Discount_target_max
→ 「30%以上値下げしないと理論上限に入らない」と判断し、原則見送り候補とする。
（例外的に案件ストーリーが強い場合のみ別管理で検討）

D2. P_ask <= P_max_win または Discount_req_cap <= Discount_target_max の場合
→ 10〜30%ディスカウントを基本方針として交渉する。

4.4 交渉提示レンジ

交渉開始時のインディカティブな提示レンジを以下で定義する。

P_offer_low  = max( C,          P_ask × (1 - Discount_target_max) )
P_offer_high = min( P_max_super, P_ask × (1 - Discount_target_min) )

・P_offer_low は「30%ディスカウント目安と内部留保下限の高い方」
・P_offer_high は「10%ディスカウント目安と P_max_super の低い方」

P_offer_mid = (P_offer_low + P_offer_high) / 2 を想定成約価格候補 P_proposed として扱ってよい。
	5.	レバレッジ条件 (借入制約)

5.1 DSCR 近似

Interest ≒ r × D_total
DSCR_approx ≒ E / Interest = E / (r × D_total)

5.2 レバレッジ上限

安全側ルール:

D_total <= min( P × 0.7,  E / (r × DSCR_target_hard) )

推奨値:

・DSCR_target_hard = 3.0
・D_total / P <= 0.7 を一つの上限目安とし、それでも DSCR_approx >= 3.0 を満たす範囲に D_new を抑える。
	6.	ディール分類ルール (価格・ディスカウント・レバレッジ統合)

6.1 手順概要

Step 1: ハード条件チェック
・H1〜H6 の全てを満たしているか確認。
・1つでもNGがあれば原則「見送り」。例外検討案件は別レイヤー管理。

Step 2: 希望価格とディスカウントの整合性

・NI, C から P_max_super, P_max_win, P_cap を算出（通常 P_cap = P_max_win）。
・P_ask と Discount_req_cap を算出。

2-1. 希望価格が高すぎるケース:
・P_ask > P_max_win かつ Discount_req_cap > Discount_target_max
→ 「目標30%ディスカウントでも理論上限に届かない」ため、原則見送り候補。

2-2. 許容範囲内のケース:
・上記以外 → 交渉継続可とし、P_offer_low, P_offer_high を算出。

Step 3: 想定成約価格 P_proposed の設定

・P_proposed を P_offer_mid など合理的な点に仮置きする。
・案件によっては P_offer_high 側や中間点を採用してシナリオ別に評価してもよい。

Step 4: 価格ラベル

P_proposed に対して以下を適用する。

・P_proposed <= P_max_super
→ 価格ラベル: 「超勝ち価格候補」

・P_max_super < P_proposed <= P_max_win
→ 価格ラベル: 「勝ち価格候補」

・P_proposed > P_max_win
→ 価格NG（ディスカウント不足または理論上限超過）

Step 5: レバレッジ安全性チェック

・P = P_proposed として D_new を仮置きし、D_total, DSCR_approx を計算する。
・DSCR_approx < DSCR_target_hard となる場合は D_new を削減し再計算する。
・D_new を削ってもディールが成立しない場合は「資金構成上NG」として見送り。

Step 6: 最終ラベル

以下の全てを満たす案件のみ、クロージング検討フェーズへ進める。

・H1〜H6 全てOK
・価格ラベルが「超勝ち価格候補」または「勝ち価格候補」
・DSCR_approx >= DSCR_target_hard
・Discount(P_proposed) が概ね 10〜30% のレンジ内
（レンジ外の場合は、なぜそれを許容したかを案件メモに明示する）
	7.	DD時の必須確認ポイント (NI の再評価トリガ)

上記の数式条件を満たしていても、以下に大きな問題があれば NI を下方修正し、再評価する。

D1. ノーマライズの妥当性
・役員報酬、親族給与、私的経費の混入状況
・一過性費用 / 利益の扱い
・本来必要な CAPEX・修繕費が十分に織り込まれているか

D2. 設備・更新負担
・設備の老朽化状況と近未来の更新投資負担
・減価償却を削って見かけの利益を水増ししていないか

D3. 訴訟・保証・オフバランス
・潜在的な訴訟、長期保証、瑕疵担保、環境リスク等
・オフバランスの重要な債務・保証の有無

D4. 人材構造
・キーマンの年齢分布、退職リスク
・オーナー以外のナンバー2・実務責任者の存在とリテンションの設計余地

D5. 許認可の更新条件
・更新サイクルと要件（有資格者数、常勤要件など）
・買収後に要件割れするリスクの有無

重大なマイナスが判明した場合の処理:
	1.	NI を保守的に再計算する。
	2.	P_max_super, P_max_win, P_cap, Discount_req_cap を再計算する。
	3.	価格ラベルおよびディスカウント達成可能性を再評価する。
	4.	必要に応じて「勝ち → 見送り」などラベルを更新する。
	5.	運用フロー (v0.0.2)
	6.	案件情報の一次収集
・過去10年分の PL/BS
・顧客上位リスト、売上構成
・許認可一覧
・従業員年齢構成、キーマン情報
・売り手希望価格 P_ask
	7.	ハード条件 H1〜H6 チェック
・NGがあれば原則見送り。例外検討は別リスト管理。
	8.	NI, C のノーマライズ算定
	9.	P_max_super, P_max_win, P_cap, Discount_req_cap の計算
	10.	希望価格 P_ask と Discount_req_cap に基づき、
・30%以上のディスカウントがないと P_cap に入らない場合は見送り候補とフラグ。
・それ以外は P_offer_low, P_offer_high, P_proposed を算出。
	11.	P_proposed に基づき価格ラベル（超勝ち / 勝ち / 価格NG）を付与
	12.	D_new を仮置きし、D_total, DSCR_approx を算出
・DSCR_approx >= DSCR_target_hard となる範囲へ D_new を調整
・調整不可能なら「資金構成上NG」として見送り
	13.	詳細DDで D1〜D5 を検証し、必要に応じて NI・P_cap・ラベルを更新
	14.	最終判定
・ハード条件OK
・最終 P_proposed で価格ラベルが「超勝ち」または「勝ち」
・Discount(P_proposed) が概ね 10〜30%（外れる場合は理由を明示）
・DSCR_approx >= DSCR_target_hard
・DDで致命的リスクなし
上記を全て満たした案件のみ、最終交渉・クロージング検討へ進める。
	15.	フィードバックとパラメータ更新
実案件の蓄積に応じて、以下を中心に v0.0.3 以降で調整する。
・k のレンジ（1.5 / 2.0 の妥当性）
・DSCR_target_hard の値
・H6 顧客集中閾値（40%ライン）の妥当性
・Discount_target_min / Discount_target_max の実効性

以上を「推奨M&A v0.0.2」とする。
