# Prediction log pattern

## Usecase
- 推論結果や所要時間、ログをもとにサービスを改善したいとき
- ログの発生量が多く、ログを集約するDWHへの負荷に懸念があるとき
- ログを用いてアラートを送りたいとき

## Architecture
MLシステムを組み込んだサービスを改善するためには推論結果や推論速度、クライアントへの影響、その後のイベントを収集し、分析する必要があります。各種ログはMLシステムの各コンポーネントで発生しますが、分析のためには統一したDWHに集約するほうが効率的です。DWHへはキューを通してログを追加することで、DWHへのアクセス負荷を軽減し、データロストを防ぐことができます。ただし、キューの処理時間が長引くと、DWHログの鮮度は低下します。<br>
ログは分析以外にも有用です。たとえばクライアントログや推論ログの結果が想定と違う（または従来から大きく変わった）場合、ワークフローとして異常が起きている可能性があります。そうした場合にはログからアラートを受け取り、異常を分析、解消する必要があります。クライアントの仕様変更により不意に入力データが変わってしまう、ということもあります。その場合に推論が失敗してシステム障害になれば気付くことが容易ですが、推論は通るけど結果が異常、ということもあり得ます。そうした場合に備えて、推論ログやクライアントログの異常状態を定義し、アラート対象にしておくことが重要です。


## Diagram
![diagram](diagram.png)


## Pros
- 推論によるクライアントやユーザ、連携するシステムへの影響分析を実行することが可能。
- 必要に応じてアラート可能。

## Cons
- ログの量によってコストが増加する。

## Needs consideration
- ログの収集頻度やログレベル
- DWHへ格納する頻度や期間
- 分析の目的

## Sample
https://github.com/shibuiwilliam/ml-system-in-actions/tree/main/chapter5_operations/prediction_log_pattern