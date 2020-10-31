# ドキュメント
(注)体裁は適当です。。。<br>

##  データに関して
使っているデータは[Amazon Dataset](https://jmcauley.ucsd.edu/data/amazon/)のBeautyとLuxuryのデータセット。Luxuryに関しては5-core。（k-coreに関しては上のリンクに説明あり）。Beautyに関しては2-core、3-coreあたりを使っている.<br>


* データセット処理のためのコード
  データセットは`.json`で配布されている
  * `dataload.ipynb`<br>
  `.json`ファイルを`pandas.DataFrame`に変換したりしてる

  * `data_resize.ipynb`<br>
  k-coreデータセットにするためのノートブック


* データセット格納ディレクトリ
pandas.dataFrameを.csvで保存したディレクトリ
  * `All_Beauty/`
  * `All_Beauty_2core/`
  * `All_Beauty_3core/`
  

* データセットをID化し、train-valid-testに分割するコード / ディレクトリ
  * `/preprocess.py`<br>
  train-valid-testに分割するコード。
  valid1とvalid２があるのは、ハイパラチューニングのときに2-corss validationするため.
  * `/preprocess_es.py`<br>
  train-valid-testに分割するコード。early stopping用のvalidデータがある。
  * `data_xxxxxx_kcore/`<br>
    train-valid-testに分割されたデータセット


## 手法に関して
https://amitness.com/toolbox/ とかの既製のライブラリを使った方がいい気がする...

* NFM<br>
    Neural Factorization Machine
    * `data/`<br>
    使ってない
    * `result_luxury` ` result_beauty`<br>
    それぞれのデータセットを実行した時の結果
    * `data_loader.py` <br>
    `../data_xxxxx_kcore`の
    データをロードしてバッチを作ったりするための`AmazonDataset`クラスを定義している

    * `model.py`<br>
    `NFM`を定義

    * `training.py`<br>
    バッチをロードしてn epoch回すための`TrainIterater`クラス定義

    * `evaluate.py`<br>
    valid or testデータで評価するためのmetricsとクラスを定義。

    * `earlystop.py`<br>
    earlystopingのためのクラス定義.

    * `run.py`<br>
    trainingをするスクリプト. Optunaでハイパラチューニングしている

    * `test.py`<br>
    testをするスクリプト
    
* BPR_test
  Bayesian Personalized Ranking

* PPR 
  Bayesian Personalized Ranking

* KG_Embed
    knowledge graph embedding系

* PROPOSED
  提案手法
