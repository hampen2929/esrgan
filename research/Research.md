super-resolution-gan

# paper

## SRGAN

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

https://arxiv.org/abs/1609.04802

![image](https://user-images.githubusercontent.com/34574033/77926617-48095280-72e1-11ea-9314-73e8cd6edebf.png)

画像超解像（SR）のための生成的逆境ネットワーク（GAN）であるSRGANを提案

4倍のアップスケーリングファクターでフォトリアリスティックな自然画像を推論

知覚損失関数を提案

逆説的損失は、超解像画像と元のフォトリアリスティック画像を区別するために訓練された識別ネットワークを用いて、我々の解を自然画像

認知的類似性を動機とするコンテンツロスを利用する 画素空間における類似度の 我々の深層残差ネットワーク は、重度の 公開されているベンチマーク上の画像をダウンサンプリング

広範な
平均意見スコア(MOS)テストでは、大きな有意性を示します。SRGANを用いた知覚の質の向上

## ESRGAN

- SRGANからの改善点


1. アーキテクチャ (Generator)

- バッチノーマリゼーションを除去

学習データとテストデータで性質(statistics)が違うと好ましくないartifactsが出てきて、生成能力が制限される。
BNレイヤを取り除けば生成能力も上がるし、計算も減ってメモリにとってもありがたい

1. Residual in Residual Dense Block (RRDB)
レイヤの出力を次のレイヤだけでなく別のレイヤにも加える
SRGANにあるresidualブロックよりも深層で複雑。メインの流れにはDense blockを用いている
=> Dense blockによってモデルの容量が大きくなる。

- techniques

residualスケーリング
main pathに値を加算する前に0~1の定数をかける
=> 不安定さを防ぐ

smaller initialization
パラメータの初期値の分散が小さい方が、residual構造の学習が楽
  

![image](https://user-images.githubusercontent.com/34574033/77927764-8e12e600-72e2-11ea-9d16-56ce2be17e67.png)

https://qiita.com/yuji_tech/items/43274a5cb2b794fd90a9

- 低解像度の料理画像を超解像するための SRGAN の応用

https://confit.atlas.jp/guide/event-img/jsai2018/3A1-03/public/pdf?type=in

https://arxiv.org/abs/1809.00219



https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x



## 参考
- SRGAN: Training Dataset Matters
https://arxiv.org/pdf/1903.09922.pdf


https://gigazine.net/news/20200205-deep-learning-4k-oldist-movie/?fbclid=IwAR2cPhnJCiLn8NZHEvhLTGYllptvvFHpEvQ9CZR8BoSGlGfUa_3rkb3k1WE

https://ai-scholar.tech/articles/treatise/tecogan-154?fbclid=IwAR00GmFVHrhi9D87PZ8ZA6-IGkvPXbTkvhsNtslD8rDEwTwOKQETM-RMOAM

https://qiita.com/koshian2/items/aefbe4b26a7a235b5a5e


？？
https://github.com/NVlabs/few-shot-vid2vid

xxx



