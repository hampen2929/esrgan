super-resolution-gan

# 変遷
## 深層学習ベース

- SRCNN (最初のアプローチ)
- PNSR-oriented approaches (人間的にはよく見えない)
- (SRResNet?)
- SRGAN (Perceptual-driven methods)
- ESRGAN (the proposed method)
[3]

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

### SRGANからの改善点

#### アーキテクチャ (Generator)

##### バッチノーマリゼーションを除去

学習データとテストデータで性質(statistics)が違うと好ましくないartifactsが出てきて、生成能力が制限される。
BNレイヤを取り除けば生成能力も上がるし、計算も減ってメモリにとってもありがたい[2]

BatchNormalizationは入力の平均・分散を求めますがTrain時/Test時のデータの違いによって、この統計的処理が異なりアーティファクトが生じることがあります。従って、このアーティファクトを防ぐためSRGANのResidual BlockからBatch Normalizationを除去しました。[3]


##### Residual in Residual Dense Block (RRDB)
レイヤの出力を次のレイヤだけでなく別のレイヤにも加える
SRGANにあるresidualブロックよりも深層で複雑。メインの流れにはDense blockを用いている
=> Dense blockによってモデルの容量が大きくなる。

SRGANでは中間層としてResidual Blockを用いていましたがESRGANでは下図のようなResidual in Residual Dense Block（RRDB)を用いています。DenseNetでも用いられるDense Blockを弱い残差で結合しており、これによってより広いコンテキストも拾えるようになるとしています。[2]

##### Relativistic average Discriminator

ある程度学習が進みDiscriminatorの判別力が上昇すると、Discriminatorの出力はFake画像をRealと判別した時のみ誤差が生じるようになります。これではReal画像で学習が出来ません。そこで、Real画像を入力に用いた時のDiscriminatorの出力とFake画像を入力に用いた時の出力（正確にはその平均）の差分を見て学習を進めます。これにより、Real画像Fake画像どちらを用いても学習が可能になります。[2]

##### Loss Function
VGGの各層の出力で誤差を取るPerceptual lossにおいて、SRGANでは活性化関数であるReLUの後を出力としていました。しかし以下の画像のように、ReLUの後では画素情報が失われてしまいます。そこでESRGANではReLUの前を出力としてPerceptual lossを求めました。また、SRGANの損失関数であるAdversarial loss＋Perceptual lossに加えてL1 lossも加えています。[2]


### Loss




- techniques

residualスケーリング
main pathに値を加算する前に0~1の定数をかける
=> 不安定さを防ぐ

smaller initialization
パラメータの初期値の分散が小さい方が、residual構造の学習が楽
  

![image](https://user-images.githubusercontent.com/34574033/77927764-8e12e600-72e2-11ea-9d16-56ce2be17e67.png)


## 参考

### Paper
- SRGAN: Training Dataset Matters[1]

https://arxiv.org/abs/1903.09922

- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

https://arxiv.org/abs/1809.00219

- ESRGAN+ : Further Improving Enhanced Super-Resolution Generative Adversarial Network

https://arxiv.org/abs/2001.08073

- Residual Dense Network for Image Super-Resolution, CVPR 18

http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf

- SRCNN

- 

- The relativistic discriminator: a key element missing from standard GAN

https://arxiv.org/abs/1807.00734

- Real-World Super-Resolution via Kernel Estimation and Noise Injection

https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf

https://github.com/jixiaozhong/RealSR


### 記事
- 続・GANによる二次元美少女画像生成[2]
https://medium.com/@crosssceneofwindff/%E7%B6%9A-gan%E3%81%AB%E3%82%88%E3%82%8B%E4%BA%8C%E6%AC%A1%E5%85%83%E7%BE%8E%E5%B0%91%E5%A5%B3%E7%94%BB%E5%83%8F%E7%94%9F%E6%88%90-a32fbe808eb0

- ESRGAN論文まとめ[3]
https://qiita.com/yuji_tech/items/43274a5cb2b794fd90a9

- 【Intern CV Report】超解像の歴史探訪 -SRGAN編-[4]
https://buildersbox.corp-sansan.com/entry/2019/04/29/110000

- SRGAN: Training Dataset Matters [5]
https://arxiv.org/pdf/1903.09922.pdf


- RelativisticGANの論文を読んでPytorchで実装した　その1

http://owatank.hatenablog.com/entry/2018/09/24/000938

- [2]
https://gigazine.net/news/20200205-deep-learning-4k-oldist-movie/?fbclid=IwAR2cPhnJCiLn8NZHEvhLTGYllptvvFHpEvQ9CZR8BoSGlGfUa_3rkb3k1WE




https://ai-scholar.tech/articles/treatise/tecogan-154?fbclid=IwAR00GmFVHrhi9D87PZ8ZA6-IGkvPXbTkvhsNtslD8rDEwTwOKQETM-RMOAM


https://qiita.com/koshian2/items/aefbe4b26a7a235b5a5e


？？
https://github.com/NVlabs/few-shot-vid2vid




https://qiita.com/yuji_tech/items/43274a5cb2b794fd90a9

- 低解像度の料理画像を超解像するための SRGAN の応用

https://confit.atlas.jp/guide/event-img/jsai2018/3A1-03/public/pdf?type=in

https://arxiv.org/abs/1809.00219



https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x


