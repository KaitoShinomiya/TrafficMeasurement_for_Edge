# Single Shot MultiBox Detector Implementation in Pytorchを使ったエッジデバイス最適化実験

## 実験条件
特に言及のないパラメータはSSD論文準拠です。

> - 学習率 : 1e-3
>   - cosine schedulerで減衰
> - batch size : 32
> - num epochs : 50

- base
  特に追加条件なし。
- condition 1  
  - 量子化
- condition 2
  - 25% 枝刈り
- condition 3
  - 量子化
  - 25% 枝刈り
- condition 4
  - 50% 枝刈り
- condition 5
  - 量子化
  - 50% 枝刈り
- condition 6
  - 75% 枝刈り
- condition 7
  - 量子化
  - 75% 枝刈り

## 結果：[Google Drive](https://drive.google.com/drive/folders/1k2y94lxhRNAV4x-muFMJgEmZpYmMehAx?usp=sharing)

## SSDの基本的な使い方は[こちら](https://github.com/Jeong-Labo/pytorch-ssd)

## Based on, LICENCE
>Copyright (c) 2019 mashyko:　https://github.com/mashyko/pytorch-ssd  
>MIT LICENCE: `./LICENCE`
