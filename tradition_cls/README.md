1. 运行 需要将 /home/ubuntu/zachary/Text-Classification-Pytorch/model 放在当前目录下

2. 支持  # elmo/glove/elmo_word2vec/elmo_glove/word2vec/all 这几种向量拼接方式

3. 运行 main.py

4. 参数 在 config.py 进行修改

5. 目前跑通了 RNN.py

6. 使用 https://github.com/HIT-SCIR/ELMoForManyLangs

7. 中文glove：

Text Classification is one of the basic and most important task of Natural Language Processing. In this repository, I am focussing on one such text classification task and that is Sentiment Analysis. So far I have covered following six different models in this repo.

  * RNN
  * LSTM
  * LSTM + Attention
  * Self Attention
  * CNN
  * RCNN

## Requirements
  * Python==3.6.6
  * PyTorch==0.4.0
  * torchtext==0.2.3

## Downloads and Setup
Once you clone this repo, run the main.py file to process the dataset and to train the model.
```shell
$ python main.py
```

## References
  * A Structured Self-Attentive Sentence Embedding : [Paper][1]
  * Convolutional Neural Networks for Sentence Classification : [Paper][2]
  * Recurrent Convolutional Neural Networks for Text Classification : [Paper][3]

[1]:https://arxiv.org/pdf/1703.03130.pdf
[2]:https://arxiv.org/pdf/1408.5882.pdf
[3]:https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiRxa37_PbbAhWOfSsKHW9bAtIQFggrMAA&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI15%2Fpaper%2Fdownload%2F9745%2F9552&usg=AOvVaw37k05lV8569fo_aCghlO9i

## License
MIT
