# ft-serving

a minimum Scala implementation for serving the fastText models

# Quick Start

 1. train a supervised model using [fastText](https://github.com/facebookresearch/fastText)
 3. predict `sbt "runMain fasttext.app.Predict /path/to/fasttext-superviced-model"`
 
# Supports

 1. supervised model / softmax loss
   - quantization is also supported
     - Currently, only quantized with `-qnorm=false -qout=false` is supported.
