# fastTextServing

a minimum Scala implementation for serving the fastText model

# Quick Start

 1. train a supervised model using [fastText](https://github.com/facebookresearch/fastText)
 2. copy model to RocksDB as `sbt "runMain fasttext.CopyModel /path/to/fasttext.model.bin /path/to/output"`
 3. predict using the RockDB model `sbt "runMain fasttext.Predict /path/to/output"`
 
# Plan
 - [x] copy model to RocksDB
 - [x] predict
 - [ ] predict-prob
 - [ ] print-word-vectors
 - [ ] print-sentence-vectors
 - [ ] print-ngrams
 - [ ] nn
 - [ ] analogies
 - [ ] dump
 - [ ] REST serving layer
 
