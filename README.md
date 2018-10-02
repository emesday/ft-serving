# ft-serving

a minimum Scala implementation for serving the fastText models

# Dependency

ft-serving is available on Maven Central.

```
libraryDependencies += "com.github.mskimm" %% "ft-serving" % "0.0.1"
```

# Quick Start

```
 $ sbt console
 scala> val model = fasttext.FastText.load("data/cooking.ftz")
 scala> model.predict("are egg whites generally available at the store ?", 1)
```
 
# Supports

 - supervised model (incl. quantized with `-qnorm=false -qout=false`) with softmax loss
