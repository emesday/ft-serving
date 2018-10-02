package fasttext

import java.io.{BufferedInputStream, FileInputStream}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.reflectiveCalls

case class Args(
  magic: Int, version: Int, dim: Int, ws: Int, epoch: Int,
  minCount: Int, neg: Int, wordNgrams: Int, loss: Int, model: Int,
  bucket: Int, minn: Int, maxn: Int, lrUpdateRate: Int, t: Double,
  size: Int, nwords: Int, nlabels: Int, ntokens: Long, pruneidxSize: Long) {
  override def toString: String = {
    s"""magic:        $magic
       |version:      $version
       |dim:          $dim
       |ws :          $ws
       |epoch:        $epoch
       |minCount:     $minCount
       |neg:          $neg
       |wordNgrams:   $wordNgrams
       |loss:         $loss
       |model:        $model
       |bucket:       $bucket
       |minn:         $minn
       |maxn:         $maxn
       |lrUpdateRate: $lrUpdateRate
       |t:            $t
       |size:         $size
       |nwords:       $nwords
       |nlabels:      $nlabels
       |ntokens:      $ntokens
       |pruneIdxSize: $pruneidxSize
       |""".stripMargin
  }
}

case class Line(labels: Array[Int], words: Array[Int])

case class Entry(wid: Int, count: Long, tpe: Byte, subwords: Array[Int])

object FastText {
  val EOS = "</s>"
  val BOW = "<"
  val EOW = ">"

  val FASTTEXT_VERSION = 12 // Version 1b
  val FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314

  val MODEL_CBOW = 1
  val MODEL_SG = 2
  val MODEL_SUP = 3

  val LOSS_HS = 1
  val LOSS_NS = 2
  val LOSS_SOFTMAX = 3

  def tokenize(in: String): Array[String] = in.split("\\s+") ++ Array("</s>")

  def getSubwords(word: String, minn: Int, maxn: Int): Array[String] = {
    val l = math.max(minn, 1)
    val u = math.min(maxn, word.length)
    val r = l to u flatMap word.sliding
    r.filterNot(s => s == BOW || s == EOW).toArray
  }

  def hash(str: String): Long = {
    var h = 2166136261L.toInt
    for (b <- str.getBytes) {
      h = (h ^ b) * 16777619
    }
    h & 0xffffffffL
  }

  def computeSubwords(word: String, args: Args): Array[Int] =
    getSubwords(word, args.minn, args.maxn).map { w => args.nwords + (hash(w) % args.bucket.toLong).toInt }

  def readArgs(in: LittleEndianDataInputStream): Args = {
    Args(
      in.getInt, in.getInt, in.getInt, in.getInt, in.getInt, in.getInt, in.getInt, in.getInt, in.getInt, in.getInt,
      in.getInt, in.getInt, in.getInt, in.getInt, in.getDouble, in.getInt, in.getInt, in.getInt, in.getLong, in.getLong)
  }

  def readVocab(in: LittleEndianDataInputStream, args: Args): (Map[String, Entry], Array[String]) = {
    val vocab = new mutable.HashMap[String, Entry]
    val labels = new ArrayBuffer[String]()
    val wb = new ArrayBuffer[Byte]
    for (wid <- 0 until args.size) {
      wb.clear()
      var b = in.read()
      while (b != 0) {
        wb += b.toByte
        b = in.read()
      }
      val word = new String(wb.toArray, "UTF-8")

      val count = in.getLong
      val tpe = in.getByte
      val subwords = if (word != EOS && tpe == 0) Array(wid) ++ computeSubwords(BOW + word + EOW, args) else Array(wid)
      val entry = Entry(wid, count, tpe, subwords)

      vocab += word -> entry

      if (tpe == 1) {
        val label = wid - args.nwords
        require(labels.length == label)
        labels += word
      }
    }
    (vocab.toMap, labels.toArray)
  }

  def using[A, B <: { def close(): Unit }](closeable: B)(f: B => A): A = {
    try {
      f(closeable)
    } finally {
      if (closeable != null) {
        closeable.close()
      }
    }
  }

  def load(name: String, verbose: Boolean = false): FastText = {
    using(new LittleEndianDataInputStream(
      new BufferedInputStream(new FileInputStream(name)))) { in =>
      val args = readArgs(in)
      if (verbose) println(args)
      // only sup/softmax supported
      // others are the future work.
      require(args.magic == FASTTEXT_FILEFORMAT_MAGIC_INT32)
      require(args.version == FASTTEXT_VERSION)
      require(args.model == MODEL_SUP)
      require(args.loss == LOSS_SOFTMAX)
      require(args.pruneidxSize < 0)
      val (vocab, labels) = readVocab(in, args)
      val inputVectors = Matrix.load(in)
      val outputVectors = Matrix.load(in)
      new FastText(args, vocab, labels, inputVectors, outputVectors)
    }
  }

}

class FastText(args: Args, vocab: Map[String, Entry], labels: Array[String],
  inputVectors: MatrixBase, outputVectors: MatrixBase) extends Serializable {

  import FastText._

  def getEntry(word: String): Entry =
    vocab.getOrElse(word, Entry(-1, 0L, 0, Array.emptyIntArray))

  def addWordNgrams(line: ArrayBuffer[Int], hashes: Seq[Int], n: Int): Unit = {
    val mask = BigInt("18446744073709551615")
    val size = hashes.length
    var i = 0
    while (i < size) {
      var h = hashes(i) & mask
      var j = i + 1
      while (j < size && j < i + n) {
        h = ((h * 116049371) & mask) + hashes(j)
        line += (h % args.bucket).toInt + args.nwords
        j += 1
      }
      i += 1
    }
  }

  def getLine(in: String): Line = {
    val tokens = tokenize(in)
    val words = new ArrayBuffer[Int]()
    val labels = new ArrayBuffer[Int]()
    val wordHashes = new ArrayBuffer[Int]()
    tokens foreach { token =>
      val Entry(wid, count, tpe, subwords) = getEntry(token)
      if (tpe == 0) {
        // addSubwords
        if (wid < 0) { // OOV
          if (token != EOS) {
            words ++= computeSubwords(BOW + token + EOW, args)
          }
        } else {
          words ++= subwords
        }
        wordHashes += hash(token).toInt
      } else if (tpe == 1 && wid > 0) {
        labels += wid - args.nwords
      }
    }
    addWordNgrams(words, wordHashes, args.wordNgrams)
    Line(labels.toArray, words.toArray)
  }

  def computeHidden(input: Array[Int]): Array[Float] = {
    val hidden = new Array[Float](args.dim)
    for (i <- input) {
      inputVectors.addRow(i, hidden, input.length)
    }
    hidden
  }

  def predict(line: String, k: Int = 1): Array[(String, Float)] = {
    val line1 = getLine(line)
    val hidden = computeHidden(line1.words)
    val output = outputVectors.dot(hidden)
    val max = output.max
    var i = 0
    var z = 0.0f
    while (i < output.length) {
      output(i) = math.exp((output(i) - max).toDouble).toFloat
      z += output(i)
      i += 1
    }
    i = 0
    while (i < output.length) {
      output(i) /= z
      i += 1
    }
    output.zipWithIndex.sortBy(-_._1).take(k).map { case (prob, i) =>
      labels(i) -> prob
    }
  }

}
