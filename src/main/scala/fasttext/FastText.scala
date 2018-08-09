package fasttext

import java.io.{BufferedInputStream, FileInputStream, InputStream}
import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class Line(labels: Array[Int], words: Array[Long])

case class Entry(wid: Int, count: Long, tpe: Byte, subwords: Array[Long])

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

  def computeSubwords(word: String, args: FastTextArgs): Array[Long] =
    getSubwords(word, args.minn, args.maxn).map { w => args.nwords + (hash(w) % args.bucket.toLong) }

  def readVocab(is: InputStream, args: FastTextArgs): (Map[String, Entry], Array[String]) = {
    val vocab = new mutable.HashMap[String, Entry]
    val labels = new ArrayBuffer[String]()

    val bb = ByteBuffer.allocate(9).order(ByteOrder.LITTLE_ENDIAN)
    val wb = new ArrayBuffer[Byte]

    for (wid <- 0 until args.size) {
      bb.clear()
      wb.clear()
      var b = is.read()
      while (b != 0) {
        wb += b.toByte
        b = is.read()
      }
      val word = new String(wb.toArray, "UTF-8")

      is.read(bb.array(), 0, 9)
      val count = bb.getLong
      val tpe = bb.get
      val subwords = if (word != EOS && tpe == 0) Array(wid.toLong) ++ computeSubwords(BOW + word + EOW, args) else Array(wid.toLong)
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

  def readVectors(is: BufferedInputStream, args: FastTextArgs): Array[Array[Float]] = {
    require(is.read() == 0, "not implemented")
    val bb = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
    val floats = ByteBuffer.allocate(args.dim * 4).order(ByteOrder.LITTLE_ENDIAN)
    is.read(bb.array())
    val m = bb.getLong.toInt
    val n = bb.getLong.toInt
    require(n * 4 == floats.capacity())
    Array.fill(m) {
      floats.clear()
      is.read(floats.array())
      Array.fill(n)(floats.getFloat)
    }
  }

}

class FastText(name: String) extends Serializable {

  import FastText._

  val is = new BufferedInputStream(new FileInputStream(name))
  private val args: FastTextArgs = FastTextArgs.fromInputStream(is)
  private val (vocab: Map[String, Entry], labels: Array[String]) = readVocab(is, args)
  private val inputVectors: Map[Long, Array[Float]] =
    readVectors(is, args).zipWithIndex.map { case (v, i) => i.toLong -> v }.toMap
  private val outputVectors: Array[Array[Float]] = readVectors(is, args)
  is.close()

  println(args)

  require(args.magic == FASTTEXT_FILEFORMAT_MAGIC_INT32)
  require(args.version == FASTTEXT_VERSION)

  // only sup/softmax supported
  // others are the future work.
  require(args.model == MODEL_SUP)
  require(args.loss == LOSS_SOFTMAX)

  def getEntry(word: String): Entry =
    vocab.getOrElse(word, Entry(-1, 0L, 1, Array.emptyLongArray))

  def getLine(in: String): Line = {
    val tokens = tokenize(in)
    val words = new ArrayBuffer[Long]()
    val labels = new ArrayBuffer[Int]()
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
      } else if (tpe == 1 && wid > 0) {
        labels += wid - args.nwords
      }
    }
    Line(labels.toArray, words.toArray)
  }

  def computeHidden(input: Array[Long]): Array[Float] = {
    val hidden = new Array[Float](args.dim)
    for (row <- input.map(inputVectors)) {
      var i = 0
      while (i < hidden.length) {
        hidden(i) += row(i) / input.length
        i += 1
      }
    }
    hidden
  }

  def predict(line: Line, k: Int = 1): Array[(String, Float)] = {
    val hidden = computeHidden(line.words)
    val output = outputVectors.map { o =>
      o.zip(hidden).map(a => a._1 * a._2).sum
    }
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
