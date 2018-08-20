package fasttext

trait MatrixBase {

  def addRow(i: Int, hidden: Array[Float], norm: Float): Unit

  def dot(y: Array[Float]): Array[Float]

}

object Matrix {
  private val ByteBytes = java.lang.Byte.BYTES
  private val LongBytes = java.lang.Long.BYTES
  private val IntBytes = java.lang.Integer.BYTES

  def load(in: LittleEndianDataInputStream): MatrixBase = {
    if (in.getByte == 0) {
      val m = in.getLong.toInt
      val n = in.getLong.toInt
      val rows = Array.fill(m)(Array.fill(n)(in.getFloat))
      new Matrix(rows)
    } else {
      val qnorm = in.getBoolean
      val m = in.getLong.toInt
      val n = in.getLong.toInt
      val codesize = in.getInt
      val codes = Array.fill[Byte](codesize)(in.readByte())
      val pq = ProductQuantizer.load(in)
      if (qnorm) {
        ???
      }
      new QMatrix(pq, null, codes, null, qnorm, m, n, codesize)
    }
  }
}

class Matrix(rows: Array[Array[Float]]) extends MatrixBase {

  def addRow(i: Int, hidden: Array[Float], norm: Float): Unit = {
    val row = rows(i)
    var j = 0
    while (j < hidden.length) {
      hidden(j) += row(j) / norm
      j += 1
    }
  }

  override def dot(y: Array[Float]): Array[Float] = {
    rows map { row =>
      var acc = 0.0f
      var j = 0
      while (j < row.length) {
        acc += row(j) * y(j)
        j += 1
      }
      acc
    }
  }
}

class QMatrix(
  pq: ProductQuantizer, npq: ProductQuantizer,
  codes: Array[Byte], normCodes: Array[Byte],
  qnorm: Boolean, m: Int, n: Int, codesize: Int) extends MatrixBase {

  override def addRow(i: Int, hidden: Array[Float], norm: Float): Unit = {
    if (qnorm) ???
    pq.addcode(hidden, codes, i, 1 / norm)
  }

  override def dot(y: Array[Float]): Array[Float] = ???
}