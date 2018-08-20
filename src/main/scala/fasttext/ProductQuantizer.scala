package fasttext

object ProductQuantizer {
  private val nbits = 8
  private val ksub = 1 << nbits
  private val maxPointsPerCluster = 256
  private val maxPoints = maxPointsPerCluster * ksub
  private val nitesr = 25
  private val eps = 1e-7

  def load(in: LittleEndianDataInputStream): ProductQuantizer = {
    val dim = in.getInt
    val nsubq = in.getInt
    val dsub = in.getInt
    val lastdsub = in.getInt
    val centroids = Array.fill(dim * ksub)(in.getFloat)
    new ProductQuantizer(dim, dsub, nsubq, lastdsub, centroids)
  }
}

class ProductQuantizer(
  dim: Int, dsub: Int, nsubq: Int, lastdsub: Int, centroids: Array[Float]) {

  import ProductQuantizer._

  def getCentroid(m: Int, i: Int): Int = {
    if (m == nsubq -1) {
      m * ksub * dsub + i * lastdsub
    } else {
      (m * ksub + i) * dsub
    }
  }

  def addcode(x: Array[Float], codes: Array[Byte], t: Int, alpha: Float): Unit = {
    var d = dsub
    var m = 0
    while (m < nsubq) {
      val c = getCentroid(m, codes(nsubq * t + m) & 0xff)
      if (m == nsubq - 1) d = lastdsub
      var n = 0
      while (n < d) {
        x(m * dsub + n) += alpha * centroids(c + n)
        n += 1
      }
      m += 1
    }
  }

}
