package fasttext

object Predict {
  def main(args: Array[String]): Unit = {
    val fastText = new FastText(args(0))

    var done = false
    while (!done) {
      val line0 = scala.io.StdIn.readLine("input: ")
      if (line0 != null) {
        val line = fastText.getLine(line0)
        fastText.predict(line, 1).foreach(println)
      } else {
        done = true
      }
    }

    fastText.close()
  }
}
