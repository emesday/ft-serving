package fasttext

object Predict {
  def main(args: Array[String]): Unit = {
    val model = args(0)
    val k = if (args.length == 2) args(1).toInt else 1
    val fastText = new FastText(model)

    var done = false
    while (!done) {
      val line = scala.io.StdIn.readLine("input: ")
      if (line != null) {
        fastText.predict(line, k).foreach(println)
      } else {
        done = true
      }
    }

  }
}
