package fasttext.app

import java.io.{BufferedReader, FileInputStream, InputStreamReader}

import fasttext.FastText

object Predict {
  def main(args: Array[String]): Unit = {
    val model = args(0)
    val input = args(1)
    val k = if (args.length == 3) args(2).toInt else 1
    val fastText = FastText.load(model)

    val is = if (input == "-") {
      System.in
    } else {
      new FileInputStream(input)
    }

    val br = new BufferedReader(new InputStreamReader(is))

    var done = false
    while (!done) {
      val line = br.readLine()
      if (line != null) {
        val res = fastText.predict(line, k).map(x => s"${x._1} ${x._2}").mkString(" ")
        println(res)
      } else {
        done = true
      }
    }

    br.close()
  }
}
