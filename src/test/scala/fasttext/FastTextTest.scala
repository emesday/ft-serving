package fasttext

import org.scalatest.Matchers._
import org.scalatest._

class FastTextTest extends FunSuite {

  val ft = new FastText("data/cooking.bin")

  test("predict") {
    val input = scala.io.Source.fromFile("data/cooking.valid").getLines()
    val pred = scala.io.Source.fromFile("data/cooking.valid.pred").getLines()
    for ((l, p) <- input.zip(pred))  {
      val actual = ft.predict(l, 4)
      val expected = p.split(" ").grouped(2).map { case Array(label, score) => (label, score.toFloat) }.toArray
      println(actual.toSeq)
      println(expected.toSeq)
      actual.zip(expected) foreach { case ((al, as), (el, es)) =>
        al shouldBe el
        as shouldBe es +- 1e-4f
      }
    }
  }

}
