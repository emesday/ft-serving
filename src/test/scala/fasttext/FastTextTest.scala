package fasttext

import org.scalatest.Matchers._
import org.scalatest._

class FastTextTest extends FunSuite {

  test("predict") {
    val ft = FastText.load("data/cooking.bin", verbose = true)
    val input = scala.io.Source.fromFile("data/cooking.valid").getLines()
    val pred = scala.io.Source.fromFile("data/cooking.valid.pred").getLines()
    for ((l, p) <- input.zip(pred))  {
      val actual = ft.predict(l, 4)
      val expected = p.split(" ").grouped(2).map { case Array(label, score) => (label, score.toFloat) }.toArray
      actual.zip(expected) foreach { case ((al, as), (el, es)) =>
        al shouldBe el
        as shouldBe es +- 1e-4f
      }
    }
  }

  test("predict - quant") {
    val ft = FastText.load("data/cooking.ftz")
    val input = scala.io.Source.fromFile("data/cooking.valid").getLines()
    val pred = scala.io.Source.fromFile("data/cooking.valid.quant.pred").getLines()
    for ((l, p) <- input.zip(pred))  {
      val actual = ft.predict(l, 4)
      val expected = p.split(" ").grouped(2).map { case Array(label, score) => (label, score.toFloat) }.toArray
      actual.zip(expected) foreach { case ((al, as), (el, es)) =>
        al shouldBe el
        as shouldBe es +- 1e-4f
      }
    }
  }

}
