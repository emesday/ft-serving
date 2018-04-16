package fasttext

import org.scalatest.Matchers._
import org.scalatest._

class OperationTest extends FunSuite {
  import FastText._

  val oovLines = "__label__0 생축\n" + // short OOV
    "__label__0 생축입니당행복하세요" // long OOV

  val oovExpected = Array(
    Array("__label__0", "생축", "</s>"),
    Array("__label__0", "생축입니당행복하세요", "</s>")
  )

  val oovSubwordExpected = Array(
    Array("<생", "<생축", "생", "생축", "생축>", "축", "축>"),
    Array("<생", "<생축", "생", "생축", "생축입", "축", "축입", "축입니", "입", "입니", "입니당", "니", "니당",
      "니당행", "당", "당행", "당행복", "행", "행복", "행복하", "복", "복하", "복하세", "하", "하세", "하세요",
      "세", "세요", "세요>", "요", "요>")
  )

  val expectedHashResult = Array(0L)

  val lines = "__label__1 열심히 평창 동계 올림픽을 응원합니다\n" +
    "__label__1 평창 동계올림픽 함께 응원합니다\n" +
    "__label__1 좋은밤 기쁨가득 하시고 평창 동계 올림픽 응원도 잘하시고 저녁 맛난것도 든든히 잘드시고 건강하세요\n" +
    "__label__0 생일축하한당" + "\n" + oovLines

  val expected = Array(
    Array("__label__1", "열심히", "평창", "동계", "올림픽을", "응원합니다", "</s>"),
    Array("__label__1", "평창", "동계올림픽", "함께", "응원합니다", "</s>"),
    Array("__label__1", "좋은밤", "기쁨가득", "하시고", "평창", "동계", "올림픽", "응원도", "잘하시고",
      "저녁", "맛난것도", "든든히", "잘드시고", "건강하세요", "</s>"),
    Array("__label__0", "생일축하한당", "</s>")
  ) ++ oovExpected

  test("tokenize") {
    val actual = lines.split("\n") map FastText.tokenize
    actual.zip(expected) foreach { case (a, e) =>
      a should be (e)
    }
  }

  test("getSubwords") {
    val parsed = oovLines.split("\n") flatMap FastText.tokenize
    val words = parsed.filterNot(_.startsWith("__label__")).filterNot(_ == "</s>")
    val actual = words map { w => FastText.getSubwords(BOW + w + EOW, 0, 3) }
    actual zip oovSubwordExpected foreach { case (a, e) =>
      a.sorted should be (e.sorted)
    }
  }

  test("hash") {
    val log = "걸릴까요 1284665252\n이제서야 2899849607\n멋진글 1079896981\n겨워서 932092734\n행복에 2698479257\n" +
      "보고 597925910\n첫금메달 3008949556"
    val actual = log.split("\n").map { l =>
      val Array(word, hash) = l.split(" ")
      (word, hash.toLong)
    }
    val expected = actual.map(_._1).map(FastText.hash)
    actual.map(_._2) should be (expected)
  }

}
