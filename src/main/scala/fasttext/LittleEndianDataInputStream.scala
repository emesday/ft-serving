package fasttext

import java.io.{DataInputStream, EOFException, InputStream}

class LittleEndianDataInputStream(in: InputStream) extends DataInputStream(in) {

  def getBoolean: Boolean = {
    readBoolean()
  }

  def getByte: Byte = {
    readByte()
  }

  def getInt: Int = {
    val ch4 = in.read()
    val ch3 = in.read()
    val ch2 = in.read()
    val ch1 = in.read()
    if ((ch1 | ch2 | ch3 | ch4) < 0)
      throw new EOFException()
    (ch1 << 24) + (ch2 << 16) + (ch3 << 8) + (ch4 << 0)
  }

  private val readBuffer = new Array[Byte](8)

  def getLong: Long = {
    readFully(readBuffer, 0, 8)
    (readBuffer(7).toLong << 56) +
      ((readBuffer(6).toLong & 255) << 48) +
      ((readBuffer(5).toLong & 255) << 40) +
      ((readBuffer(4).toLong & 255) << 32) +
      ((readBuffer(3).toLong & 255) << 24) +
      ((readBuffer(2) & 255) << 16) +
      ((readBuffer(1) & 255) << 8) +
      ((readBuffer(0) & 255) << 0)
  }

  def getDouble: Double = {
    java.lang.Double.longBitsToDouble(getLong)
  }

  def getFloat: Float = {
    java.lang.Float.intBitsToFloat(getInt)
  }

}
