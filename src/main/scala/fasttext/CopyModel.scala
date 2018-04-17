package fasttext

import java.io.{BufferedInputStream, FileInputStream, InputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.util

import scala.collection.JavaConverters._
import org.rocksdb._

import scala.collection.mutable.ArrayBuffer

object CopyModel {

  def writeArgs(db: RocksDB, handle: ColumnFamilyHandle, args: FastTextArgs): Unit = {
    val wo = new WriteOptions()
    db.put(handle, wo, "args".getBytes("UTF-8"), args.serialize)
    wo.close()
  }

  def writeVocab(is: InputStream, db: RocksDB, handle: ColumnFamilyHandle, args: FastTextArgs): Unit = {
    val wo = new WriteOptions()
    val bb = ByteBuffer.allocate(13).order(ByteOrder.LITTLE_ENDIAN)
    val wb = new ArrayBuffer[Byte]
    for (wid <- 0 until args.size) {
      bb.clear()
      wb.clear()
      var b = is.read()
      while (b != 0) {
        wb += b.toByte
        b = is.read()
      }
      bb.putInt(wid)
      is.read(bb.array(), 4, 9)
      db.put(handle, wo, wb.toArray, bb.array())
    }
    wo.close()
  }

  def writeVectors(is: InputStream, db: RocksDB, handle: ColumnFamilyHandle, args: FastTextArgs): Unit = {
    require(is.read() == 0, "not implemented")
    val wo = new WriteOptions()
    val bb = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
    val key = ByteBuffer.allocate(8)
    val value = new Array[Byte](args.dim * 4)
    is.read(bb.array())
    val m = bb.getLong
    val n = bb.getLong
    require(n * 4 == value.length)
    var i = 0L
    while (i < m) {
      key.clear()
      key.putLong(i)
      is.read(value)
      db.put(handle, wo, key.array(), value)
      i += 1
    }
    wo.close()
  }

  def main(args: Array[String]): Unit = {
    val in = args(0)
    val out = args(1)

    RocksDB.destroyDB(out, new Options)

    val dbOptions = new DBOptions().setCreateIfMissing(true).setCreateMissingColumnFamilies(true)
    val descriptors = new java.util.LinkedList[ColumnFamilyDescriptor]()
    descriptors.add(new ColumnFamilyDescriptor(RocksDB.DEFAULT_COLUMN_FAMILY))
    descriptors.add(new ColumnFamilyDescriptor("vocab".getBytes()))
    descriptors.add(new ColumnFamilyDescriptor("i".getBytes()))
    descriptors.add(new ColumnFamilyDescriptor("o".getBytes()))
    val handles = new util.LinkedList[ColumnFamilyHandle]()
    val db = RocksDB.open(dbOptions, out, descriptors, handles)

    val is = new BufferedInputStream(new FileInputStream(in))
    val fastTextArgs = FastTextArgs.fromInputStream(is)

    require(fastTextArgs.magic == FastText.FASTTEXT_FILEFORMAT_MAGIC_INT32)
    require(fastTextArgs.version == FastText.FASTTEXT_VERSION)

    println("step 1: writing args")
    writeArgs(db, handles.get(0), fastTextArgs)
    println("step 2: writing vocab")
    writeVocab(is, db, handles.get(1), fastTextArgs)
    println("step 3: writing input vectors")
    writeVectors(is, db, handles.get(2), fastTextArgs)
    println("step 4: writing output vectors")
    writeVectors(is, db, handles.get(3), fastTextArgs)
    println("step 5: compactRange")
    db.compactRange()
    println("done")

    handles.asScala.foreach(_.close())
    db.close()
    is.close()
  }

}
